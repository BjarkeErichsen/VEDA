from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)

def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

def gamma(t):
    return 2 * torch.log(t)

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=0.02):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.squeeze(-1)
        x = x.ger((2 * np.pi * self.freqs.to(x.device)).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
class EnElucidatingDM(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics: models.EGNN_dynamics_QM9, in_node_nf: int, n_dims: int,
            timesteps: int = 1000, norm_values=(1., 1., 1.), 
            norm_biases=(None, 0.2, 0.25), include_charges=True, style = 'VE', time_nf = 1, predictor = 'noise_pred', sigma_data = 1, variance_penalty = 0, coordinate_predictor = 'coordinate_error', ot_noise =False,
            max_sigma = 80, min_sigma=0.001):
        super().__init__()

        # The network that will predict the denoising.
        self.dynamics = dynamics
        self.include_charges = include_charges
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))
        
        self.gamma = gamma
        self.max_sigma = max_sigma/norm_values[0]
        self.min_sigma = min_sigma/norm_values[0]
        self.style = style
        self.sigma_data = sigma_data/norm_values[0]
        self.time_nf = time_nf
        if self.time_nf > 1:
            self.time_embedding = FourierEmbedding(num_channels = self.time_nf)
        self.predictor = predictor
        self.variance_penalty = variance_penalty
        self.coordinate_predictor = coordinate_predictor
        self.ot_noise = ot_noise
    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])
        print(f'Checking norm values {self.norm_values} with sigma_0 {sigma_0:.5f}')
        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def phi(self, x, t, node_mask, edge_mask, context):
        gamma_t = self.inflate_batch_array(self.gamma(t), x)        
        sigma_t = self.sigma(gamma_t, x)
        if self.style == 'EDM':
                # c_skip and c_out as defined in the EDM paper
            c_skip = self.sigma_data**2 / (self.sigma_data**2 + sigma_t**2)
            c_out = sigma_t*self.sigma_data / (self.sigma_data**2 + sigma_t**2).sqrt()
            c_in = 1 / (sigma_t**2 + self.sigma_data**2).sqrt()
            c_noise =  0.25 * torch.log(t)
            if self.time_nf > 1:
                c_noise = self.time_embedding(c_noise)
            if self.predictor == 'distance_square_error_v3':
                # c_in = 1 / ((sigma_t**2 + self.sigma_data**2)*24**0.5)
                coordinate_c_out = self.sigma_data*sigma_t / (self.sigma_data**4 + sigma_t**4)**0.25
                coordinate_c_skip = self.sigma_data**2 / (self.sigma_data**4 + sigma_t**4)**0.5
            elif self.predictor == 'distance_square_error_v4' or self.predictor == 'distance_square_error_v5':
                c_out = self.sigma_data*sigma_t / (self.sigma_data**4 + sigma_t**4)**0.25
                c_skip = self.sigma_data**2 / (self.sigma_data**4 + sigma_t**4)**0.5
            net_in = x*c_in
            # print('net_in',net_in[:,:,:self.n_dims].mean().detach(), net_in[:,:,:self.n_dims].std().detach())
            net_out = self.dynamics._forward(c_noise, net_in, node_mask, edge_mask, context)
            
            # print('net_out',net_out[:,:,:self.n_dims].mean().detach(), net_out[:,:,:self.n_dims].std().detach())
            if self.predictor == 'noise_pred':
                dx = net_out*c_out
            elif self.predictor == 'x_pred':
                dx = torch.cat([net_out[:,:,:self.n_dims] - net_in[:,:,:self.n_dims], net_out[:,:,self.n_dims:]], axis=-1)*c_out
            elif self.predictor == 'x_pred_v2' or self.predictor == 'x_pred_v6' or self.predictor == 'distance_error' or self.predictor == 'distance_error_v2' or self.predictor == 'distance_square_error_v4' or self.predictor == 'distance_square_error_v5':
                dx = torch.cat([(net_out[:,:,:self.n_dims] - net_in[:,:,:self.n_dims])*c_out/math.sqrt(2), net_out[:,:,self.n_dims:]*c_out], axis=-1)
            elif self.predictor == 'x_pred_v3':
                res_factor = 1-sigma_t/self.sigma_data
                # c_skip_new = (sigma_t*self.sigma_data + self.sigma_data**2) / (self.sigma_data**2 + sigma_t**2)
                dx = torch.cat([net_out[:,:,:self.n_dims] - net_in[:,:,:self.n_dims]/(res_factor**2+1), net_out[:,:,self.n_dims:]], axis=-1)*c_out
            elif self.predictor == 'x_pred_v4':
                res_factor = 1-sigma_t/self.sigma_data
                dx = torch.cat([(net_out[:,:,:self.n_dims] - res_factor*net_in[:,:,:self.n_dims])/(res_factor**2+1).sqrt(), net_out[:,:,self.n_dims:]], axis=-1)*c_out
            elif self.predictor == 'x_pred_v5':
                dx = (net_out - net_in)*c_out
            elif self.predictor == 'distance_square_error_v3':
                dx = torch.cat([(net_out[:,:,:self.n_dims] - net_in[:,:,:self.n_dims])*coordinate_c_out/math.sqrt(2), net_out[:,:,self.n_dims:]*c_out], axis=-1)
                
            # diffusion_utils.assert_mean_zero_with_mask(dx[:, :, :self.n_dims], node_mask)
            x_pred = x*c_skip + dx
            if self.predictor == 'distance_square_error_v3':
                x_pred = torch.cat([x[:,:,:self.n_dims]*coordinate_c_skip, x[:,:,self.n_dims:]*c_skip], axis=-1) + dx
            # print(f"[{float(t[0,0]):.4f}, "
            #     f"{float(torch.std(net_out[:,:,:self.n_dims][node_mask.squeeze(-1) > 0])):.4f}, "
            #     f"{float(torch.std(net_out[:,:,:self.n_dims][node_mask.squeeze(-1) > 0] - net_in[:,:,:self.n_dims][node_mask.squeeze(-1) > 0])):.4f}, "
            #     f"{float(torch.std(x_pred[:,:,:self.n_dims][node_mask.squeeze(-1) > 0])):.4f}],")

            return x_pred
        elif self.style == 'VE':
            x_normalized = x / (sigma_t**2 + self.sigma_data**2).sqrt()  # to avoid division by zero
            c_noise =  torch.log(t/2)
            net_out = self.dynamics._forward(c_noise, x_normalized, node_mask, edge_mask, context)
            dx = net_out* (sigma_t**2 + self.sigma_data**2).sqrt()
            # diffusion_utils.assert_mean_zero_with_mask(dx[:, :, :self.n_dims], node_mask)
            x_pred = x - sigma_t * dx
            return x_pred
            
    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.exp(gamma/2), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.ones(gamma.shape).to(gamma.device), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)
    
    def print_stats_and_plot(self, variable, name):
        # print(variable)
        if isinstance(variable, torch.Tensor):
            variable = variable.detach().cpu().numpy()
        
        mean = np.mean(variable)
        std = np.std(variable)
        
        print(f"{name} - Mean: {mean:.4f}, Std: {std:.4f}")
    def normalize(self, x, h, node_mask):
        # self.print_stats_and_plot(x[0], 'x')
        x = x / self.norm_values[0]
        # print('x std', torch.std(x))
        # print('x*node_mask std', torch.std(x[node_mask.squeeze(-1) > 0]))
        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]
        # self.print_stats_and_plot(x[node_mask.squeeze(-1) > 0], 'x')
        # self.print_stats_and_plot(h_cat[node_mask.squeeze(-1) > 0], 'h_cat')
        # self.print_stats_and_plot(h_int[node_mask.squeeze(-1) > 0], 'h_int')
        # self.print_stats_and_plot(x, 'x')
        # self.print_stats_and_plot(h_cat, 'h_cat')
        # self.print_stats_and_plot(h_int, 'h_int')
        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, eps):
        """Computes error, i.e. the most likely prediction of x."""
        error = sum_except_batch((eps - net_out) ** 2)/(self.n_dims + self.in_node_nf) / eps.shape[1]
        if self.variance_penalty != 0:
            
            var_pred = net_out.var(dim=[1, 2], unbiased=True)# prediction variance
            var_target = eps.var(dim=[1, 2], unbiased=True)  # target variance

            # Only add penalty for too small variance
            var_penalty = self.variance_penalty * torch.relu(var_pred - var_target)
            # print('error:', error, 'var_penalty:',var_penalty, 'var_target', var_target, 'var_pred', var_pred)
            print('error:', error.mean(), 'var_penalty:',var_penalty.mean())
            error += var_penalty
        
        # print(error)
        return error*self.T
    def compute_distance_matrix(self, net_out):
        """
        Computes the pairwise distance matrix for each batch.
        net_out: Tensor of shape (batch_size, node_num, 3)
        
        Returns:
        dist_matrix: Tensor of shape (batch_size, node_num, node_num)
        """
        # Expand dimensions for broadcasting
        net_out_exp = net_out.unsqueeze(2)  # (batch_size, node_num, 1, 3)
        
        # Compute pairwise squared differences
        dist_matrix = torch.norm(net_out_exp - net_out.unsqueeze(1), dim=-1)  # (batch_size, node_num, node_num)
        
        return dist_matrix
    def compute_distance_square_matrix(self, net_out):
        """
        Computes the pairwise distance matrix for each batch.
        net_out: Tensor of shape (batch_size, node_num, 3)
        
        Returns:
        dist_matrix: Tensor of shape (batch_size, node_num, node_num)
        """
        diff = net_out.unsqueeze(2) - net_out.unsqueeze(1)  # Shape: (batch_size, node_num, node_num, 3)
        # Sum squared differences along the last dimension (avoid sqrt)
        dist_matrix = (diff ** 2).sum(dim=-1)  # Shape: (batch_size, node_num, node_num)
        
        
        
        return dist_matrix
    def compute_distance_square_error(self, net_out, xt, node_mask, threshold = -1, mode = 'soft_threshold'):
        """Computes distance error, i.e. the difference between the predicted and target distances matrix"""
        assert net_out.size() == xt.size()
        dist_sq_matrix_pred = self.compute_distance_square_matrix(net_out)
        dist_sq_matrix_target = self.compute_distance_square_matrix(xt)
        if threshold > 0:
            if mode == 'hard_threshold':
                dist_matrix_target_weight = (dist_sq_matrix_target*self.norm_values[0]**2) < threshold
            elif mode == 'soft_threshold':
                dist_matrix_target_weight = 1 - torch.sigmoid((dist_sq_matrix_target**0.5*self.norm_values[0]) / threshold**0.5)
            dist_sq_matrix_pred = dist_sq_matrix_pred * dist_matrix_target_weight
            dist_sq_matrix_target = dist_sq_matrix_target * dist_matrix_target_weight
        error = torch.norm(dist_sq_matrix_pred - dist_sq_matrix_target, dim=(1,2))
        error = error/(self.n_dims + self.in_node_nf) / node_mask.sum(dim=(1,2))**2
        return error*self.T
    def compute_distance_error(self, net_out, xt, node_mask):
        """Computes distance error, i.e. the difference between the predicted and target distances matrix"""
        assert net_out.size() == xt.size()
        dist_matrix_pred = self.compute_distance_matrix(net_out)
        dist_matrix_target = self.compute_distance_matrix(xt)
        error = torch.norm(dist_matrix_pred - dist_matrix_target, dim=(1,2))
        use_pseudo_huber_distance = False
        if use_pseudo_huber_distance:
            c = 0.01
            error = (error**2 + c**2)**0.5-c
        error = error/(self.n_dims + self.in_node_nf) / node_mask.sum(dim=(1,2))**2

        return error*self.T
    def permute_atoms(self, noise, xt, node_mask):
        
        # noise is [:,:,n_dim+self.in_node_nf+self.include_charge]
        noise_coord = noise[:,:,:self.n_dims]*node_mask
        xt = xt*node_mask
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.transform import Rotation
        diff = noise_coord.unsqueeze(2) - xt.unsqueeze(1)
        cost_matrix = np.array(torch.norm(diff, dim=-1).cpu())
        print(cost_matrix[0].trace())
        batch_size = len(noise)
        new_noise = list()
        for i in range(batch_size):            
            _, indices = linear_sum_assignment(cost_matrix[i])
            print(indices, node_mask[i])
            indices = torch.tensor(indices, device=noise.device)
            noise[i] = noise[i][indices]
            noise_coord[i] = noise_coord[i][indices]
            rotation, _ = Rotation.align_vectors(xt[i].cpu(), noise_coord[i].cpu())
            rotated_noise_coord = rotation.apply(noise_coord[i].cpu())
            new_noise.append(np.concatenate([rotated_noise_coord, noise[i][:,self.n_dims:].cpu()], axis=-1))
        new_noise = np.array(new_noise)
        return torch.tensor(new_noise, device=noise.device, dtype=noise.dtype)


    def compute_classifying_error(self, net_out, xt, node_mask):
        """Computes classifying error, i.e. the cross entropy loss between the predicted and target one-hot encoding of atom types"""
        assert net_out.size() == xt.size()


        # Convert one-hot xt to class indices
        xt_indices = xt.argmax(dim=-1)  # Shape: [batch_size, seq_len]
        
        # Compute cross-entropy (automatically handles numerical stability)
        loss_per_element = F.cross_entropy(
            net_out.view(-1, net_out.size(-1)),  # Flatten to [batch_size*seq_len, num_classes]
            xt_indices.view(-1),                 # Flatten to [batch_size*seq_len]
            reduction='none'
        ).view_as(xt_indices)                    # Reshape back to [batch_size, seq_len]
        
        # Apply mask and average over valid positions
        masked_loss = loss_per_element * node_mask.squeeze(-1)  # Mask shape: [batch_size, seq_len]
        error = masked_loss.sum(dim=1) /(self.n_dims + self.in_node_nf) / node_mask.sum(dim=(1,2))
        return error*self.T
    def compute_classifying_accuracy(self, net_out, xt, node_mask):
        """Computes classifying accuracy between the predicted and target one-hot encoding of atom types"""
        assert net_out.size() == xt.size()
        
        # Convert predictions to class indices
        pred_indices = net_out.argmax(dim=-1)  # Shape: [batch_size, seq_len]
        true_indices = xt.argmax(dim=-1)       # Shape: [batch_size, seq_len]
        
        # Compare predictions with ground truth
        correct = (pred_indices == true_indices).float()  # 1.0 for correct, 0.0 for wrong
        
        # Apply mask (only consider valid positions)
        masked_correct = correct * node_mask.squeeze(-1)
        
        # Calculate accuracy: sum(correct predictions) / sum(masked elements)
        accuracy = masked_correct.sum(dim=1) / node_mask.sum(dim=(1,2))
        
        return accuracy
        
    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def compute_loss(self, x, h, node_mask, edge_mask, context):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""
        loss_dict = {}
        if self.style == 'VE':
            rnd_uniform = torch.rand((x.size(0), 1), device=x.device)
            t = self.min_sigma * ((self.max_sigma / self.min_sigma) ** rnd_uniform)
        elif self.style == 'EDM':
            import math
            rnd_gaussian = torch.randn((x.size(0), 1), device=x.device) * math.log((self.max_sigma / self.min_sigma))
            t =  torch.exp(math.log(((self.max_sigma / self.min_sigma)**0.5) * self.min_sigma) + rnd_gaussian/8) #4 is for +- 2 sigma
            
        # Compute gamma_s and gamma_t via the network. 
        gamma_t = self.inflate_batch_array(self.gamma(t), x) # [64, 1, 1]

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x) # should be [64, 1, 1]
        sigma_t = self.sigma(gamma_t, x)
        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        if self.coordinate_predictor  == 'distance_square_error_v5' or self.ot_noise:
            eps = self.permute_atoms(eps, x, node_mask=node_mask)
            eps = self.permute_atoms(eps, x, node_mask=node_mask)
            eps = self.permute_atoms(eps, x, node_mask=node_mask)

        # print(torch.std(eps[:, :, :self.n_dims][node_mask.squeeze(-1) > 0]))
        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        # print('x.shape: ', x.size(), 'h_categorical.shape: ', h['categorical'].size(), 'h_integer.shape: ', h['integer'].size())
        z = alpha_t * xh + sigma_t * eps
        
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        # Neural net prediction.
        
            
        x_pred = self.phi(z, t, node_mask, edge_mask, context)

        # '''
        # TEMP DEBUG
        # '''        
        # c_skip = self.sigma_data**2 / (self.sigma_data**2 + sigma_t**2)
        # c_out = sigma_t*self.sigma_data / (self.sigma_data**2 + sigma_t**2).sqrt()
        # estimated_y = ((xh - c_skip*z )/c_out)
        # c_in = 1 / (sigma_t**2 + self.sigma_data**2).sqrt()
        # net_in = c_in*z
        # new_estimated_y = estimated_y[0,:,:self.n_dims]-net_in[0,:,:self.n_dims]
        # print('estimated_y:', estimated_y[:,:,:self.n_dims].mean(), estimated_y[:,:,:self.n_dims].std())
        # print('estimated_y[0]:', estimated_y[0,:,:self.n_dims].mean(), estimated_y[0,:,:self.n_dims].std(), sigma_t[0])
        # print('new_estimated_y[0]:', new_estimated_y.mean(), new_estimated_y.std())
        # print('x_pred: ', x_pred[:,:,:self.n_dims].mean(), x_pred[:,:,:self.n_dims].std())
        # print('xh: ', xh[:,:,:self.n_dims].mean(), xh[:,:,:self.n_dims].std())

        # d = (z - x_pred) / sigma_t
        # diffusion_utils.assert_mean_zero_with_mask(d[:, :, :self.n_dims], node_mask)
        # '''
        # END TEMP DEBUG 
        # '''
        
        
        # Compute the error.
        if self.predictor == 'x_pred_v4':
            res_factor = 1-sigma_t/self.sigma_data
            res_factor = res_factor.squeeze(-1).squeeze(-1)
        else:
            res_factor = 0
       
        
        if self.predictor == 'x_pred_v6':
            error = self.compute_error(x_pred[:,:,:self.n_dims], xh[:,:,:self.n_dims]) + self.compute_error(x_pred[:,:,self.n_dims:-1], xh[:,:,self.n_dims:-1])*100 + self.compute_error(x_pred[:,:,-1], xh[:,:,-1])*16
            error = error/10
        elif self.predictor == 'distance_error' or self.coordinate_predictor == 'distance_error':
            error = self.compute_distance_error(x_pred[:,:,:self.n_dims], xh[:,:,:self.n_dims], node_mask) + self.compute_error(x_pred[:,:,self.n_dims:], xh[:,:,self.n_dims:])
        elif "distance_square" in self.coordinate_predictor:
            if self.coordinate_predictor == 'distance_square_error':
                # error = self.compute_distance_square_error(x_pred[:,:,:self.n_dims], xh[:,:,:self.n_dims], node_mask) + self.compute_error(x_pred[:,:,self.n_dims:], xh[:,:,self.n_dims:])
                coordiante_error = self.compute_distance_square_error(x_pred[:,:,:self.n_dims], xh[:,:,:self.n_dims], node_mask)/5
            elif self.coordinate_predictor == 'distance_square_error_v2':
                coordiante_error = self.compute_distance_square_error(x_pred[:,:,:self.n_dims], xh[:,:,:self.n_dims], node_mask, threshold=9, mode = 'hard_threshold')
            elif self.coordinate_predictor == 'distance_square_error_v3' or self.coordinate_predictor == 'distance_square_error_v4' or self.coordinate_predictor == 'distance_square_error_v5':
                coordiante_error = self.compute_distance_square_error(x_pred[:,:,:self.n_dims], xh[:,:,:self.n_dims], node_mask, threshold=9)
            atom_error = self.compute_error(x_pred[:,:,self.n_dims:-1], xh[:,:,self.n_dims:-1])
            charge_error = self.compute_error(x_pred[:,:,-1], xh[:,:,-1])

            error = coordiante_error + atom_error + charge_error
    
        elif self.predictor == 'distance_error_v2':
            assert self.include_charges
            coordiante_error = self.compute_distance_error(x_pred[:,:,:self.n_dims], xh[:,:,:self.n_dims], node_mask)
            atom_error = self.compute_classifying_error(x_pred[:,:,self.n_dims:-1], xh[:,:,self.n_dims:-1], node_mask)
            charge_error = self.compute_error(x_pred[:,:,-1], xh[:,:,-1])
            error = coordiante_error + atom_error + charge_error
        else:
            error = self.compute_error(x_pred[:,:,:self.n_dims], xh[:,:,:self.n_dims])*(res_factor**2+1) + self.compute_error(x_pred[:,:,self.n_dims:], xh[:,:,self.n_dims:])
        if 'coordiante_error' in locals():
            atom_accuracy = self.compute_classifying_accuracy(x_pred[:,:,self.n_dims:-1], xh[:,:,self.n_dims:-1], node_mask)
            print('coordiante_error: ', coordiante_error.mean(), 'atom_error: ', atom_error.mean(), 'charge_error: ', charge_error.mean(), 'atom_accuracy: ', atom_accuracy.mean())




        if self.style == 'VE':
            SNR_weight = (1 / (sigma_t ** 2)).squeeze(-1).squeeze(-1)
        elif self.style == 'EDM':
            SNR_weight = ((sigma_t ** 2 + self.sigma_data**2) / (sigma_t * self.sigma_data) ** 2).squeeze(-1).squeeze(-1)
        if self.predictor == 'distance_square_error_v3':
            c_out = self.sigma_data*sigma_t/(self.sigma_data**4 + sigma_t**4)**0.25
            coordinate_SNR_weight = (1/c_out**2).squeeze(-1).squeeze(-1)
        elif self.predictor == 'distance_square_error_v4' or self.predictor == 'distance_square_error_v5':
            c_out = self.sigma_data*sigma_t/(self.sigma_data**4 + sigma_t**4)**0.25
            SNR_weight = (1/c_out**2).squeeze(-1).squeeze(-1)
        # print('SNR_weight: ', SNR_weight)
        
        # print('error: ', error.size(), 'SNR_weight: ', SNR_weight.size())
        assert error.size() == SNR_weight.size()
        loss_t = 0.5 * SNR_weight * error
        if self.predictor == 'distance_error_v2':
            loss_t = 0.5* (SNR_weight * (coordiante_error + charge_error) + atom_error)
        if self.predictor == 'distance_square_error_v3':
            loss_t = 0.5* (coordinate_SNR_weight * coordiante_error + SNR_weight * (atom_error + charge_error))
        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].

        neg_log_constants = torch.zeros_like(loss_t)

        loss =  loss_t + neg_log_constants #+ loss_term_0
        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'
        # loss_dict = {'t': t.squeeze(), 'loss_t': loss.squeeze(), 'error': error.squeeze()}
        loss_dict['t'] = t.squeeze()
        loss_dict['loss_t'] = loss.squeeze()
        loss_dict['error'] = error.squeeze()
        
        return loss, loss_dict

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h = self.normalize(x, h, node_mask)
        loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context)

        return loss, loss_dict

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, use_heun=False, sampler=None, alpha=1):
        """
        Draw samples from the generative model.
        """

        # print(self.predictor)
        if sampler is None:
            sampler = self.style
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask) * self.max_sigma#* math.sqrt(self.max_sigma**2+1.7)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask) * self.max_sigma#* math.sqrt(self.max_sigma**2+1.7)
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        if sampler == 'dpmsolver':
            from sampling_config import edm_dpm_solver_config
            config = edm_dpm_solver_config
            edm_dpm_solver_config['steps'] = self.T
            def model_dpmsolver(z, t):
                # 注意：这里将 t 归一化处理（例如除以 1000），具体归一化方式请根据实际情况调整
                # t = torch.full((n_samples, 1), t, device=z.device)
                t = t.unsqueeze(-1)
                return self.phi(z, t, node_mask, edge_mask, context)
            from equivariant_diffusion.sampler.dpm_solver_pytorch import NoiseScheduleEDM, edm_model_wrapper, DPM_Solver
            # 定义离散时间的噪声调度，这里采用 polynomial_schedule，具体参数可根据需求调整
            noise_schedule = NoiseScheduleEDM()

            # 将离散时间的模型转换为连续时间的噪声预测模型，类型为 "noise"
            model_fn = edm_model_wrapper(
                model_dpmsolver,
                noise_schedule
            )
            

            # 初始化 DPM-Solver，这里使用 "dpmsolver++" 算法（也可试试 "dpmsolver"）
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++", correcting_x0_fn=None)

            # 使用 DPM-Solver 进行采样

            z = dpm_solver.sample(                
                z,
                t_start=self.max_sigma,
                t_end=self.min_sigma,
                **config
            )
        elif sampler == 'VE':
            sigmas = torch.exp(torch.linspace(np.log(self.max_sigma), np.log(self.min_sigma), self.T+1))
        elif sampler == 'EDM':
            # rho = 7
            # step_indices = torch.arange(self.T, dtype=torch.float64)
            # sigmas = (self.max_sigma ** (1 / rho) + step_indices / (self.T - 1) * (self.min_sigma ** (1 / rho) - self.max_sigma ** (1 / rho))) ** rho
            # sigmas = torch.concat([sigmas, torch.tensor([0])])
            sigmas = torch.exp(torch.linspace(np.log(self.max_sigma), np.log(self.min_sigma), self.T+1))
        elif sampler == 'CM':
            t = torch.full((n_samples, 1), self.max_sigma, device=z.device)
            z = self.phi(z, t, node_mask, edge_mask, context)
            # rho = 7
            # step_indices = torch.arange(self.T, dtype=torch.float64)
            # sigmas = (self.max_sigma ** (1 / rho) + step_indices / (self.T-1) * (self.min_sigma ** (1 / rho) - self.max_sigma ** (1 / rho))) ** rho
            sigmas = torch.exp(torch.linspace(np.log(self.max_sigma), np.log(self.min_sigma), self.T))
        elif sampler == 'test':            
            # def generate_sigmas(i, max_sigma, min_sigma):
            #     log_max = np.log(max_sigma)
            #     log_min = np.log(min_sigma)
            #     delta_log = (log_min - log_max) / 5  # 将整个log区间分为5段
            #     segments = []
                
            #     for j in range(5):
            #         start = log_max + j * delta_log
            #         end = start + delta_log
            #         # 生成当前分段的log空间点
            #         if j == i:
            #             # 第i个分段生成200个区间步（201个点）
            #             segment = torch.linspace(start, end, 200 + 1)
            #         else:
            #             # 其他分段生成6个区间步（7个点）
            #             segment = torch.linspace(start, end, 6 + 1)
            #         # 除第一个分段外，其他分段的第一个点与前一个分段的最后一个点重复，需去掉
            #         if j == 0:
            #             segments.append(segment)
            #         else:
            #             segments.append(segment[1:])  # 去掉第一个点
                
            #     # 合并所有分段的点并转换为原始空间
            #     log_sigmas = torch.cat(segments)
            #     sigmas = torch.exp(log_sigmas)
            #     return sigmas
            # sigmas = generate_sigmas(2, self.max_sigma, self.min_sigma)
            rho = 2.5
            step_indices = torch.linspace(0, 1, self.T+1)
            
            # 使用cos变换增强中间区域密度
            transformed = (torch.arccos(1-step_indices*2)) / np.pi
            
            # 混合原始线性插值与cos变换结果
            mixed = (1-rho)*step_indices + rho*transformed
            mixed /= mixed.max()  # 保持[0,1]范围
            
            # 对数空间插值d
            log_sigmas = np.log(self.max_sigma) + mixed * (np.log(self.min_sigma) - np.log(self.max_sigma))
            sigmas = torch.exp(log_sigmas)
        elif sampler == 'test_CM':
            t = torch.full((n_samples, 1), self.max_sigma, device=z.device)
            z = self.phi(z, t, node_mask, edge_mask, context)
            rho = 2
            step_indices = torch.linspace(0, 1, self.T)
            transformed = (torch.arccos(1-step_indices*2)) / np.pi
            mixed = (1-rho)*step_indices + rho*transformed
            mixed /= mixed.max()
            log_sigmas = np.log(self.max_sigma) + mixed * (np.log(self.min_sigma) - np.log(self.max_sigma))
            sigmas = torch.exp(log_sigmas)
            
        if sampler in ['VE', 'EDM','CM', 'test', 'test_CM']:
            for i, (sigma_t, sigma_tplus1) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1
                
                # print(i, (sigma_t**2+self.sigma_data**2*1.7).sqrt(), torch.std(z[:, :, :self.n_dims][node_mask.squeeze(-1) > 0]))
                # z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

                # diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
                # diffusion_utils.assert_mean_zero_with_mask(d[:, :, :self.n_dims], node_mask)
                # update z steps
                if sampler == 'VE':
                    t = torch.full((n_samples, 1), sigma_t, device=z.device)
                    x_pred = self.phi(z, t, node_mask, edge_mask, context)
                    d = (z - x_pred) / sigma_t
                    d = torch.cat(
                        [diffusion_utils.remove_mean_with_mask(d[:, :, :self.n_dims], node_mask),
                        d[:, :, self.n_dims:]], dim=2)
                    z = z - d * (sigma_t - sigma_tplus1)  # sigma_t.next is next noise level
                    z = self.sample_normal(z, torch.sqrt(sigma_t**2 - sigma_tplus1**2)*0.5, node_mask, fix_noise)
                elif sampler == 'EDM' or sampler == 'test':
                    # if i == len(sigmas)//3*2:
                    #     sampler = 'CM'
                    #     t = torch.full((n_samples, 1), sigma_t, device=z.device)
                    #     z = self.phi(z, t, node_mask, edge_mask, context)
                        # continue
                    S_chrun = 100
                    gamma = min(S_chrun/self.T, 0.4)
                    gamma = 1 + gamma if 0.03/self.norm_values[0] <= sigma_t <= 3/self.norm_values[0] else 1
                    # gamma = 1.1 if 0.0001 <= sigma_t <= 1 else 1
                    sigma_hat = sigma_t * gamma
                    # print(i,sigma_t, sigma_hat)
                    t_hat = torch.full((n_samples, 1), sigma_hat, device=z.device)
                    z = self.sample_normal(z, torch.sqrt(sigma_hat**2 - sigma_t**2), node_mask, fix_noise)
                    x_pred = self.phi(z, t_hat, node_mask, edge_mask, context)

                    d = (x_pred - z) / sigma_hat*alpha

                    # d_bias_weight = torch.zeros(size=d.shape, device=d.device)
                    # d_bias_weight[:, :, :self.n_dims] = 2
                    # d_bias_weight[:, :, self.n_dims:] = 1
                    # d*= d_bias_weight
                    # d = (x_pred - z) / sigma_hat*sigma_hat/(sigma_hat - sigma_tplus1)
                    z_new = z + d * (sigma_hat - sigma_tplus1)  # sigma_t.next is next noise level
                        # Project down to avoid numerical runaway of the center of gravity.
                    if use_heun and sigma_tplus1 != 0 and i != self.T-1 and i != 0:
                        tplus1 = torch.full((n_samples, 1), sigma_tplus1, device=z.device)
                        d_hat = (self.phi(z_new, tplus1, node_mask, edge_mask, context) - z_new)/sigma_tplus1*alpha#It should be z_new
                        z = z + (sigma_hat - sigma_tplus1) * (0.5*d + 0.5*d_hat)
                    else:
                        z = z_new
                elif sampler == 'CM' or sampler == 'test_CM':
                    t = torch.full((n_samples, 1), sigma_tplus1, device=z.device)
                    z = self.sample_normal(z, torch.sqrt(sigma_tplus1**2), node_mask, fix_noise)
                    z = self.phi(z, t, node_mask, edge_mask, context)*alpha - z*(alpha-1)
                    # print(i, (sigma_tplus1+self.sigma_data**2), torch.std(z[:, :, :self.n_dims]))
                z = torch.cat(
                    [diffusion_utils.remove_mean_with_mask(z[:, :, :self.n_dims],
                                                        node_mask),
                    z[:, :, self.n_dims:]], dim=2)
                
                # print(i, math.sqrt((sigma_tplus1**2+self.sigma_data**2*1.7)), 
                #       torch.std(z[:, :, :self.n_dims][node_mask.squeeze(-1) > 0]), 
                #       torch.std(z[:, :, self.n_dims:][node_mask.squeeze(-1) > 0]), 
                #       torch.std(d[:, :, :self.n_dims][node_mask.squeeze(-1) > 0]), 
                #       torch.std(d[:, :, self.n_dims:][node_mask.squeeze(-1) > 0]))

                # print(i, torch.std(z[:, :, :self.n_dims]))
                # print(i, torch.std(x_pred[:,:,:self.n_dims]))
                
            
        x = z[:, :, :self.n_dims]
        h_int = z[:, :, -1:] if self.include_charges else torch.zeros(0).to(z.device)
        x, h_cat, h_int = self.unnormalize(x, z[:, :, self.n_dims:-1], h_int, node_mask)
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)
        return x, h
    
    
class EnConsistency(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics: models.EGNN_dynamics_QM9, in_node_nf: int, n_dims: int,
            timesteps: int = 1000, norm_values=(1., 1., 1.), 
            norm_biases=(None, 0.2, 0.25), include_charges=True, style = 'VE', time_nf = 1, predictor= 'noise_pred', sigma_data = 1.4):
        super().__init__()

        # The network that will predict the denoising.
        self.dynamics = dynamics
        self.include_charges = include_charges
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))
        
        self.gamma = gamma
        self.max_sigma = 80/norm_values[0]
        self.min_sigma = 1e-3/norm_values[0]
        self.style = style
        self.sigma_data = sigma_data/norm_values[0]
        self.time_nf = time_nf
        if self.time_nf > 1:
            self.time_embedding = FourierEmbedding(num_channels = self.time_nf)
        self.predictor = predictor
    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])
        print(f'Checking norm values {self.norm_values} with sigma_0 {sigma_0:.5f}')
        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def phi(self, x, t, node_mask, edge_mask, context):
        gamma_t = self.inflate_batch_array(self.gamma(t), x)        
        sigma_t = self.sigma(gamma_t, x)
        epsilon = self.min_sigma
        # c_skip and c_out as defined in the EDM paper
        c_skip = self.sigma_data**2 / (self.sigma_data**2 + (sigma_t-epsilon)**2)
        c_out = (sigma_t-epsilon)*self.sigma_data / (self.sigma_data**2 + sigma_t**2).sqrt()
        c_in = 1 / (sigma_t**2 + self.sigma_data**2).sqrt()
        c_noise =  0.25 * torch.log(t)
        net_in = x*c_in
        # print('net_in',net_in[:,:,:self.n_dims].mean().detach(), net_in[:,:,:self.n_dims].std().detach())
        net_out = self.dynamics._forward(c_noise, net_in, node_mask, edge_mask, context)
        # print('net_out',net_out[:,:,:self.n_dims].mean().detach(), net_out[:,:,:self.n_dims].std().detach())
        if self.predictor == 'noise_pred':
            dx = net_out*c_out
        elif self.predictor == 'x_pred':
            dx = torch.cat([(net_out[:,:,:self.n_dims] - net_in[:,:,:self.n_dims])/math.sqrt(2), net_out[:,:,self.n_dims:]], axis=-1)*c_out 
        # diffusion_utils.assert_mean_zero_with_mask(dx[:, :, :self.n_dims], node_mask)
        x_pred = x*c_skip + dx
        return x_pred
            
    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.exp(gamma/2), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.ones(gamma.shape).to(gamma.device), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)
    

    def print_stats_and_plot(self, variable, name):
        # print(variable)
        if isinstance(variable, torch.Tensor):
            variable = variable.detach().cpu().numpy()
        
        mean = np.mean(variable)
        std = np.std(variable)
        
        print(f"{name} - Mean: {mean:.4f}, Std: {std:.4f}")
    def normalize(self, x, h, node_mask):
        # self.print_stats_and_plot(x[0], 'x')
        x = x / self.norm_values[0]

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]
        # self.print_stats_and_plot(x[0], 'x')
        # self.print_stats_and_plot(h_cat[0], 'h_cat')
        # self.print_stats_and_plot(h_int[0], 'h_cat')

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int



    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, eps):
        """Computes error, i.e. the most likely prediction of x."""
        # error = sum_except_batch((eps - net_out) ** 2)/(self.n_dims + self.in_node_nf) / eps.shape[1]
        c = 0.01
        error = sum_except_batch((eps - net_out) ** 2)/(self.n_dims + self.in_node_nf) / eps.shape[1]
        error = (error+c**2).sqrt() - c
        # print(error)
        return error
    def compute_cat_error(self, net_out, eps):
        """Computes the cross-entropy error between predicted and true labels."""
        # Assuming net_out is logits, convert them to probabilities using softmax
        # log_probs = F.log_softmax(net_out, dim=-1)
        # # Compute cross-entropy loss
        # error = -sum_except_batch(eps * log_probs) /(self.n_dims + self.in_node_nf) / eps.shape[1]  # Average loss over batch
        error = sum_except_batch((eps - net_out) ** 2)/(self.n_dims + self.in_node_nf) / eps.shape[1]-\
            sum_except_batch(((1-eps) - net_out) ** 2)/(self.n_dims + self.in_node_nf) / eps.shape[1]
        return error
    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def compute_loss(self, x, h, node_mask, edge_mask, context):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""
        import math
        rnd_gaussian = torch.randn((x.size(0), 1), device=x.device) * math.log((self.max_sigma / self.min_sigma))
        t =  torch.exp(math.log(((self.max_sigma / self.min_sigma)**0.5) * self.min_sigma) + rnd_gaussian/8) #4 is for +- 2 sigma
        # Compute gamma_s and gamma_t via the network. 
        gamma_t = self.inflate_batch_array(self.gamma(t), x) # [64, 1, 1]

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x) # should be [64, 1, 1]
        sigma_t = self.sigma(gamma_t, x)
        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)



        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        z = alpha_t * xh + sigma_t * eps
        
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        # Neural net prediction.
        
        x_pred = self.phi(z, t, node_mask, edge_mask, context)
        
        error = self.compute_error(x_pred, z)
        SNR_weight = ((sigma_t ** 2 + self.sigma_data**2) / (sigma_t * self.sigma_data) ** 2).squeeze(-1).squeeze(-1)

        assert error.size() == SNR_weight.size()
        loss = 0.5 * SNR_weight * error

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'
        return loss

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None, n = None, eps=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h = self.normalize(x, h, node_mask)
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        t = torch.full((x.size(0), 1), self.sigmas(self.T)[n], device=xh.device)
        gamma_t = self.inflate_batch_array(self.gamma(t), x) # [64, 1, 1]
        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x) # should be [64, 1, 1]
        sigma_t = self.sigma(gamma_t, x)
        z = alpha_t * xh + sigma_t * eps
        print(self.sigmas(self.T)[n])
        output = self.phi(z, t, node_mask, edge_mask, context)
        return output

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sigmas(self, N):
        rho = 7
        step_indices = torch.arange(N, dtype=torch.float64)
        sigmas = (self.max_sigma ** (1 / rho) + step_indices / (N - 1) * (self.min_sigma ** (1 / rho) - self.max_sigma ** (1 / rho))) ** rho
        #Here sigmas[0] > sigmas[1] > ...> sigmas[N-1]
        return sigmas

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, use_heun=False, sampler=None):
        """
        Draw samples from the generative model.
        """
        # print(self.predictor)
        if sampler is None:
            sampler = self.style
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask) * self.max_sigma
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask) * self.max_sigma

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        
        
        t = torch.full((n_samples, 1), self.max_sigma, device=z.device)
        z = self.phi(z, t, node_mask, edge_mask, context)
        sigmas = self.sigmas(self.T)
        # sigmas = torch.exp(torch.linspace(np.log(self.max_sigma), np.log(self.min_sigma), self.T))
        for i, (_, sigma_tplus1) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1  
            # z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

            # diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
            # diffusion_utils.assert_mean_zero_with_mask(d[:, :, :self.n_dims], node_mask)
            # update z steps           

            # print(i,sigma_tplus1)
            t = torch.full((n_samples, 1), sigma_tplus1, device=z.device)
            z = self.sample_normal(z, torch.sqrt(sigma_tplus1**2 - self.min_sigma**2), node_mask, fix_noise)
            z = self.phi(z, t, node_mask, edge_mask, context)
            # z = torch.cat(
            #     [diffusion_utils.remove_mean_with_mask(z[:, :, :self.n_dims],
            #                                         node_mask),
            #     z[:, :, self.n_dims:]], dim=2)
            z = torch.cat(
                [diffusion_utils.remove_mean_with_mask(z[:, :, :self.n_dims],
                                                    node_mask),
                z[:, :, self.n_dims:]], dim=2)
            
        x = z[:, :, :self.n_dims]
        h_int = z[:, :, -1:] if self.include_charges else torch.zeros(0).to(z.device)
        x, h_cat, h_int = self.unnormalize(x, z[:, :, self.n_dims:-1], h_int, node_mask)
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}
        # # Finally sample p(x, h | z_0).


        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)
        return x, h
