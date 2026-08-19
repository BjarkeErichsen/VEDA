import torch
import numpy as np

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context, generative_model_ema=None, epoch = 0):
    loss_dict = {}
    bs, n_nodes, n_dims = x.size()
    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = generative_model(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    elif args.probabilistic_model == 'elucidating_dm':
        nll, loss_dict = generative_model(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    elif args.probabilistic_model == 'consistency':
        easy_CM = False
        use_denoising = False
        smooth_steps = True
        import math,random
        if args.steps_scheduler =='adaptive':
            s0 = 10
            s1 = args.diffusion_steps
            N = math.ceil(math.sqrt(epoch/args.n_epochs*(s1**2 -s0**2)+s0**2)- 1) + 1
        elif args.steps_scheduler =='adaptive_half':
            N = min(args.diffusion_steps, math.ceil( math.sqrt(epoch*2/args.n_epochs*(args.diffusion_steps**2 -2**2)+2**2)- 1) + 1)
        elif args.steps_scheduler =='improved':
            s0 = 10
            s1 = args.diffusion_steps
            k = epoch
            if smooth_steps:
                K_pi = math.ceil(args.n_epochs/(math.log2(math.ceil(s1/s0))+1))
                N = math.ceil(min(s0*2**(k/K_pi), s1) + 1)
            else:
                N = min(s0*2**(math.ceil(k/K_pi)), s1) + 1

        elif args.steps_scheduler =='fix':
            N = args.diffusion_steps
        generative_model.T = N
        generative_model_ema.T = N
        # print(generative_model.T)
        sigmas = generative_model.sigmas(N)
        if args.steps_scheduler in ['adaptive', 'adaptive_half','fix']:
            n = random.randint(0, N-2)
        elif args.steps_scheduler =='improved':
            prob_n = torch.erf((torch.log(sigmas[:-1])+1.1)/(math.sqrt(2)*2)) - torch.erf((torch.log(sigmas[1:])+1.1)/(math.sqrt(2)*2))
            prob_n = prob_n/torch.sum(prob_n)
            n = np.random.choice(N-1, p=prob_n)
        eps = generative_model.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
        output_student = generative_model(x, h, node_mask, edge_mask, context, n, eps)
        
        if easy_CM:
            k = random.randint(1, N-1-n)
        else:
            k = 1
        if args.steps_scheduler in ['adaptive', 'adaptive_half','fix']:
            output_teacher = generative_model_ema(x, h, node_mask, edge_mask, context, n+k, eps).detach()
        elif args.steps_scheduler == 'improved':
            output_teacher = generative_model(x, h, node_mask, edge_mask, context, n+k, eps).detach()
        weight = 1/(sigmas[n]-sigmas[n+k])
        use_edm_weight = False
        if use_edm_weight:
            weight = (sigmas[n] ** 2 + generative_model.sigma_data**2).sqrt()*(sigmas[n+1] ** 2 + generative_model.sigma_data**2).sqrt() / (sigmas[n]*sigmas[n+1] * generative_model.sigma_data ** 2).squeeze(-1).squeeze(-1)
            consistency_loss = generative_model.compute_error(output_student, output_teacher) * weight
        else:
            consistency_loss = generative_model.compute_error(output_student, output_teacher) * weight * 100
        consistency_loss = consistency_loss.mean(0)

        reg_term = torch.tensor([0.]).to(consistency_loss.device)
        mean_abs_z = 0.
        # Linearly decrease denoising loss weight from 1 to 0 over training
        nll = consistency_loss
        
        if use_denoising:
            denoising_weight = max(0.0, 1.0 - 2*epoch/args.n_epochs)
            if denoising_weight > 0:
                denoising_loss = generative_model.compute_loss(x, h, node_mask, edge_mask, context)
                denoising_loss = denoising_loss.mean(0)                
                nll = nll + denoising_weight * denoising_loss * 10
                print(denoising_weight,denoising_loss,nll)
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z, loss_dict
