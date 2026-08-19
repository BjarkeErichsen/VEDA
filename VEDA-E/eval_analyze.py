# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from qm9 import dataset
from qm9.models import get_model
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules, analyze_node_distribution
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses
import json
try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')

from sampling_config import dpm_solver_config, ddim_config
from qm9.analyze import save_sdf_file, compute_fcd_score, compute_energy, compute_rmsd, compute_strain_energy

def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample, 
            sampler=eval_args.sampler, use_heun=eval_args.use_heun, alpha=eval_args.alpha)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        if save_to_xyz:
            id_from = i * batch_size
            qm9_visualizer.save_xyz_file(
                join(eval_args.model_path, 'eval/analyzed_molecules/'),
                one_hot, charges, x, dataset_info, id_from, name='molecule',
                node_mask=node_mask)

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    os.makedirs(join(eval_args.model_path, 'eval/analyzed_molecules/'), exist_ok=True)
    torch.save(molecules, join(eval_args.model_path, 'eval/analyzed_molecules/molecules.pt'))
    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info)
    if eval_args.compute_fcd:
        compute_fcd_score(molecules, dataset_info)
    if eval_args.compute_energy:
        compute_energy(molecules, dataset_info)
        compute_rmsd(molecules, dataset_info)
        compute_strain_energy(molecules, dataset_info)
    if eval_args.save_to_sdf:
        sdf_path = join(eval_args.model_path, eval_args.model_file.split('.')[0] +'_' +'eval')
        if not os.path.exists(sdf_path):
            os.makedirs(sdf_path)
        save_sdf_file(
            join(sdf_path, 'generated_mol.sdf'), 
            molecules, dataset_info)
    return stability_dict, rdkit_metrics


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test', num_passes=1):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    for pass_number in range(num_passes):
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get data
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

                batch_size = x.size(0)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                h = {'categorical': one_hot, 'integer': charges}

                if len(args.conditioning) > 0:
                    context = prepare_context(args.conditioning, data).to(device, dtype)
                    assert_correctly_masked(context, node_mask)
                else:
                    context = None

                # transform batch through flow
                nll, _, _, loss_dict = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask,
                                                        edge_mask, context)
                # standard nll from forward KL
                if 'coordiante_error' in loss_dict:
                    coordiante_error_epoch += loss_dict['coordiante_error'].item() * batch_size
                    atom_error_epoch += loss_dict['atom_error'].item() * batch_size
                    charge_error_epoch += loss_dict['charge_error'].item() * batch_size
                    atom_accuracy_epoch += loss_dict['atom_accuracy'].item() * batch_size
                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")
            if 'coordiante_error' in loss_dict:
                print(f"coordiante_error: {coordiante_error_epoch/n_samples:.2f}, atom_error: {atom_error_epoch/n_samples:.2f}, charge_error: {charge_error_epoch/n_samples:.2f}, atom_accuracy: {atom_accuracy_epoch/n_samples:.2f}")

    return nll_epoch/n_samples
def generate_log_filename(model_path, sampler):
    """
    生成日志文件名，格式为 eval_log_{sampler}_{id}.txt。
    如果 sampler 为 None, 则使用默认文件名 eval_log.txt。
    """
    if sampler is None:
        return join(model_path, 'eval_log.txt')

    base_name = f'eval_log_{sampler}'
    file_id = 1
    while True:
        filename = join(model_path, f'{base_name}_{file_id}.txt')
        if not os.path.exists(filename):
            return filename
        file_id += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--model_file', type=str, default="generative_model_ema.npy",
                        help='Specify model name')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to the processed GEOM-Drugs .npy file')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify batch size for generation')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')
    parser.add_argument('--sampler', type=str, default=None,
                        help='Specify model sampler, e.g. dpm-solver, ddim')
    parser.add_argument('--skip_nll', type=eval, default=False, help='Skip NLL evaluation')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps')
    parser.add_argument('--save_to_sdf', type=str, default=False, help='Save generated molecules to sdf file')
    parser.add_argument('--use_heun', type=eval, default=False, help='Use Heun integration')
    parser.add_argument('--alpha', type=float, default=1, help='')
    parser.add_argument('--compute_fcd', type=eval, default=False, help='compute fcd score')
    parser.add_argument('--compute_energy', type=eval, default=False, help='compute energy')
    eval_args, unparsed_args = parser.parse_known_args()
    print(eval_args)
    assert eval_args.model_path is not None
    # Set the number of steps for the sampler

    # if eval_args.sampler is not None and eval_args.steps is not None:
    #     if eval_args.sampler == 'ddim':
    #         ddim_config['steps'] = eval_args.steps
    #     elif eval_args.sampler == 'dpmsolver':
    #         dpm_solver_config['steps'] = eval_args.steps
    #     else:       
    #         raise ValueError(f"Invalid sampler: {eval_args.sampler}")

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    if eval_args.data_file is not None:
        args.data_file = eval_args.data_file
            
    if hasattr(args, 'use_representation'):
        args.use_representation = False
        
    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)
    args.time_nf = getattr(args, 'time_nf', 1)
    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    # Load model
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)
    if eval_args.use_heun:
        generative_model.T = eval_args.steps//2+1
    else:
        generative_model.T = eval_args.steps
    # fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    fn = eval_args.model_file
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    
    # Add this line to remove the representation layers from the state_dict
    flow_state_dict = {k: v for k, v in flow_state_dict.items() if "representation" not in k}

    generative_model.load_state_dict(flow_state_dict)

    # Analyze stability, validity, uniqueness and novelty
    stability_dict, rdkit_metrics = analyze_and_save(
        args, eval_args, device, generative_model, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz)
    print(stability_dict)

    
    # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    if args.dataset == 'geom':
        val_name = 'val'
        num_passes = 1
    else:
        val_name = 'valid'
        num_passes = 5
    if not eval_args.skip_nll:
        # Evaluate negative log-likelihood for the validation and test partitions
        val_nll = test(args, generative_model, nodes_dist, device, dtype,
                    dataloaders[val_name],
                    partition='Val')
        print(f'Final val nll {val_nll}')
        test_nll = test(args, generative_model, nodes_dist, device, dtype,
                        dataloaders['test'],
                        partition='Test', num_passes=num_passes)
        print(f'Final test nll {test_nll}')
    
    if rdkit_metrics is not None:
        rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")
    if not eval_args.skip_nll:
        print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
        
    # log_filename = generate_log_filename(eval_args.model_path, eval_args.sampler)
    # with open(log_filename, 'w') as f:
    #     print(f'Number of samples: {eval_args.n_samples}', file=f)
    #     if not eval_args.skip_nll:
    #         print(f'Overview: val nll {val_nll} test nll {test_nll}',file=f)
    #     print(stability_dict, file=f)
    #     if rdkit_metrics is not None:
    #         print(f"Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]), file=f)
    #         # 读取 sampler_config.json 并写入日志
    #     if eval_args.sampler is not None:
    #         print(eval_args.sampler, file=f)
    #         if eval_args.sampler == 'dpmsolver':
    #             print(json.dumps(dpm_solver_config), file=f)
    #         elif eval_args.sampler == 'ddim':
    #             print(json.dumps(ddim_config), file=f)
if __name__ == "__main__":
    main()
