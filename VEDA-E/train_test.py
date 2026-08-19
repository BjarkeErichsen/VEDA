import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
from qm9.analyze import save_sdf_file, compute_fcd_score

def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, scheduler=None):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    # 定义一个列表，用于存储钩子的句柄
    hooks = []
    # 定义一个全局字典，用于存储每个模块的输出
    module_outputs = {}
    # 定义一个钩子函数，用于打印每一层的输出维度
    def hook_fn(module_name):
        def hook(module, input, output):
            # print(f"Module: {module_name} ({module.__class__.__name__})")
            # print(f"Epoch: {epoch}, Batch: {i}")  # 打印当前 epoch 和 batch
            # print(f"Input shapes: {[x.shape if isinstance(x, torch.Tensor) else [y.shape for y in x] for x in input]}")
            # print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else [y.shape for y in output]}")
            module_outputs[module_name] = output[0] if isinstance(output, tuple) else output  # 提取并存储到字典中
            # print("-" * 50)            
        return hook
    # 注册钩子，只对 EquivariantBlock 类型的模块
    for name, module in model_dp.named_modules():
        from egnn.egnn_new import EquivariantBlock
        if isinstance(module, EquivariantBlock) or name == 'dynamics.egnn.embedding':  # 只关注 EquivariantBlock
            hook_handle = module.register_forward_hook(hook_fn(name))  # 传递模块名称
            hooks.append(hook_handle)  # 存储钩子句柄
            # module.register_forward_hook(hook_fn_print(name))
            
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()
                        
        # transform batch through flow
        nll, reg_term, mean_abs_z, loss_dict = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, node_mask, edge_mask, context, model_ema, epoch)
        loss = nll + args.ode_regularization * reg_term
        # print('module_outputs')
        # print(module_outputs.keys())
        # print(module_outputs['dynamics.egnn.e_block_8'].shape)
        # print(f"Requires grad: {module_outputs['dynamics.egnn.e_block_8'].requires_grad}")
        # print(module_outputs['dynamics.egnn.e_block_8'].reshape((args.batch_size, -1, args.nf)).shape)
        # print(data['atom_representation'].shape)
        if args.use_representation:
            cosine_sim = 0
            for idx, key in enumerate(module_outputs.keys()):
                if idx >= len(model_dp.atom_representation_proj):
                    break
                proj_layer = model_dp.atom_representation_proj[idx]
                output = proj_layer(module_outputs[key].reshape((x.size(0), -1, args.nf)))
                representation_mask = (h['integer']>=6)
                # print(f"Output shape: {output.shape}, Representation shape: {data['atom_representation'].shape}, Mask shape: {representation_mask.shape}")
                cosine_sim += torch.nn.functional.cosine_similarity(output*representation_mask, data['atom_representation'].to(device)*representation_mask, dim=-1)
                # cosine_sim += torch.nn.functional.cosine_similarity(output, data['atom_representation'].to(device), dim=-1)
            cosine_sim = cosine_sim.mean()  # mean over layers
            # print(f"Representation cosine similarity: {cosine_sim.item()}")
            wandb.log({"repr loss": -cosine_sim}, commit=False)
            # standard nll from forward KL
            loss -= cosine_sim*getattr(args, 'representation_loss_weight', 1)
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
        nll_epoch.append(nll.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (i == 0):
            analyze_and_save(epoch, model_ema, nodes_dist, args, device,
                                            dataset_info, prop_dist, n_samples=args.n_stability_samples)
            if dataset_info['name'] == 'geom' and scheduler is not None:
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]}, commit=True)
        #     start = time.time()
        #     if len(args.conditioning) > 0:
        #         save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
        #     save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
        #                           batch_id=str(i))
        #     sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
        #                                     prop_dist, epoch=epoch)
        #     print(f'Sampling took {time.time() - start:.2f} seconds')

        #     # vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
        #     # vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
        #     # if len(args.conditioning) > 0:
        #     #     vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
        #     #                         wandb=wandb, mode='conditional')
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if dataset_info['name'] == 'geom' and scheduler is not None:
            scheduler.step()
            
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    print("Epoch NLL: ", np.mean(nll_epoch))

    # 在 train_epoch 结束时移除钩子
    for hook in hooks:
        hook.remove()  # 移除钩子


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _, loss_dict = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)
    print(validity_dict)
    if 'qm9' in dataset_info['name']:
        if args.compute_fcd:
            fcd_score = compute_fcd_score(molecules, dataset_info)
            wandb.log({'fcd_score':fcd_score})
    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def analyze_and_save_sampler(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100, sample_steps = 100, sampler=None):
    print(f'Analyzing molecule stability at epoch {epoch}..., with sample_steps = {sample_steps}')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    original_sample_steps = model_sample.T
    model_sample.T = sample_steps
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)
    print(validity_dict)
    if args.compute_fcd and 'qm9' in dataset_info['name']:
        fcd_score = compute_fcd_score(molecules, dataset_info)
        wandb.log({'fcd_score'+str(sample_steps):fcd_score})
    # wandb.log(validity_dict)
    wandb.log({'mol_stable'+str(sample_steps): validity_dict['mol_stable'],'atm_stable'+str(sample_steps): validity_dict['atm_stable'],})
    if rdkit_tuple is not None:
        wandb.log({'Validity'+str(sample_steps): rdkit_tuple[0][0], 'Uniqueness'+str(sample_steps): rdkit_tuple[0][1], 'Novelty'+str(sample_steps): rdkit_tuple[0][2]})
    model_sample.T = original_sample_steps
    print('model_sample.T set back to ', model_sample.T)
    return validity_dict

def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
