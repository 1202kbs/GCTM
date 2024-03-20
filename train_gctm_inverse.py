from utils import create_dir, eval_inverse, viz_img, delete_all_but_N_files
from discretizations import get_discretization
from averagemeter import AverageMeter
from distances import get_distance
from solvers import get_solver
from networks import SongUNet
from data import Sampler

import torch.optim as optim
import numpy as np
import pprint

import argparse
import torch
import lpips
import copy
import math
import time
import os

def save_ckpt(X0_eval, X1_eval, net, net_ema, opt_DSM, opt_CTM, avgmeter, best_PSNR, ckpt_dir, idx, best=False):
    
    ckpt = {
        'X0_eval': X0_eval,
        'X1_eval': X1_eval,
        'net': net.state_dict(),
        'net_ema' : net_ema.state_dict(),
        'opt_DSM' : opt_DSM.state_dict(),
        'opt_CTM' : opt_CTM.state_dict(),
        'avgmeter': avgmeter.state_dict(),
        'best_PSNR' : best_PSNR
    }
    
    if best:
        torch.save(ckpt, os.path.join(ckpt_dir, 'idx_0_best.pt'))
    else:
        torch.save(ckpt, os.path.join(ckpt_dir, 'idx_{}_curr.pt'.format(idx)))

def train(datasets, data_roots, X1_eps_std, vars, coupling, lmda_CTM, solver, ctm_distance, compare_zero, size, rho, discretization, smin, smax, edm_rho,
          t_sm_dists, disc_steps, init_steps, ODE_N, bs, coupling_bs, lr, use_pcgrad, ema_decay, n_grad_accum, offline, double_iter, t_ctm_dists,
          nc, model_channels, num_blocks, dropout, param, v_iter, s_iter, b_iter, FID_iter, FID_bs, n_FID, n_viz, n_save, base_dir, ckpt_name):
    
    size = max(size,32)
    sampler = Sampler(datasets, data_roots, nc, size, X1_eps_std, coupling, coupling_bs, bs)
    disc = get_discretization(discretization,disc_steps,smin=smin,smax=smax,rho=edm_rho,t_sm_dists=t_sm_dists,t_ctm_dists=t_ctm_dists)
    ctm_dist, l2_loss = get_distance(ctm_distance), get_distance('l2')
    solver = get_solver(solver,disc)

    vars[1] += X1_eps_std**2
    net = SongUNet(vars=vars, param=param, discretization=disc, img_resolution=size, in_channels=nc, out_channels=nc,
                   num_blocks=num_blocks, dropout=dropout, model_channels=model_channels).cuda()
    opt_DSM = optim.Adam(net.parameters(), lr=lr/(lmda_CTM+1))
    opt_CTM = optim.Adam(net.parameters(), lr=lr)
    net_ema = copy.deepcopy(net)

    avgmeter = AverageMeter(window=125,
                            loss_names=['DSM Loss', 'CTM Loss', 'PSNR G', 'PSNR g', 'SSIM G', 'SSIM g', 'LPIPS G', 'LPIPS g'],
                            yscales=['log','log','linear', 'linear','linear', 'linear','linear', 'linear'])

    loss_dir = os.path.join(base_dir, 'losses')
    sample_B_dir = os.path.join(base_dir, 'samples_B')
    sample_F_dir = os.path.join(base_dir, 'samples_F')
    ckpt_dir = os.path.join(base_dir, 'ckpts')

    if ckpt_name:
        print('\nLoading state from [{}]\n'.format(ckpt_name))
        ckpt = torch.load(os.path.join(ckpt_dir, ckpt_name))
        X0_eval = ckpt['X0_eval'].cuda()
        X1_eval = ckpt['X1_eval'].cuda()
        net.load_state_dict(ckpt['net'])
        opt_DSM.load_state_dict(ckpt['opt_DSM'])
        opt_CTM.load_state_dict(ckpt['opt_CTM'])
        net_ema.load_state_dict(ckpt['net_ema'])
        avgmeter.load_state_dict(ckpt['avgmeter'])
        loss_DSM = avgmeter.losses['DSM Loss'][-1]
        loss_CTM = avgmeter.losses['CTM Loss'][-1]
        best_PSNR = ckpt['best_PSNR']
    else:
        X0_eval = torch.cat([sampler.sample_X0() for _ in range(math.ceil(n_viz/bs))], dim=0)[:n_viz]
        X1_eval = torch.cat([sampler.sample_X1() for _ in range(math.ceil(n_viz/bs))], dim=0)[:n_viz]
        best_PSNR = 0
        create_dir(base_dir,prompt=True)
        create_dir(loss_dir)
        create_dir(sample_B_dir)
        create_dir(sample_F_dir)
        create_dir(ckpt_dir)
    
    # Evaluate initial FID and visualize X1 samples
    t0 = disc.get_ts(disc_steps)[0]
    t1 = disc.get_ts(disc_steps)[-1]
    curr_PSNR_G, curr_SSIM_G, curr_LPIPS_G = eval_inverse(sampler, t1, t0, net_ema, n_FID, tweedie=False, verbose=True)
    curr_PSNR_g, curr_SSIM_g, curr_LPIPS_g = eval_inverse(sampler, t1, t1, net_ema, n_FID, tweedie=True, verbose=True)
    viz_img(X1_eval, t1, t1, net_ema, sample_B_dir, 0)
    
    # Training loop
    seed = int(time.time())
    while True:
        if double_iter is not None:
            sub_steps = min(disc_steps,init_steps*2**(avgmeter.idx//double_iter))
        else:
            sub_steps = init_steps
        
        opt_CTM.zero_grad()
        for accum_idx in range(n_grad_accum):
            # Sample data
            X0, X1 = sampler.sample_joint()
            t_sm_idx, t_sm = disc.sample_sm_times(bs, disc_steps)
            Xt_sm = (1 - t_sm).reshape(-1,1,1,1) * X0 + t_sm.reshape(-1,1,1,1) * X1

            t_idx, s_idx, u_idx, v_idx, t, s, u, v = disc.sample_ctm_times(bs, sub_steps)
            Xt = (1 - t).reshape(-1,1,1,1) * X0 + t.reshape(-1,1,1,1) * X1

            # Calculate CTM Loss
            with torch.no_grad():
                if offline:
                    net_ema.eval()
                    Xu_real = solver.solve(Xt,t_idx,u_idx,net_ema,sub_steps,ODE_N)
                else:
                    net.train()
                    Xu_real = solver.solve(Xt,t_idx,u_idx,net,sub_steps,ODE_N,seed)
                
                net.train()
                torch.manual_seed(seed)
                Xs_real = net(Xu_real,u,s)[0]
                
                net_ema.eval()
                X_real = net_ema(Xs_real,s,t0*torch.zeros_like(s)) if compare_zero else Xs_real

            net.train()
            net_ema.eval()
            torch.manual_seed(seed)
            Xs_fake, cout = net(Xt,t,s)
            X_fake = net_ema(Xs_fake,s,t0*torch.zeros_like(s)) if compare_zero else Xs_fake
            loss_CTM = ctm_dist(X_fake,X_real,cout*(1-s/t))/(n_grad_accum*bs)
            (lmda_CTM*loss_CTM).backward()
            seed += 1

            # Calculate DSM Loss
            net.train()
            X0_fake, cout = net(Xt_sm,t_sm,t_sm,return_g=True)
            loss_DSM = l2_loss(X0_fake,X0,cout)/(n_grad_accum*bs)
            loss_DSM.backward()

            if accum_idx == n_grad_accum-1:
                opt_CTM.step()

        # EMA update
        if double_iter is not None:
            ema_list = [0.999, 0.9999, 0.99995]
            ema_decay_curr = ema_list[min(int(avgmeter.idx//double_iter),2)]
        else:
            ema_decay_curr = ema_decay
        with torch.no_grad():
            for p, p_ema in zip(net.parameters(),net_ema.parameters()):
                p_ema.data = ema_decay_curr * p_ema + (1 - ema_decay_curr) * p

        # Loss tracker update
        avgmeter.update({'DSM Loss' : loss_DSM.item()*n_grad_accum,
                         'CTM Loss' : loss_CTM.item()*n_grad_accum,
                         'PSNR G'   : curr_PSNR_G,
                         'PSNR g'   : curr_PSNR_g,
                         'SSIM G'   : curr_SSIM_G,
                         'SSIM g'   : curr_SSIM_g,
                         'LPIPS G'   : curr_LPIPS_G,
                         'LPIPS g'   : curr_LPIPS_g})
        
        # Loss and sample visualization
        if avgmeter.idx % v_iter == 0:
            print(avgmeter)
            avgmeter.plot_losses(os.path.join(loss_dir, 'losses.jpg'), nrows=2)
            viz_img(X1_eval, t1, t0, net_ema, sample_B_dir, None)

        # Saving checkpoint
        if avgmeter.idx % s_iter == 0:
            print('\nSaving checkpoint at [{}], Best PSNR : {:.2f}\n'.format(ckpt_dir,best_PSNR))
            save_ckpt(X0_eval, X1_eval, net, net_ema, opt_DSM, opt_CTM, avgmeter, best_PSNR, ckpt_dir, avgmeter.idx)
            delete_all_but_N_files(ckpt_dir, lambda x : int(x.split('_')[1]), n_save, 'best')
            viz_img(X1_eval, t1, t0, net_ema, sample_B_dir, avgmeter.idx)
        
        # Saving backup checkpoint
        if avgmeter.idx % b_iter == 0:
            print('\nSaving backup checkpoint at [{}]\n'.format(base_dir))
            save_ckpt(X0_eval, X1_eval, net, net_ema, opt_DSM, opt_CTM, avgmeter, best_PSNR, base_dir, avgmeter.idx)
        
        # Evaluating Quick FID
        if avgmeter.idx % FID_iter == 0:
            curr_PSNR_G, curr_SSIM_G, curr_LPIPS_G = eval_inverse(sampler, t1, t0, net_ema, n_FID, tweedie=False, verbose=True)
            curr_PSNR_g, curr_SSIM_g, curr_LPIPS_g = eval_inverse(sampler, t1, t1, net_ema, n_FID, tweedie=True, verbose=True)
            if curr_PSNR_G > best_PSNR:
                best_PSNR = curr_PSNR_G
                save_ckpt(X0_eval, X1_eval, net, net_ema, opt_DSM, opt_CTM, avgmeter, best_PSNR, ckpt_dir, avgmeter.idx, best=True)

def main():
    parser = argparse.ArgumentParser()

    # Basic experiment settings
    parser.add_argument('--datasets', type=str, nargs='+', default=['cifar10','gaussian'])
    parser.add_argument('--data_roots', type=str, nargs='+', default=['../data','../data'])
    parser.add_argument('--base_dir', type=str, default='results/cifar10')
    parser.add_argument('--ckpt_name', type=str, default=None)

    # p(X0,X1) settings
    # inverse tasks = {'sr4x-pool', 'sr4x-bicubic', 'inpaint-center', 'inpaint-random', 'blur-uni', 'blur-gauss'}
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--X1_eps_std', type=float, default=0.0)
    parser.add_argument('--vars', type=float, nargs='+', default=[0.25,1.0,0.0])
    parser.add_argument('--coupling', type=str, default='independent')
    parser.add_argument('--coupling_bs', type=int, default=64)

    # ODE settings
    parser.add_argument('--disc_steps', type=int, default=1024)
    parser.add_argument('--init_steps', type=int, default=8)
    parser.add_argument('--double_iter', type=int, default=None)
    parser.add_argument('--solver', type=str, default='heun')
    parser.add_argument('--discretization', type=str, default='edm_n2i')
    parser.add_argument('--smin', type=float, default=0.002)
    parser.add_argument('--smax', type=float, default=80.0)
    parser.add_argument('--edm_rho', type=int, default=7)
    parser.add_argument('--t_sm_dists', type=str, nargs='+', default=[])
    parser.add_argument('--t_ctm_dists', type=float, nargs='+', default=[1.2,2])
    parser.add_argument('--param', type=str, default='LIN')
    parser.add_argument('--ODE_N', type=int, default=1)

    # Optimization settings
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--lmda_CTM', type=float, default=0.1)
    parser.add_argument('--ctm_distance', type=str, default='l1')
    parser.add_argument('--ema_decay', type=float, default=0.9)
    parser.add_argument('--n_grad_accum', type=int, default=1)
    parser.add_argument('--compare_zero', action='store_true')
    parser.add_argument('--use_pcgrad', action='store_true')
    parser.add_argument('--offline', action='store_true')

    # Model settings
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Evaluation settings
    parser.add_argument('--v_iter', type=int, default=25)
    parser.add_argument('--s_iter', type=int, default=250)
    parser.add_argument('--b_iter', type=int, default=25000)
    parser.add_argument('--FID_iter', type=int, default=250)
    parser.add_argument('--n_FID', type=int, default=5000)
    parser.add_argument('--FID_bs', type=int, default=500)
    parser.add_argument('--n_viz', type=int, default=100)
    parser.add_argument('--n_save', type=int, default=2)
    
    args = parser.parse_args()
    def print_args(**kwargs):
        print('\nTraining with settings :\n')
        pprint.pprint(kwargs)
    print_args(**vars(args))
    train(**vars(args))

if __name__ == '__main__':
    main()