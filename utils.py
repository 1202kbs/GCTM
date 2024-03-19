from pytorch_fid.fid_score import calculate_fid_given_paths, save_fid_stats
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_msssim import ssim
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import shutil
import torch
import lpips
import math
import sys
import os

LPIPS_ALEX = lpips.LPIPS(net='alex').cuda()

def create_FID_statistics(sample_save_dir, stat_save_dir, data_loader, device='cuda:0'):
    create_dir(sample_save_dir)
    num_created = 0
    for img in data_loader:
        img = tensor2img(img[0])
        for i in range(img.shape[0]):
            img_i = Image.fromarray(img[i])
            img_i.save(os.path.join(sample_save_dir, '{}.png'.format(num_created)))
            num_created += 1
    save_fid_stats([sample_save_dir, stat_save_dir], batch_size=50, device=device, dims=2048)
    shutil.rmtree(sample_save_dir)

def eval_FID_ODE(sample_Xt, Xs_path, t_idx, s_idx, net, solver, sub_steps, n_samples, sample_dir, fid_bs=500, invert=lambda x : x, verbose=True):
    FID_dir = os.path.join(sample_dir, 'FID_samples')
    create_dir(FID_dir)
    num_created = 0
    net.eval()
    bs = sample_Xt().shape[0]
    n_iter = math.ceil(n_samples/bs)
    if verbose:
        print('\n--------Calculating FID--------')
    for _ in tqdm(range(n_iter), desc='Creating FID Samples'.ljust(34), disable=(not verbose)):
        with torch.no_grad():
            Xt = sample_Xt()
            t_idx_vec = t_idx * torch.ones(size=[Xt.shape[0]]).cuda().long()
            s_idx_vec = s_idx * torch.ones(size=[Xt.shape[0]]).cuda().long()
            samples = solver.solve(Xt,t_idx_vec,s_idx_vec,net,sub_steps)
            Xs = tensor2img(invert(samples))
        for i in range(Xs.shape[0]):
            img = Image.fromarray(Xs[i])
            img.save(os.path.join(FID_dir, '{}.png'.format(num_created)))
            num_created += 1
            if num_created == n_samples:
                break
    with torch.no_grad():
        fid = calculate_fid_given_paths([Xs_path, FID_dir], batch_size=fid_bs, device='cuda:0', dims=2048, num_workers=os.cpu_count(), verbose=verbose)
    if verbose:
        print('FID from Idx {} to Idx {} : {:.3f}'.format(t_idx,s_idx,fid))
        print('-------------------------------\n')
    shutil.rmtree(FID_dir)
    return fid

def eval_FID(sample_Xt, Xs_path, t, s, net, n_samples, sample_dir, fid_bs=500, invert=lambda x : x, verbose=True):
    FID_dir = os.path.join(sample_dir, 'FID_samples')
    create_dir(FID_dir)
    num_created = 0
    net.eval()
    bs = sample_Xt().shape[0]
    n_iter = math.ceil(n_samples/bs)
    if verbose:
        print('\n--------Calculating FID--------')
    for _ in tqdm(range(n_iter), desc='Creating FID Samples'.ljust(34), disable=(not verbose)):
        with torch.no_grad():
            Xt = sample_Xt()
            t_vec = t * torch.ones(size=[Xt.shape[0]]).cuda()
            s_vec = s * torch.ones(size=[Xt.shape[0]]).cuda()
            Xs = tensor2img(invert(net(Xt,t_vec,s_vec)))
        for i in range(Xs.shape[0]):
            img = Image.fromarray(Xs[i])
            img.save(os.path.join(FID_dir, '{}.png'.format(num_created)))
            num_created += 1
            if num_created == n_samples:
                break
    with torch.no_grad():
        fid = calculate_fid_given_paths([Xs_path, FID_dir], batch_size=fid_bs, device='cuda:0', dims=2048, num_workers=os.cpu_count(), verbose=verbose)
    if verbose:
        print('FID from {} to {} : {:.3f}'.format(t,s,fid))
        print('-------------------------------\n')
    shutil.rmtree(FID_dir)
    return fid

@torch.no_grad()
def eval_inverse(sampler, t, s, net, n_samples, tweedie=False, verbose=True):
    net.eval()
    bs = sampler.bs
    n_iter = math.ceil(n_samples/bs)

    t_vec = t * torch.ones(size=[bs]).cuda()
    s_vec = s * torch.ones(size=[bs]).cuda()

    if verbose:
        print('\n--------Evaluating Model--------')
    
    Xs_fake = []
    Xs_real = []
    for _ in tqdm(range(n_iter), desc='Creating Samples'.ljust(34), disable=(not verbose)):
        Xs, Xt = sampler.sample_joint()
        Xs_real.append(Xs)
        if net == None:
            Xs_fake.append(Xt)
        else:
            Xs_fake.append(net(Xt, t_vec, s_vec, return_g=tweedie))
    Xs_fake = torch.cat(Xs_fake, dim=0)
    Xs_real = torch.cat(Xs_real, dim=0)

    # Measure PSNR
    psnrs = []
    for out_, label_ in zip(Xs_fake, Xs_real):
        out_ = tensor2img(out_[None, ...])
        label_ = tensor2img(label_[None, ...])
        psnrs.append(psnr(label_, out_, data_range=255.0))
    psnr_res = np.array(psnrs).mean()

    # Measure SSIM
    ssims = []
    for out_, label_ in zip(Xs_fake, Xs_real):
        out_ = out_[None].clamp(-1,1) * 0.5 + 0.5
        label_ = label_[None].clamp(-1,1) * 0.5 + 0.5
        ssims.append(ssim(label_, out_, data_range=1, size_average=True).item())
    ssim_res = np.array(ssims).mean()

    # Measure LPIPS
    lpips_res = LPIPS_ALEX(Xs_fake,Xs_real).mean().item()

    if verbose:
        print('PSNR  from {} to {} : {:.3f}'.format(t,s,psnr_res))
        print('SSIM  from {} to {} : {:.3f}'.format(t,s,ssim_res))
        print('LPIPS from {} to {} : {:.3f}'.format(t,s,lpips_res))
        print('--------------------------------\n')
    return psnr_res, ssim_res, lpips_res

@torch.no_grad()
def eval_PSNR(sampler, t, s, net, n_samples, tweedie=False, verbose=True):
    if net is not None:
        net.eval()

    bs = sampler.bs
    n_iter = math.ceil(n_samples/bs)

    t_vec = t * torch.ones(size=[bs]).cuda()
    s_vec = s * torch.ones(size=[bs]).cuda()

    Xs_fake = [] 
    Xs_real = []

    if verbose:
        print('\n--------Calculating PSNR--------')
    for _ in tqdm(range(n_iter), desc='Creating PSNR Samples'.ljust(34), disable=(not verbose)):
        Xs, Xt = sampler.sample_joint()
        if net == None:
            Xs_fake.append(Xt)
        else:
            Xs_fake.append(net(Xt, t_vec, s_vec, return_g=tweedie))
        Xs_real.append(Xs)
    Xs_fake = torch.cat(Xs_fake, dim=0)
    Xs_real = torch.cat(Xs_real, dim=0)

    psnrs = []
    for out_, label_ in zip(Xs_fake, Xs_real):
        out_ = tensor2img(out_[None, ...])
        label_ = tensor2img(label_[None, ...])
        psnrs.append(psnr(label_, out_, data_range=255.0))
    
    psnr_res = np.array(psnrs).mean()
    if verbose:
        print('PSNR from {} to {} : {:.3f}'.format(t,s,psnr_res))
        print('--------------------------------\n')
    return psnr_res

@torch.no_grad()
def eval_LPIPS(sampler, t, s, net, n_samples, tweedie=False, verbose=True):
    lpips_dist = lpips.LPIPS(net='alex').cuda()
    if net is not None:
        net.eval()

    bs = sampler.bs
    n_iter = math.ceil(n_samples/bs)

    t_vec = t * torch.ones(size=[bs]).cuda()
    s_vec = s * torch.ones(size=[bs]).cuda()

    Xs_fake = [] 
    Xs_real = []

    if verbose:
        print('\n--------Calculating LPIPS--------')
    for _ in tqdm(range(n_iter), desc='Creating PSNR Samples'.ljust(34), disable=(not verbose)):
        Xs, Xt = sampler.sample_joint()
        if net == None:
            Xs_fake.append(Xt)
        else:
            Xs_fake.append(net(Xt, t_vec, s_vec, return_g=tweedie))
        Xs_real.append(Xs)
    Xs_fake = torch.cat(Xs_fake, dim=0)
    Xs_real = torch.cat(Xs_real, dim=0)

    lpips_res = lpips_dist(Xs_fake,Xs_real).mean().item()
    if verbose:
        print('LPIPS from {} to {} : {:.3f}'.format(t,s,lpips_res))
        print('---------------------------------\n')
    return lpips_res

def eval_MMD(sample_Xt, sample_Xs, t, s, net, n_samples, verbose=True):
    net.eval()
    bs = sample_Xt().shape[0]
    n_iter = math.ceil(n_samples/bs)
    Xs_real = torch.cat([sample_Xs() for _ in range(n_iter)], dim=0)
    Xs_fake = []
    for _ in range(n_iter):
        Xt = sample_Xt()
        t_vec = t * torch.ones(size=[Xt.shape[0]]).cuda()
        s_vec = s * torch.ones(size=[Xt.shape[0]]).cuda()
        with torch.no_grad():
            Xs_fake.append(net(Xt,t_vec,s_vec))
    Xs_fake = torch.cat(Xs_fake, dim=0)
    mmd = MMD(Xs_real,Xs_fake,'multiscale').item()
    if verbose:
        print('\n-------------------------------')
        print('MMD from {} to {} : {:.3f}'.format(t,s,mmd))
        print('-------------------------------\n')
    return mmd

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)

def viz_img(Xt_eval, t, s, net, sample_dir, idx, invert=lambda x : x):
    net.eval()
    nc = Xt_eval.shape[1]
    with torch.no_grad():
        t_vec = t * torch.ones(size=[Xt_eval.shape[0]]).cuda()
        s_vec = s * torch.ones(size=[Xt_eval.shape[0]]).cuda()
        Xs = tensor2img(invert(net(Xt_eval,t_vec,s_vec)))

    N = int(np.sqrt(Xt_eval.shape[0]))
    plt.figure(figsize=(1.5*N,1.5*N))
    for i in range(N**2):
        plt.subplot(N,N,i+1)
        plt.imshow(Xs[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    if idx is not None:
        plt.savefig(os.path.join(sample_dir, 'sample_{}.jpg'.format(idx)), bbox_inches='tight', dpi=200)
    else:
        plt.savefig(os.path.join(sample_dir, 'sample_curr.jpg'), bbox_inches='tight', dpi=200)
    plt.close()

def viz_2d(Xt_eval, t, s, net, sample_dir, idx):
    net.eval()
    with torch.no_grad():
        t_vec = t * torch.ones(size=[Xt_eval.shape[0]]).cuda()
        s_vec = s * torch.ones(size=[Xt_eval.shape[0]]).cuda()
        Xs = net(Xt_eval,t_vec,s_vec).detach().cpu().numpy()

    plt.figure(figsize=(3,3))
    plt.hist2d(Xs[:,0], Xs[:,1], bins=50, range=[[-3.0,3.0],[-3.0,3.0]])
    plt.xlim([-3.0,3.0])
    plt.ylim([-3.0,3.0])
    plt.xticks([])
    plt.yticks([])
    if idx : plt.savefig(os.path.join(sample_dir, 'sample_{}.jpg'.format(idx)), bbox_inches='tight', dpi=200)
    plt.savefig(os.path.join(sample_dir, 'sample_curr.jpg'), bbox_inches='tight', dpi=200)
    plt.close()

def create_dir(path,prompt=False):
    if os.path.isdir(path):
        if prompt:
            reply = input("\nOverwrite directory [{}]? [y/[n]] ".format(path))
            if reply=='y':
                shutil.rmtree(path)
            else:
                print("Aborting...")
                sys.exit()
        else:
            shutil.rmtree(path)
    os.makedirs(path)

def tensor2img(X):
    X = (X * 127.5 + 128).clip(0,255).to(torch.uint8)
    if X.shape[1]==3:
        return X.permute(0,2,3,1).detach().cpu().numpy()
    else:
        return X[:,0][...,None].repeat([1,1,1,3]).detach().cpu().numpy()

def delete_all_but_N_files(path, sort_key, N, skip_str):
    files = os.listdir(path)
    files.sort(key=sort_key)
    rm_files = files[:-N]
    if len(rm_files) > 0:
        for f in rm_files:
            file_path = os.path.join(path,f)
            if skip_str in file_path:
                continue
            os.remove(file_path)

def get_grad_list(net):
    grad_list = []
    for p in net.parameters():
        if p.requires_grad:
            grad_list.append(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
    return grad_list

def set_grad_list(net, grad_list):
    grad_idx = 0
    for p in net.parameters():
        if p.requires_grad:
            p.grad.data = grad_list[grad_idx]
            grad_idx += 1

def update_grad_list(net, grad_list):
    grad_idx = 0
    for p in net.parameters():
        if p.requires_grad:
            grad_list[grad_idx] += p.grad.detach().clone()
            grad_idx += 1

def project(x, y):
    return x - (x * y).sum() / (y.norm().square() + 1e-8) * y