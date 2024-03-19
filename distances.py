import numpy as np
import torch
import lpips
import math

def get_distance(distance):
    if distance == 'l1':
        return l1_loss
    elif distance == 'l2':
        return l2_loss
    elif distance == 'ph':
        return ph_loss
    elif distance == 'eph':
        return elt_ph_loss
    elif 'lpips' in distance:
        lpips_dist = lpips.LPIPS(distance.split('_')[1]).cuda()
        def lpips_loss(x,y,weight=None,reduce=True):
            weight = weight if (weight is not None) else torch.ones(size=[x.shape[0]]).to(x.device)
            if reduce:
                return (weight * lpips_dist(x,y).reshape(-1)).sum()
            else:
                return weight * lpips_dist(x,y).reshape(-1)
        return lpips_loss
    else:
        return None

def l1_loss(x,y,weight=None,reduce=True):
    weight = weight if (weight is not None) else torch.ones(size=[x.shape[0]]).to(x.device)
    if reduce:
        return (1/weight * (x-y).flatten(start_dim=1).abs().sum(dim=1)).sum()
    else:
        return 1/weight * (x-y).flatten(start_dim=1).abs().sum(dim=1)

def l2_loss(x,y,weight=None):
    weight = weight if (weight is not None) else torch.ones(size=[x.shape[0]]).to(x.device)
    return (1/weight.square() * (x-y).flatten(start_dim=1).square().sum(dim=1)).sum()

def ph_loss(x,y,weight=None,reduce=True):
    dim = x[0].flatten().shape[0]
    c = 0.00054 * np.sqrt(dim)
    weight = weight if (weight is not None) else torch.ones(size=[x.shape[0]]).to(x.device)
    c = c * weight
    if reduce:
        return (1/weight * (((x-y).flatten(start_dim=1).square().sum(dim=1) + c**2).sqrt() - c)).sum()
    else:
        return 1/weight * (((x-y).flatten(start_dim=1).square().sum(dim=1) + c**2).sqrt() - c)

def elt_ph_loss(x,y,weight=None,reduce=True,tau=2/256,eps=0.05):
    c = tau**2 * (1/(1-eps)**2 - 1)
    weight = weight if (weight is not None) else torch.ones(size=[x.shape[0]]).to(x.device)
    c = c * weight
    if reduce:
        return (1/weight * (((x-y).flatten(start_dim=1).square() + c**2).sqrt() - c).sum(dim=1)).sum()
    else:
        return 1/weight * (((x-y).flatten(start_dim=1).square() + c**2).sqrt() - c).sum(dim=1)