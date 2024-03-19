from scipy.stats import beta, norm, uniform
from abc import *

import numpy as np
import torch

def get_discretization(discretization, disc_steps, **kwargs):
    if discretization == 'edm_n2i':
        return EDM_N2I(disc_steps, **kwargs)
    elif discretization == 'edm_i2i':
        return EDM_I2I(disc_steps, **kwargs)
    else:
        return None
    
class EDM:
    def __init__(self, disc_steps, smin=0.002, smax=80, rho=7, t_sm_dists=[], t_ctm_dists=[1.2,2]):
        self.disc_steps = disc_steps
        self.smin = smin
        self.smax = smax
        self.rho = rho
        self.t_sm_dists = t_sm_dists
        self.t_ctm_dists = t_ctm_dists
    
    @abstractmethod
    def get_ts(self, sub_steps):
        pass

    def discretize(self, t, sub_steps):
        ts = self.get_ts(sub_steps)
        t_disc, t_idx = (t[:,None] - ts[None]).abs().min(dim=1)
        t_idx.data[t_disc<t] = (t_idx+1)[t_disc<t].clamp(max=sub_steps)
        t_disc = ts[t_idx]
        return t_disc, t_idx

    def sample_ctm_times(self, bs, sub_steps, max_skip_N=1):
        ts = self.get_ts(sub_steps)
        
        s_pdf = 1.0 - ts[:-1]
        s_pdf = s_pdf / s_pdf.sum()
        s_idx = torch.multinomial(s_pdf, num_samples=bs, replacement=True)

        # lmda = torch.tensor(beta.rvs(a=1.2,b=self.ctm_beta,size=bs)).cuda()
        lmda = torch.tensor(beta.rvs(a=self.t_ctm_dists[0],b=self.t_ctm_dists[1],size=bs)).cuda()
        t_temp = lmda + (1 - lmda) * ts[s_idx]
        _, t_idx = self.discretize(t_temp, sub_steps)
        t_idx = t_idx.clamp(min=s_idx+1)

        u_rng = (t_idx - s_idx)
        u_dlt = (torch.rand(size=[bs]).cuda() * u_rng).clamp(max=u_rng-1).long() + 1
        u_idx = t_idx - u_dlt

        v_idx = (t_idx - max_skip_N).clamp(min=u_idx)

        ts = self.get_ts(sub_steps)
        t, s, u, v = ts[t_idx], ts[s_idx], ts[u_idx], ts[v_idx]
        return t_idx, s_idx, u_idx, v_idx, t, s, u, v
    
    def sample_sm_times(self, bs, sub_steps):
        ts = self.get_ts(sub_steps)
        if len(self.t_sm_dists) == 0:
            t_idx = torch.randint(low=1, high=sub_steps+1, size=[bs]).cuda()
        else:
            t_sm_dist = np.random.choice(self.t_sm_dists)
            if 'lognormal' in t_sm_dist:
                ss = 1/(1/ts-1)
                cdf = torch.tensor(norm.cdf(ss.log().cpu(),loc=-1.2,scale=1.2))
            elif 'beta' in t_sm_dist:
                _, a, b = t_sm_dist.split('_')
                cdf = torch.tensor(beta.cdf(ts.cpu(),float(a),float(b)))
            elif 't-uniform' in t_sm_dist:
                _, a, b = t_sm_dist.split('_')
                cdf = torch.tensor(uniform.cdf(ts.cpu(),loc=a,scale=b))
            elif 'idx-uniform' in t_sm_dist:
                _, a, b = t_sm_dist.split('_')
                a, b = float(a), float(b)
                ss = 1/(1/ts-1)
                ss[0], ss[-1] = self.smin, self.smax
                smin_sqrt = self.smin**(1/self.rho)
                smax_sqrt = self.smax**(1/self.rho)
                s_lb = (smin_sqrt + a * (smax_sqrt - smin_sqrt))**self.rho
                s_ub = (smin_sqrt + b * (smax_sqrt - smin_sqrt))**self.rho
                s_lb_cdf = (s_lb**(1/self.rho) - smin_sqrt) / (smax_sqrt - smin_sqrt)
                s_ub_cdf = (s_ub**(1/self.rho) - smin_sqrt) / (smax_sqrt - smin_sqrt)
                cdf = (ss**(1/self.rho) - smin_sqrt) / (smax_sqrt - smin_sqrt)
                cdf = cdf.clamp(min=s_lb_cdf, max=s_ub_cdf)

            pdf = cdf[1:] - cdf[:-1]
            pdf = pdf / pdf.sum()
            t_idx = torch.multinomial(pdf, num_samples=bs, replacement=True) + 1
        return t_idx, ts[t_idx]
    
class EDM_N2I(EDM):

    def __init__(self, disc_steps, smin=0.002, smax=80, rho=7, t_sm_dists=[], t_ctm_dists=[1.2,2]):
        super().__init__(disc_steps, smin, smax, rho, t_sm_dists, t_ctm_dists)
    
    def get_ts(self, sub_steps):
        ts = torch.linspace(np.power(self.smin,1/self.rho),np.power(self.smax,1/self.rho),self.disc_steps+1).pow(self.rho)
        ts = ts / (ts + 1)
        ts[0] = 0.0
        ts[-1] = 1.0
        ts = ts[torch.linspace(0,self.disc_steps,sub_steps+1,dtype=int)]
        return ts.cuda()

class EDM_I2I(EDM):

    def __init__(self, disc_steps, smin=0.002, smax=80, rho=7, t_sm_dists=[]):
        super().__init__(disc_steps, smin, smax, rho, t_sm_dists)
    
    def get_ts(self, sub_steps):
        ts = torch.linspace(np.power(self.smin,1/self.rho),np.power(self.smax,1/self.rho),int(0.5*self.disc_steps)+1).pow(self.rho)
        ts = ts / (ts + 1)
        ts[0] = 0.0
        ts[-1] = 1.0
        ts = torch.cat([0.5*ts[:-1],1-0.5*ts.flip(dims=[0])])
        ts = ts[torch.linspace(0,self.disc_steps,sub_steps+1,dtype=int)]
        return ts.cuda()