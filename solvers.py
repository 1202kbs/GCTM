from abc import *
import torch

def get_solver(solver, discretization):
    if solver == 'euler':
        return EulerSolver(discretization)
    elif solver == 'heun':
        return HeunSolver(discretization)
    else:
        return None

class Solver:
    def __init__(self, discretization):
        self.disc = discretization

    def get_grad(self,xt,t,net,seed=None):
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
                grad = (xt - net(xt,t,t,return_g=True)[0]) / t.reshape(-1,1,1,1)
            else:
                grad = (xt - net(xt,t,t,return_g=True)) / t.reshape(-1,1,1,1)
        return grad
    
    @abstractmethod
    def __step__(self,xt,t,tm1,net,seed=None):
        pass

    def solve(self,xt,t_idx,s_idx,net,sub_steps,ODE_N=1,seed=None):
        ODE_N = min(ODE_N,int(self.disc.disc_steps/sub_steps))
        t_idx, s_idx, sub_steps = ODE_N*t_idx, ODE_N*s_idx, ODE_N*sub_steps
        ts = self.disc.get_ts(sub_steps)
        x_curr = xt.detach().clone()
        i_curr = t_idx
        with torch.no_grad():
            for _ in range(int((t_idx-s_idx).max())):
                update = ~(i_curr == s_idx)
                x_next = self.__step__(x_curr[update],ts[i_curr[update]],ts[i_curr[update]-1],net,seed)
                x_curr.data[update] = x_next
                i_curr.data[update] = (i_curr - 1)[update]
        return x_curr

class EulerSolver(Solver):
    def __init__(self, discretization):
        super().__init__(discretization)
    
    def __step__(self,xt,t,tm1,net,seed=None):
        grad_xt = self.get_grad(xt,t,net,seed)
        xtm1 = xt + grad_xt * (tm1 - t).reshape(-1,1,1,1)
        return xtm1

class HeunSolver(Solver):
    def __init__(self, discretization):
        super().__init__(discretization)
    
    def __step__(self,xt,t,tm1,net,seed=None):
        grad_xt = self.get_grad(xt,t,net,seed)
        xtm1_temp = xt + grad_xt * (tm1 - t).reshape(-1,1,1,1)
        grad_xtm1 = self.get_grad(xtm1_temp,tm1,net,seed)
        xtm1 = xt + 0.5 * (grad_xt + grad_xtm1) * (tm1 - t).reshape(-1,1,1,1)
        xtm1.data[tm1==0] = xtm1_temp.data[tm1==0]
        return xtm1