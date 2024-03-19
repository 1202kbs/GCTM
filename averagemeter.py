import matplotlib.pyplot as plt
import numpy as np
import time
import math

class AverageMeter:

    def __init__(self, window, loss_names, yscales=None, buf_size=np.inf):
        self.window = window
        self.loss_names = loss_names
        self.buf_size = buf_size
        self.losses = {loss_name : [] for loss_name in loss_names}
        self.yscales = ['linear'] * len(loss_names) if yscales is None else yscales
        self.start = time.time()
        self.end = time.time()
        self.elapsed = 0
        self.idx = 0
    
    def update(self, losses):
        '''
        losses : a dictionary of losses of format {loss_name : loss_value}
        '''
        window_idx = self.idx % self.window
        
        if window_idx == 0:
            for ln in self.loss_names:
                self.losses[ln].append(0)
                self.losses[ln] = self.losses[ln][-min(0,self.buf_size):]
        
        for ln in self.loss_names:
            self.losses[ln][-1] = (window_idx * self.losses[ln][-1] + losses[ln]) / (window_idx + 1)
        
        self.idx += 1
        self.end = time.time()
        self.elapsed += (self.end - self.start)
        self.start = self.end
    
    def plot_losses(self, file_name, nrows=1):
        n_windows = math.ceil(self.idx/self.window)
        n_ticks = min(n_windows, self.buf_size)
        e_idx = self.window * n_windows
        s_idx = self.window * max(1, n_windows-self.buf_size)
        ticks = np.linspace(s_idx, e_idx, n_ticks, dtype=int)

        if len(str(e_idx)) < 4:
            unit = '1'
            ticks = ticks / 1
        else:
            unit = '{}k'.format(10**(len(str(e_idx))-4))
            ticks = ticks / 10**(len(str(e_idx))-1)

        ncols = math.ceil(len(self.loss_names)/nrows)
        plt.figure(figsize=(4*ncols,3*nrows))
        for i, ln in enumerate(self.loss_names):
            plt.subplot(nrows,ncols,i+1)
            plt.plot(ticks, self.losses[ln])
            plt.grid(c='gray', ls='--')
            plt.yscale(self.yscales[i])
            plt.xlabel('Iteration (x{})'.format(unit))
            plt.title(ln)
        plt.tight_layout()        
        plt.savefig(file_name, bbox_inches='tight', dpi=400)
        plt.close()
    
    def state_dict(self):
        return {'losses' : self.losses, 'start' : self.start, 'idx' : self.idx, 'elapsed' : self.elapsed}

    def load_state_dict(self, state_dict):
        self.losses = state_dict['losses']
        self.idx = state_dict['idx']
        if 'elapsed' in state_dict:
            self.elapsed = state_dict['elapsed']
        else:
            self.elapsed = 0
        self.start = time.time()
        self.end = time.time()
    
    def __get_time__(self):
        duration = self.elapsed
        if duration < 60:
            unit = 'sec.'
        elif duration >= 60 and duration < 3600:
            duration = duration / 60
            unit = 'min.'
        else:
            duration = duration / 3600
            unit = 'hours'
        return '{:.2f} {}'.format(duration, unit)
    
    def __str__(self):
        info = 'Iteration {}'.format(self.idx)
        for ln in self.loss_names:
            info += ' | {} : {:.2e}'.format(ln, self.losses[ln][-1])
        info += (' | Time : ' + self.__get_time__())
        return info