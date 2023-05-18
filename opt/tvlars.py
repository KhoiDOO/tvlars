import torch
from torch import optim

class TVLARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=5e-4, momentum=0.9, eta=0.001, lmbda=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter, lmbda=lmbda)
        super().__init__(params, defaults)
        
        self.step = 0
        self.ratio_log = {}

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        lst = []
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    ratio = g['eta'] * torch.pow(
                                                torch.exp(
                                                    g['lmbda'] * torch.FloatTensor([self.epoch_cnt + 1])
                                                    )[0], -1
                                                ) * param_norm / update_norm
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (ratio), one), one)
                    dp = dp.mul(q)
                    lst.append(ratio)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])

        self.ratio_log[self.step] = lst
        self.step += 1