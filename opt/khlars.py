import torch
from torch import optim
from torch.optim import Optimizer

class KHLARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=5e-4, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False, 
                 update_each = 1, n_samples=1):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)
        
        self.step_cnt = 0
        self.ratio_log = {}
        self.hessian_log = {}
        self.update_each = update_each
        self.n_samples = n_samples
        self.generator = torch.Generator().manual_seed(2147483647)
        
        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0
        
    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad == True)
        
    def zero_hessian(self):
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        grads = [p.grad for p in params]

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        
        self.zero_hessian()
        self.set_hessian()
        
        lst = []
        hess_lst = []
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                hp = p.hess

                if dp is None or hp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    if hp is not None:
                        hessian_norm = torch.norm(hp) if not isinstance(hp, float) else hp 
                        hess_lst.append(hessian_norm.item())                       
                    else:
                        hessian_norm = 1
                    one = torch.ones_like(param_norm)
                    ratio = g['eta'] * param_norm * hessian_norm / update_norm
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (ratio), one), one)
                    dp = dp.mul(q)
                    lst.append(ratio.item())

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
        self.ratio_log[self.step_cnt] = lst
        self.hessian_log[self.step_cnt] = hess_lst
        self.step_cnt += 1
    