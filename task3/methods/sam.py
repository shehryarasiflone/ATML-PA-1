import torch

class SAM(torch.optim.Optimizer):
    """
    Standard, non-adaptive Sharpness-Aware Minimization (Foret et al., 2021).
    Wraps an underlying base optimizer (AdamW).
    """
    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation radius rho: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Calculates normalized ascent perturbation epsilon and perturbs weights."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Cache perturbation vector e_w
                self.state[p]["e_w"] = p.grad * scale.to(p)
                # Perturb weights to local maximum: theta + epsilon
                p.add_(self.state[p]["e_w"])
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restores original weights and performs the descent step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore original weights: theta
                p.sub_(self.state[p]["e_w"])
        # Update original weights using gradient evaluated at perturbed point
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)