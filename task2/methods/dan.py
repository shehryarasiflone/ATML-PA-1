import torch
import torch.nn as nn

class MultipleKernelMMD(nn.Module):
    """
    Computes Multi-Kernel Maximum Mean Discrepancy (MK-MMD) 
    between source and target feature batches.
    """
    def __init__(self, kernels=None, bandwidths=None):
        super().__init__()
        # Default bandwidth multipliers as specified in the guide: 0.5, 1, 2
        self.bandwidths = bandwidths if bandwidths is not None else [0.5, 1.0, 2.0]

    def compute_kernel(self, x, y, h):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)

        x = x.unsqueeze(1).expand(x_size, y_size, dim)
        y = y.unsqueeze(0).expand(x_size, y_size, dim)
        dist = torch.pow(x - y, 2).sum(dim=2)
        return torch.exp(-dist / h)

    def forward(self, source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        batch_size = source_feat.size(0)
        
        # Combine batches to compute pairwise distances for adaptive bandwidth selection
        combined = torch.cat([source_feat, target_feat], dim=0)
        dot_product = torch.matmul(combined, combined.t())
        square = dot_product.diag().unsqueeze(0)
        dist = square + square.t() - 2 * dot_product
        
        # Exclude diagonal zeros
        mask = torch.eye(dist.size(0), device=dist.device).bool()
        dist_masked = dist.masked_fill(mask, float('inf'))
        median_dist = torch.median(dist_masked[dist_masked != float('inf')])
        if median_dist == 0.0 or torch.isnan(median_dist):
            median_dist = 1.0

        loss = 0.0
        for bw in self.bandwidths:
            h = median_dist * bw
            if h <= 0:
                h = 1.0
            
            k_xx = self.compute_kernel(source_feat, source_feat, h)
            k_yy = self.compute_kernel(target_feat, target_feat, h)
            k_xy = self.compute_kernel(source_feat, target_feat, h)

            # Unbiased MMD formulation
            m = batch_size
            mmd = (k_xx.sum() - m) / (m * (m - 1) + 1e-8) + \
                  (k_yy.sum() - m) / (m * (m - 1) + 1e-8) - \
                  2 * k_xy.mean()
            loss += torch.clamp(mmd, min=0.0)

        return loss