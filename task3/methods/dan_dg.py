import torch
import torch.nn as nn
from task2.methods.dan import MultipleKernelMMD

class PairwiseSourceMMD(nn.Module):
    """
    Computes average MMD discrepancy across all unordered pairs of source environments:
    Pairs: (Photo, Art), (Photo, Cartoon), (Art, Cartoon)
    """
    def __init__(self, bandwidths=None):
        super().__init__()
        self.mmd = MultipleKernelMMD(bandwidths=bandwidths)

    def forward(self, feat_dict: dict) -> torch.Tensor:
        """
        feat_dict: dictionary containing 'photo', 'art_painting', 'cartoon' feature tensors
        """
        domains = list(feat_dict.keys())
        total_mmd = 0.0
        num_pairs = 0

        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d1, d2 = domains[i], domains[j]
                mmd_val = self.mmd(feat_dict[d1], feat_dict[d2])
                total_mmd += mmd_val
                num_pairs += 1

        return total_mmd / float(num_pairs)