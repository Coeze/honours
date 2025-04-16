import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(y, n_dims):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).cuda()
    return zeros.scatter(scatter_dim, y_tensor, 1)

class DynamicRoutingLoss(nn.Module):
    def __init(self):
        super(DynamicRoutingLoss, self).__init()

    def forward(self, x, target):
        target = one_hot(target, x.shape[1])

        left = F.relu(0.9 - x) ** 2
        right = F.relu(x - 0.1) ** 2

        margin_loss = target * left + 0.5 * (1. - target) * right
        margin_loss = margin_loss.sum(dim=1).mean()
        return margin_loss