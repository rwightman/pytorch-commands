import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # pairwise distances
        dist = inputs.unsqueeze(1) - inputs.unsqueeze(0)
        dist = torch.sqrt(dist.pow(2).sum(dim=-1) + 1e-12)

        # find the hardest positive and negative
        mask_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_neg = mask_pos ^ 1
        mask_pos = mask_pos ^ Variable(torch.eye(n).byte().cuda())
        dist_p = []
        dist_n = []
        for i in range(n):
            dist_p.append(dist[i][mask_pos[i]].max())
            dist_n.append(dist[i][mask_neg[i]].min())
        dist_p = torch.cat(dist_p)
        dist_n = torch.cat(dist_n)

        # calc loss
        diff = dist_p - dist_n
        if isinstance(self.margin, str) and self.margin == 'soft':
            diff = F.softplus(diff)
        else:
            diff = torch.clamp(diff + self.margin, min=0.)
        loss = diff.mean()

        # precision stats, no impact on loss
        _, top_idx = torch.topk(dist.data, k=2, largest=False)
        top_idx = top_idx[:, 1:]
        flat_idx = top_idx.squeeze() + n * torch.arange(n, out=torch.LongTensor()).cuda()
        top1_is_same = torch.take(mask_pos.data, flat_idx)
        prec = torch.mean(top1_is_same.float())

        print(loss.data[0], prec)

        return loss, prec

