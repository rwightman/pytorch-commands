import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def pdist(v):
    dist = torch.norm(v[:, None] - v, dim=2, p=2)
    return dist


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, sample=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.sample = sample

    def forward(self, inputs, targets):
        n = inputs.size(0)
        #print(n, Counter(targets.cpu().numpy()))

        # pairwise distances
        dist = pdist(inputs)

        # find the hardest positive and negative
        mask_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_neg = ~mask_pos
        mask_pos[torch.eye(n).byte().cuda()] = 0

        if self.sample:
            # sample pos and negative to avoid outliers causing collapse
            posw = (dist + 1e-12) * mask_pos.float()
            posi = torch.multinomial(posw, 1)
            dist_p = dist.gather(0, posi.view(1, -1))
            negw = (1 / (dist + 1e-12)) * mask_neg.float()
            negi = torch.multinomial(negw, 1)
            dist_n = dist.gather(0, negi.view(1, -1))
        else:
            # hard negative
            ninf = torch.ones_like(dist) * float('-inf')
            dist_p = torch.max(dist * mask_pos.float(), dim=1)[0]
            nindex = torch.max(torch.where(mask_neg, -dist, ninf), dim=1)[1]
            dist_n = dist.gather(0, nindex.unsqueeze(0))

        # calc loss
        diff = dist_p - dist_n
        if isinstance(self.margin, str) and self.margin == 'soft':
            diff = F.softplus(diff)
        else:
            diff = torch.clamp(diff + self.margin, min=0.)
        loss = diff.mean()

        # precision stats, no impact on loss
        _, top_idx = torch.topk(dist, k=2, largest=False)
        top_idx = top_idx[:, 1:]
        flat_idx = top_idx.squeeze() + n * torch.arange(n, out=torch.LongTensor()).cuda()
        top1_is_same = torch.take(mask_pos, flat_idx)
        prec = torch.mean(top1_is_same.float())

        #print(loss.item(), prec.item())

        return loss, prec

