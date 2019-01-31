import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self, base_net, size=512, act_fn=F.relu, norm=2.):
        super(EmbeddingNet, self).__init__()
        self.base_net = base_net
        if size:
            self.fc = nn.Linear(base_net.num_features, size)
        else:
            self.fc = None
        self.act_fn = act_fn
        self.norm = norm

    def forward(self, x):
        output = self.base_net.forward_features(x)
        if self.fc is not None:
            output = self.fc(output)
        if self.act_fn is not None:
            output = self.act_fn(output)
        if self.norm is not None:
            output = output.renorm(p=self.norm, dim=0, maxnorm=1.0)
        return output
