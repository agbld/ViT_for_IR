import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

class KoLeoLossTriplet(nn.Module):
    def __init__(self):
        super(KoLeoLossTriplet, self).__init__()

    def forward(self, anchor, positive, negative):
        d_anchor_anchor = torch.cdist(anchor, anchor)
        d_anchor_positive = torch.cdist(anchor, positive)
        d_anchor_negative = torch.cdist(anchor, negative)
        d = torch.cat((d_anchor_anchor, d_anchor_positive, d_anchor_negative), dim=1)
        d_remove_self = torch.where(d == 0, d.max(), d)
        min_d = torch.min(d_remove_self, dim=1)[0]
        
        n = anchor.shape[0]
        
        loss_koleo = -(1/n) * torch.sum(torch.log(min_d))
        
        return loss_koleo
    
class KoLeoLoss(nn.Module):
    def __init__(self):
        super(KoLeoLoss, self).__init__()

    def forward(self, embeddings):
        d = torch.cdist(embeddings, embeddings)
        d_remove_self = torch.where(d == 0, d.max(), d)
        min_d = torch.min(d_remove_self, dim=1)[0]
        n = embeddings.shape[0]
        
        return -(1/n) * torch.sum(torch.log(min_d))

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = torch.cdist(
            anchor.unsqueeze(1), positive.unsqueeze(1), p=2)
        distance_negative = torch.cdist(
            anchor.unsqueeze(1), negative.unsqueeze(1), p=2)

        losses = torch.log(
            1 + torch.exp(distance_positive - distance_negative))

        return losses.mean() if size_average else losses.sum()