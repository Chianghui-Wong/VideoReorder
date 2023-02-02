import torch
import torch.nn as nn
import torch.nn.functional as F

class PairWiseLoss(nn.Module):
    '''
    For the right order i < j, the Loss defined as  
    L(V_i, V_j) = ||max( 0,  \phi_t(V_i)-\phi_t(V_j) ) ||^2

    Thus the total loss is
    L_total = \sum_{i > j} L(V_i, V_j)
    for a set of M clips, there is M(M-1)/2 pair need to calculate
    ''' 

    def __init__(self, **kwargs):
        super(PairWiseLoss, self).__init__()
        return
    
    def forward(self, repr):
        '''
        repr is a M * (N ?) torch
        M is the number of clips
        the other demension is the feature vector of the clip
        '''
        loss_total = 0.0
        len = repr.shape[0]
        for i in range(len):
            for j in range(i+1, len):
                phi_V_diff = torch.sub(repr[i], repr[j])
                zero = torch.zeros_like(phi_V_diff)
                phi_V_max_with_0 = torch.where(phi_V_diff < 0, zero, phi_V_diff)
                loss_total += torch.norm(phi_V_max_with_0)
        
        return loss_total


if __name__ == '__main__':
    LossFunc = PairWiseLoss()
    #input = torch.tensor([[1], [2], [3], [4]]).float()
    input = torch.tensor([[3], [4], [2], [1]]).float()
    loss = LossFunc(input)
    print(loss)