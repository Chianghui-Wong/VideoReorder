import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import get_order_index

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

class PairWisePred(nn.Module):

    def __init__(self, **kwargs):
        super(PairWisePred, self).__init__()
        return

    def pair_frame_loss(self, feature_i, feature_j):
        '''
        i -> j
        '''
        phi_V_diff = torch.sub(feature_i, feature_j)
        zero = torch.zeros_like(phi_V_diff)
        phi_V_max_with_0 = torch.where(phi_V_diff < 0, zero, phi_V_diff)
        return torch.norm(phi_V_max_with_0)        

    def is_correct_order(self, feature_i, feature_j):
        # for i->j
        loss_i_j = self.pair_frame_loss(feature_i, feature_j)
        loss_j_i = self.pair_frame_loss(feature_j, feature_i)
        if loss_i_j < loss_j_i:
            return True
        else:
            return False

    def __call__(self, feature_list):
        '''
        input:
        feature_list = []
        
        output:
        pred_list = []
        '''
        N = len(feature_list)
        if N <= 1: return [i for i in range(N)]

        if self.is_correct_order(feature_list[0], feature_list[1]):
            ordered_list = [0, 1]
        else:
            ordered_list = [1, 0]
        
        feature_id = 1
        for feature in feature_list[2:]:
            '''
            insert feature_id into ordered list
            '''
            feature_id += 1

            if self.is_correct_order(feature, feature_list[ordered_list[0]]):
                ordered_list.insert(0, feature_id)
                continue

            ordered_id = 1 
            is_insert= False
            while ordered_id < len(ordered_list):
                if self.is_correct_order(feature, feature_list[ordered_list[ordered_id]]):
                    ordered_list.insert(ordered_id, feature_id)
                    is_insert = True
                    break
                ordered_id += 1
            if not is_insert: ordered_list.append(feature_id)
        
        pred_list = get_order_index(ordered_list)
        return pred_list





if __name__ == '__main__':
    LossFunc = PairWiseLoss()
    #input = torch.tensor([[1], [2], [3], [4]]).float()
    input = torch.tensor([[3], [4], [2], [1]]).float()
    loss = LossFunc(input)
    print(loss)