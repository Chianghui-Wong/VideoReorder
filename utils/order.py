import torch
from torch import nn
import numpy as np
import copy

def beam_sort_and_del(beam_candidate, beam_size):
    beam_candidate = sorted(beam_candidate, reverse=True, key=lambda x:x['score'])
    if len(beam_candidate) > beam_size:
        beam_candidate = beam_candidate[:beam_size]
    return beam_candidate

def beam_step_forward(beam_candidate, score_square, beam_size):
    beam_candidate_new = []
    for beam_candidate_ele in beam_candidate:
        # next rode is p -> q
        p = beam_candidate_ele['path'][-1]
        q_list = beam_candidate_ele['rest']

        q_list = sorted(q_list, reverse=True, key = lambda q:score_square[p][q])
        if len(q_list) > beam_size:
            q_list = q_list[:beam_size]

        for q in q_list:
            beam_candidate_new_ele = copy.deepcopy(beam_candidate_ele)
            beam_candidate_new_ele['score'] += score_square[p][q]
            beam_candidate_new_ele['path'].append(q)
            beam_candidate_new_ele['rest'].remove(q)
            beam_candidate_new.append(beam_candidate_new_ele)

    # print(beam_candidate_new)
    return beam_candidate_new

def beam_search(score_square, begin_idx=0, beam_size=5):
    '''
    score_square is two dimension square array, score_square[i][j] means val(i->j)
    
    for example:
        score_sqare =   [
                            [-inf, 4, 3, 2, 1],
                            [4, -inf, 3, 2, 1],
                            [3, 4, -inf, 2, 1],
                            [2, 3, 4, -inf, 1],
                            [1, 2, 3, 4, -inf]
                        ]

    '''
    N = len(score_square)
    beam_candidate = [
        {
            'score': 0,
            'path' : [begin_idx],
            'rest' : [i for i in range(N)]
        }
    ]
    beam_candidate[0]['rest'].remove(begin_idx)
            
    for step_num in range(N-1):
        beam_candidate = beam_step_forward(beam_candidate, score_square, beam_size)
        if step_num == N-3: continue
        beam_candidate = beam_sort_and_del(beam_candidate, beam_size)

    return beam_candidate[0]['path']

if __name__ == '__main__':
    a =  [
            [float('-inf'), 4, 3, 2, 1],
            [4, float('-inf'), 1, 2, 1],
            [3, 4, float('-inf'), 1, 1],
            [2, 3, 4, float('-inf'), 1],
            [1, 2, 3, 4, float('-inf')]
        ]
    print(beam_search(a, beam_size=5))