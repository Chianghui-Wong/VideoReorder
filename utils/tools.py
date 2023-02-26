import torch
from torch import nn
import numpy as np
import copy

def class_start_zero(class_label):
    '''
    input = [5, 4, 3, 4, 5]
    output = [2, 1, 0, 1, 2] 
    '''
    return [i-min(class_label) for i in class_label]

def torch_to_list(input):
    if torch.is_tensor(input): 
        output = input.detach().cpu().numpy().tolist()   
    return output

def get_order_list(input):
    '''
    input: a list of float 
    eg: [7.5, -3.5, 2.5]

    output: the order of the float
    eg: [2, 0, 1]
    '''
    input = torch_to_list(input)
    ordered = sorted(input)
    output = []
    for i in input:
        for idx, j in enumerate(ordered):
            if i == j:
                output.append(idx)
                break

    return output

def get_order_index(input):
    '''
    input: a order_idx list 
    eg: [2, 0, 1]

    ouput: the position of 0st 1st 2nd 3rd ...
    eg: [1, 2, 0]
    '''
    input = torch_to_list(input)
    sorted_nums = sorted(enumerate(input), key=lambda x:x[1])
    output = [i[0] for i in sorted_nums]

    return output

def same_shuffle(input, template):
    '''
    template is a order of range(N)
    shuffle the input with the same order
    '''
    ZIP = zip(input, template)
    ZIP = sorted(ZIP, key=lambda x:x[1])

    output, _ = zip(*ZIP)

    return list(output)

def list_to_one_dim(input):
    output = []

    def go_deep(arr, output):
        if isinstance(arr, list):
            for ele in arr:
                go_deep(ele, output)
        else:
            output.append(arr)
            return

    go_deep(input, output)

    return output

def list_del_final_dim(input):

    def go_deep(arr):
        output = []

        if not isinstance(arr[0][0], list): 
            output = [i for j in arr for i in j]
            return output

        for ele in arr:
            output.append(go_deep(ele))
        
        return output
    
    return go_deep(input)

def list_del_first_dim(input):

    return [i for j in input for i in j]



def group_by_class(input, class_label):
    '''
    input and class_label are both list
    class_label is one demension
    input have same length on dim=0 with class_label

    for example:
        input =         [0, 1, 2, 3, 4, 5]
        class_label = [2, 1, 0, 1, 2, 2]
        output = [[2], [1, 3], [0, 4, 5]]
    '''

 

    n_class = max(class_label) - min(class_label) + 1
    output = [[] for _ in range(n_class)]
    for idx, label in enumerate(class_label):
        output[label].append(input[idx])

    return output

def group_same_with(input, template):
    '''
    input and template are both list
    template have one more dimension than input
    add one dimension with the same way

    for example:
        input = [0, 1, 2, 3, 4, 5]
        template = [[2], [1, 3], [0, 4, 5]]
        output = [[0], [1,2], [3, 4, 5]]
    '''
    group_len = [len(i) for i in template]

    idx = 0
    output = []
    for group_len_ele in group_len:
        output_ele = []
        for _ in range(group_len_ele):
            output_ele.append(input[idx])
            idx += 1
        output.append(output_ele)
    
    return output

if __name__ == '__main__':
    input = [[0, 0.1, 0,5], [1, 2, 3,4, 5,6]]
    template = [4, 3, 2, 1]

    print(list_del_first_dim(input))

    