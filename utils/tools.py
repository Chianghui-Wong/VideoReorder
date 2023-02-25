import torch
from torch import nn
import numpy as np
import copy

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



if __name__ == '__main__':
    input = [[0, [0.1, 0,5]], [1, 2], [3,4], [5,6]]
    template = [4, 3, 2, 1]


    print(list_to_one_dim(input))
    