import torch
import numpy as np
import matplotlib.pyplot as plt

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

def tensor_to_binary_strings(tensor):
    return [''.join(map(str, map(int, row))) for row in tensor]

def calculate_max_overlap(a, b):
    max_overlap = 0
    # Check overlap up to the full length of string a
    for i in range(1, len(a) + 1):
        if a[-i:] == b[:i]:
            max_overlap = i
    return max_overlap

def construct_superstring(strings):
    superstring = strings[0]
    for i in range(1, len(strings)):
        overlap = calculate_max_overlap(superstring, strings[i])
        superstring += strings[i][overlap:]
    return superstring

def render(td, actions=None, ax=None):
    # Convert tensor to list of binary strings
    binary_strings = tensor_to_binary_strings(gather_by_index(td["codes"], actions)[0])
    
    # Construct the superstring
    print('SSP codes:\n', td['codes'][0], '\n\n Suggested order:', actions[0], '\n\n Sorted codes according to the order:\n', gather_by_index(td["codes"], actions)[0])
    superstring = construct_superstring(binary_strings)
    print(f"\nConstructed Superstring: {superstring}")
    print(f"Superstring Length: {len(superstring)}")
