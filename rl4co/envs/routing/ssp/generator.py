from typing import Callable, Union

from tensordict.tensordict import TensorDict
import torch
import numpy as np
import hashlib
import random
import math
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SSPGenerator(Generator):

    def __init__(
        self,
        num_loc: int = 10,
        fixed_len: int = 8,
        min_loc: float = 0,
        max_loc: float = 1,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.fixed_len = fixed_len

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        batch_size = batch_size[0] if isinstance(batch_size, list) else batch_size
        fixed_len = self.fixed_len
        codes = generate_batch_superstring_data(batch_size, self.num_loc, fixed_len)        
        return TensorDict({"codes": codes}, batch_size=batch_size)

def generate_batch_superstring_data(batch_size, num_str, str_dim, alphabet_size=2):
    # Generate random strings
    batch_data = torch.randint(0, alphabet_size, (batch_size, num_str, str_dim))
    
    # Generate random overlap masks
    overlap_mask = torch.rand(batch_size, num_str - 1) > 0.5
    overlap_lengths = torch.randint(1, str_dim // 2 + 1, (batch_size, num_str - 1))
    
    # Generate index tensors for efficient slicing
    overlap_indices = torch.arange(str_dim).expand(batch_size, num_str - 1, str_dim)
    overlap_mask_expanded = overlap_mask.unsqueeze(-1).expand(batch_size, num_str - 1, str_dim)
    overlap_lengths_expanded = overlap_lengths.unsqueeze(-1).expand(batch_size, num_str - 1, str_dim)

    # Generate a mask for the overlap regions
    overlap_region_mask = (overlap_indices < overlap_lengths_expanded) & overlap_mask_expanded
    
    # Copy the values to the overlap region
    previous_strings = batch_data[:, :-1, :].clone()
    for i in range(str_dim):
        current_mask = overlap_region_mask[:, :, i]
        selected_overlap_index_at_i = (str_dim - overlap_lengths + i).view(-1,1) % str_dim
        selected_overlap = previous_strings.view(-1, str_dim).gather(1, selected_overlap_index_at_i).view(batch_size, num_str - 1)
        batch_data[:, 1:, i][current_mask] = selected_overlap[current_mask]
    
    # Shuffle the num_str dimension
    perm = torch.rand(batch_size, num_str).argsort(dim=1)
    batch_data = batch_data[torch.arange(batch_size).unsqueeze(1), perm]
    
    return batch_data.float()
