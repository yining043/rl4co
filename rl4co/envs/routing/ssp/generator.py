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
    overlap_lengths_expanded = overlap_lengths.unsqueeze(-1).expand(-1, -1, str_dim)
    start_indices = str_dim - overlap_lengths_expanded

    # Generate a mask for the overlap regions
    overlap_region_mask = overlap_indices < overlap_lengths_expanded

    # Copy the values to the overlap region
    previous_strings = batch_data[:, :-1, :].clone()
    for i in range(str_dim):
        current_mask = overlap_region_mask[:, :, i]
        batch_data[:, 1:, i][current_mask] = previous_strings[:, :, -1][current_mask]
    
    return batch_data.float()
