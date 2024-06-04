from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import SSPGenerator
from .render import render

log = get_pylogger(__name__)


class SSPEnv(RL4COEnvBase):

    name = "ssp"

    def __init__(
        self,
        generator: SSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = SSPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]

        # # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.sum(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        device = td.device
        init_locs = td["codes"]

        # We do not enforce loading from self for flexibility
        num_loc = init_locs.shape[-2]

        # Other variables
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": init_locs,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: SSPGenerator):
        self.observation_spec = CompositeSpec(
            codes=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.bool)

    def _get_reward(self, td, actions) -> TensorDict:
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Gather locations in order of tour and return distance between them (i.e., -reward)
        left = gather_by_index(td["codes"], actions)
        right = torch.roll(left, -1, dims=-2)

        # Flatten the tensors for simplicity in handling
        bs, gs, d = left.size()
        left_temp = left.view(-1, d)
        right_temp = right.view(-1, d)
        
        # Initialize a tensor to store the maximum overlap lengths
        max_overlaps = torch.zeros(left_temp.shape[0], dtype=torch.long)
        
        # Check for overlaps of varying lengths
        for i in range(1, d + 1):  # Starting from 1 to avoid empty slice and up to d
            # Compare suffix of left_temp with prefix of right_temp for each length i
            matches = (left_temp[:, -i:] == right_temp[:, :i]).all(dim=1)
            
            # Update the maximum overlap lengths where matches are found
            max_overlaps = torch.where(matches.cpu(), i, max_overlaps.cpu())
        return (max_overlaps.view(bs, gs)[:,:-1].sum(-1) - self.generator.fixed_len * self.generator.num_loc)/ self.generator.num_loc
        
    def generate_data(self, batch_size) -> TensorDict:
        batch_size = batch_size[0] if isinstance(batch_size, list) else batch_size
        fixed_len = self.generator.fixed_len
        codes = torch.zeros(batch_size, self.generator.num_loc, fixed_len)
        
        for i in range(batch_size):
            # S = generator.covering_code();
            # array = np.array([list(map(int, binary_string)) for binary_string in S], dtype=np.float64)
            # np.random.shuffle(array)
            # codes[i] = torch.tensor(array[:self.generator.num_loc])
            codes[i] = generate_superstring_data(self.generator.num_loc, fixed_len)
        
        return TensorDict({"codes": codes}, batch_size=batch_size)
    
    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are visited exactly once"""
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor=None, ax = None):
        return render(td, actions, ax)