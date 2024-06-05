import torch
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger
torch.cuda.device_count()

from rl4co.envs import SSPEnv
from rl4co.models.zoo.am import AttentionModelPolicy, AttentionModel
from rl4co.utils.trainer import RL4COTrainer

from rl4co.utils.decoding import random_policy, rollout
from rl4co.utils.ops import gather_by_index
class SSPInitEmbedding(nn.Module):

    def __init__(self, embedding_dim, fixed_len, linear_bias=True):
        super(SSPInitEmbedding, self).__init__()
        node_dim = fixed_len  # x, y
        self.init_embed = nn.Linear(node_dim, embedding_dim, linear_bias)

    def forward(self, td):
        out = self.init_embed(td["codes"])
        return out

class SSPContext(nn.Module):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embedding_dim,  linear_bias=True):
        super(SSPContext, self).__init__()
        self.W_placeholder = nn.Parameter(
            torch.Tensor(embedding_dim).uniform_(-1, 1)
        )
        self.project_context = nn.Linear(
            embedding_dim, embedding_dim, bias=linear_bias
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["current_node"].dim() == 1 else (td["current_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)
        
class StaticEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0

num_loc = 100
fixed_len = 15
emb_dim = 128

env = SSPEnv(generator_params={"num_loc":num_loc,
                              "fixed_len":fixed_len})

policy = AttentionModelPolicy(env_name = env.name,
                              embed_dim=emb_dim,
                              num_encoder_layers=6,
                              num_heads=8,
                              normalization="instance",
                              init_embedding=SSPInitEmbedding(emb_dim, fixed_len),
                              context_embedding=SSPContext(emb_dim),
                              dynamic_embedding=StaticEmbedding(emb_dim),
                              use_graph_context=False
                             )

# Model: default is AM with REINFORCE and greedy rollout baseline
model = AttentionModel(env, 
            policy=policy,
            batch_size=500,
            train_data_size=100000,  # each epoch,
            val_batch_size=1000,
            val_data_size=1000,
            test_batch_size=1000,
            test_data_size=1000,
            optimizer="Adam",
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs={"milestones": [1901, ], "gamma": 0.1},
        )

# Checkpointing callback: save models when validation reward improves
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints_ssp", # save to checkpoints/
                                    filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
                                    save_top_k=5, # save only the best model
                                    save_last=True, # save the last model
                                    monitor="val/reward", # monitor validation reward
                                    mode="max") # maximize validation reward

rich_model_summary = RichModelSummary(max_depth=3)  # model summary callback
callbacks = [checkpoint_callback, rich_model_summary]

# Logger
logger = WandbLogger(project="ssp", name=f"{env.name}_{num_loc}")
# logger = None # uncomment this line if you don't want logging



# We use our own wrapper around Lightning's `Trainer` to make it easier to use
trainer = RL4COTrainer(max_epochs=2000, 
                       accelerator = 'gpu', 
                       devices=1,   
                       logger=logger,
                       callbacks=callbacks,
                      )

trainer.test(model)
trainer.fit(model)