# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: hellorl
#     language: python
#     name: python3
# ---

# %% [markdown]
# PPO, Lunar Lander Continuous
#
# This implementation matches SB3-Zoo.
#
# Not supported, from SB3-Zoo implementation:
# - vf_coef, we have separate actor/critic so not relevant

# %%
import torch
import numpy as np
import gymnasium as gym

from helloRL.modules.actors import *
from helloRL.modules.critics import *
from helloRL.modules.agents import *
from helloRL.modules.params import *
from helloRL.modules.a2c import *
from helloRL.modules.grad_norm import *
from helloRL.modules.gae import *
from helloRL.modules.lr_anneal import *
from helloRL.modules.epochs import *
from helloRL.modules.po_clipped import *
from helloRL import trainer
from helloRL.utils import plot

# %%
seed = 0
torch.manual_seed(seed if seed is not None else torch.seed())
np.random.seed(seed)

env_name = 'LunarLander-v3'
continuous = True
n_timesteps = 400000

env = gym.make(env_name, continuous=continuous)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = torch.tensor(np.stack([env.action_space.low, env.action_space.high]))

actor_params = DistributionalActorParams(
    policy_objective_method=PolicyObjectiveMethodClipped(clip_range=0.2),
    entropy_coef=0.01
)
actor = StochasticActor(state_dim=state_dim, action_dim=action_dim, action_range=action_range, params=actor_params)
critic = Critic(state_dim=state_dim)
agent = Agent(actor=actor, critics=[critic])
params = Params(
    rollout_method=RolloutMethodA2C(n_steps=1024, n_envs=16),
    gradient_transform=GradientTransformClipNorm(max_norm=0.5),
    advantage_method=AdvantageMethodGAE(lambda_=0.98),
    advantage_transform=AdvantageTransformNormalize(),
    data_load_method=DataLoadMethodEpochs(n_epochs=4, mb_size=64),
    lr_schedule=LRScheduleLinearAnneal(start_lr=0.0003, end_lr=0.0),
    gamma=0.999
)

returns, lengths = trainer.train(agent, env_name, continuous, params, n_timesteps, seed=seed)

# %%
# show plot
plot.plot_session(
    returns, lengths, 'PPO', env_name, continuous=continuous,
    n_timesteps=n_timesteps, agent=agent, params=params)

# %% [markdown]
# ---
#
# Train many sessions at once on Modal (much faster than running locally).
#
# Modal simply needs to setup auth, one time. Run `modal setup`. [Here is more info](https://modal.com/docs/guide). I’m not affiliated with Modal.
#
# The `image` code below is simplified in your own codebase, as you can simply use:
#
# ```python
# image = modal.Image.debian_slim().apt_install('swig', 'build-essential').pip_install("helloRL")
# ```

# %%
from helloRL import modal_training
import modal
from pathlib import Path

# %%
app = modal.App(name='helloRL')

project_root = Path().resolve().parent
repo_root = project_root / "src"

image = modal.Image.debian_slim()\
    .apt_install('swig', 'build-essential')\
    .pip_install_from_pyproject("../pyproject.toml")\
    .add_local_dir(str(repo_root), remote_path="/root")


# %%
def standard():
    env_name = 'LunarLander-v3'
    continuous = True

    env = gym.make(env_name, continuous=continuous)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = torch.tensor(np.stack([env.action_space.low, env.action_space.high]))

    actor_params = DistributionalActorParams(
        policy_objective_method=PolicyObjectiveMethodClipped(clip_range=0.2),
        entropy_coef=0.01
    )
    actor = StochasticActor(state_dim=state_dim, action_dim=action_dim, action_range=action_range, params=actor_params)
    critic = Critic(state_dim=state_dim)
    agent = Agent(actor=actor, critics=[critic])
    params = Params(
        rollout_method=RolloutMethodA2C(n_steps=1024, n_envs=16),
        gradient_transform=GradientTransformClipNorm(max_norm=0.5),
        advantage_method=AdvantageMethodGAE(lambda_=0.98),
        advantage_transform=AdvantageTransformNormalize(),
        data_load_method=DataLoadMethodEpochs(n_epochs=4, mb_size=64),
        lr_schedule=LRScheduleLinearAnneal(start_lr=0.0003, end_lr=0.0),
        gamma=0.999
    )

    return agent, env_name, continuous, params


# %%
n_sessions = 16
n_timesteps = 400000
timeout = 3600 # 1 hour
setup_func = standard

results = modal_training.train(n_sessions, n_timesteps, setup_func, app, image, timeout=timeout)

# %%
modal_training.plot_results(results, 'PPO', n_timesteps)
