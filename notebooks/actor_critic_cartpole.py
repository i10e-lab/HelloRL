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
# Actor Critic, CartPole

# %%
import torch
import numpy as np
import gymnasium as gym

from helloRL.modular.actors import *
from helloRL.modular.critics import *
from helloRL.modular.agents import *
from helloRL.modular.params import *
from helloRL.modular import trainer
from helloRL.utils import plot

# %%
env_name = 'CartPole-v1'
continuous = False
n_timesteps = 100000

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = DiscreteActor(state_dim=state_dim, action_dim=action_dim)
critic = Critic(state_dim=state_dim)
agent = Agent(actor=actor, critics=[critic])
params = Params()

returns, lengths = trainer.train(agent, env_name, continuous, params, n_timesteps)

# %%
# show plot
plot.plot_session(
    returns, lengths, 'Actor Critic', env_name, continuous=continuous,
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
from helloRL.utils import modal_training
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

    actor = StochasticActor(state_dim=state_dim, action_dim=action_dim, action_range=action_range)
    critic = Critic(state_dim=state_dim)
    agent = Agent(actor=actor, critics=[critic])
    params = Params()

    return agent, env_name, continuous, params


# %%
n_sessions = 16
n_timesteps = 100000
timeout = 3600 # 1 hour
setup_func = standard

results = modal_training.train(n_sessions, n_timesteps, setup_func, app, image, timeout=timeout)

# %%
modal_training.plot_results(results, 'Actor Critic', n_timesteps)
