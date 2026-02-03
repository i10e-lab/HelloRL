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

# %%
import random
import torch

from helloRL.utils.session_tracker import SessionTracker

# %%
with SessionTracker(n_timesteps=10000, print_interval=100) as tracker:
    for i in range(10000):

        # this is just to make it take some time without inserting a sleep
        a = torch.rand(400, 400)

        if i % 100 == 0:
            r = random.randint(0, 100)
            tracker.finish_episode(r, r)

        tracker.increment_timestep(1)


# %%
def collect_data_for_rollout(tracker):
    while not tracker.is_session_complete():
        tracker.increment_timestep()

        done = random.random() < 0.05
        
        if done:
            episode_return = random.randint(0, 100)
            episode_length = episode_return

            tracker.finish_episode(episode_return, episode_length)
            break


# %%
n_timesteps = 100000

# can be used with other typical steps

with SessionTracker(n_timesteps=n_timesteps, print_interval=1000, window_length=64) as tracker:
    while not tracker.is_session_complete():
        a = torch.rand(400, 400)
        rollout_data = collect_data_for_rollout(tracker)


# %%
# can be used inside a function 

def go():
    n_timesteps = 100000

    with SessionTracker(n_timesteps=n_timesteps, print_interval=1000, window_length=64) as tracker:
        while not tracker.is_session_complete():
            a = torch.rand(400, 400)
            rollout_data = collect_data_for_rollout(tracker)

go()

# %%
# incomplete

tracker = SessionTracker(n_timesteps=1000)
collect_data_for_rollout(tracker)
# tracker.close()

# %%
# tracker without progress

n_timesteps = 100000

with SessionTracker(n_timesteps=n_timesteps, should_print=False, print_interval=1000, window_length=64) as tracker:
    while not tracker.is_session_complete():
        a = torch.rand(400, 400)
        rollout_data = collect_data_for_rollout(tracker)
