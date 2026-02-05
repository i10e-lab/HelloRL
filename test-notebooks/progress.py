# ---
# jupyter:
#   jupytext:
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
# For testing the features of utils.progress
#
# Progress utility: shows little progress bars with trend graphs, to track training in a more sophisticated way than printing many lines.
#
# It uses a CLI within notebooks, so that it displays nicely in-line.
#
# Also has features to work with remote training runs. It simulates it for these tests, it doesn't actually do any remote training or require any account.

# %%
import random
import time

from helloRL.utils import progress

n_epochs = 140
bar = progress.StepProgressBar("Training", n_steps=n_epochs, minigraph=True)

for epoch in range(n_epochs):
    loss = random.random()

    time.sleep(0.05)
    bar.update(loss)

# %%
import random
import time

from helloRL.utils import progress

n_steps = 1000
increments = 10

# this is safer, as it handles closing the bar even if there's an error or exception
with progress.StepProgressBar("Training", n_steps=n_steps, increments=increments, minigraph=True) as bar:
    for step in range(0, n_steps, increments):
        score = random.random()

        time.sleep(0.05)
        bar.update(score)

# %%
import random
import time

from helloRL.utils import progress

n_epochs = 40

# this is safer, as it handles closing the bar even if there's an error or exception
with progress.EpochProgressBar("Training", n_epochs=n_epochs, minigraph=True) as bar:
    for epoch in range(n_epochs):
        loss = random.random()

        time.sleep(0.05)
        bar.update(loss)

# %%
import random
import time

from helloRL.utils import progress

n_epochs = 40

def go():
    # this is safer, as it handles closing the bar even if there's an error or exception
    with progress.EpochProgressBar("Training", n_epochs=n_epochs, minigraph=True) as bar:
        for epoch in range(n_epochs):
            loss = random.random()

            time.sleep(0.05)
            bar.update(loss)

go()

# %% [markdown]
# Modal progress bar

# %%
import time

from helloRL.utils import progress

n_timesteps = 1000

# this is safer, as it handles closing the bar even if there's an error or exception
with progress.ProgressBar("Training", n_steps=n_timesteps) as bar:
    for i in range(n_timesteps // 10):
        bar.increment(10)
        time.sleep(0.05)

# %%
import time

from helloRL.utils import progress

n_timesteps = 1000

completed_sessions = 0

# this is safer, as it handles closing the bar even if there's an error or exception
with progress.RemoteProgressBar("Training", n_steps=n_timesteps, n_sessions=10) as bar:
    for i in range(n_timesteps):

        if i % 10 == 0:
            bar.increment(10)

        if i % 100 == 0:
            completed_session += 1
            bar.update_completed_sessions(completed_sessions)

        time.sleep(0.005)

# %%
import time

from helloRL.utils import progress

n_timesteps = 1000

completed_sessions = 0

# this is safer, as it handles closing the bar even if there's an error or exception
with progress.RemoteProgressBar("Training", n_steps=n_timesteps, n_sessions=10) as bar:
    for i in range(n_timesteps):

        bar.update_value(i)

        if i % 100 == 0:
            completed_sessions += 1
            bar.update_completed_sessions(completed_sessions)

        time.sleep(0.005)
