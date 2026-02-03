import modal
import time
from functools import partial

from helloRL.modular import trainer
from helloRL.utils import plot

def gather_modal_results_for_calls(calls, n_timesteps, n_sessions, progress_dict):
    # Monitor progress by polling the Dict
    total_timesteps = n_sessions * n_timesteps

    # this file is imported to modal, but progress bar shouldn't be used there
    from .progress import RemoteProgressBar
    
    with RemoteProgressBar("Training", n_steps=total_timesteps, n_sessions=n_sessions) as bar:
        while True:
            time.sleep(0.2)  # Poll every 0.2 seconds
            
            # Sum up completed timesteps across all sessions
            completed_timesteps = 0
            completed_sessions = 0
            
            for i in range(n_sessions):
                value = progress_dict.get(i)

                if value is None:
                    continue

                completed_timesteps += value

                if value == n_timesteps:
                    completed_sessions += 1

            bar.update_completed_sessions(completed_sessions)
            bar.update_value(completed_timesteps)

            if completed_sessions >= n_sessions:
                break
    
    # Gather results
    results = modal.FunctionCall.gather(*calls)

    return results

app = modal.App(name='helloRL')

image = modal.Image.debian_slim()\
    .apt_install('swig', 'build-essential')\
        .pip_install("gymnasium", "torch", "gymnasium[box2d]", "numpy", "stable_baselines3")\
        .add_local_dir('src/modular', remote_path='/root/src/modular')\
        .add_local_python_source('src.session_tracker', 'src.modal_training')






def train_session_on_modal_with_func(train_func, session_id, n_timesteps, progress_dict):
    def progress_callback(current_timestep):
        progress_dict[session_id] = current_timestep

    train_func_return = train_func(progress_callback=progress_callback)
    progress_callback(n_timesteps)

    return train_func_return


@app.function(image=image, timeout=86400)
def _modal_train(n_timesteps, setup_func, session_id=None, progress_dict=None):
    agent, env_name, continuous, params = setup_func()

    training_func = partial(trainer.train, agent, env_name, continuous=continuous, n_timesteps=n_timesteps, should_print=False)

    train_results = train_session_on_modal_with_func(training_func, session_id, n_timesteps, progress_dict)

    return (*train_results, agent, env_name, continuous, params)

def train(n_sessions, n_timesteps, setup_func):
    with app.run():
        progress_dict = modal.Dict.from_name("training-progress", create_if_missing=True)
        progress_dict.clear()

        calls = [_modal_train.spawn(n_timesteps, setup_func=setup_func, session_id=i, progress_dict=progress_dict
        ) for i in range(n_sessions)]
        
        results = gather_modal_results_for_calls(calls, n_timesteps, n_sessions, progress_dict)

    return results

def plot_results(results, title, n_timesteps, nb_name=None, save_dir=None):
    plot_results = [(returns, lengths) for returns, lengths, _, _, _, _ in results]
    _, _, agent, env_name, continuous, params = results[0]

    plot.plot_sessions(
        plot_results,
        title,
        env_name=env_name,
        continuous=continuous,
        params=params,
        agent=agent,
        n_timesteps=n_timesteps,
        nb_name=nb_name,
        save_dir=save_dir
    )