import modal
import time
from functools import partial

from . import trainer
from .utils import plot

def gather_modal_results_for_calls(calls, n_timesteps, n_sessions, progress_dict, overflow_machines=0):
    """
    Monitor training progress and collect results from fastest machines.

    When overflow_machines > 0, will cancel slower machines once n_sessions
    complete and return only the fastest results.
    """
    total_machines = len(calls)
    threshold = n_sessions
    total_timesteps = n_sessions * n_timesteps

    # Track completion state
    completed_call_indices = set()
    cancelled_call_indices = set()
    failed_call_indices = set()

    # this file is imported to modal, but progress bar shouldn't be used there
    from .utils.progress import RemoteProgressBar

    with RemoteProgressBar("Training", n_steps=total_timesteps, n_sessions=n_sessions, total_machines=total_machines) as bar:
        while True:
            time.sleep(0.2)  # Poll every 0.2 seconds

            # Check for newly completed machines
            for i in range(total_machines):
                if i in completed_call_indices or i in cancelled_call_indices or i in failed_call_indices:
                    continue

                # Check if machine reports completion
                value = progress_dict.get(i)
                if value is not None and value >= n_timesteps:
                    try:
                        # Verify completion by polling (non-blocking)
                        calls[i].get(timeout=0)
                        completed_call_indices.add(i)
                    except modal.exception.TimeoutError:
                        # Still running
                        pass
                    except Exception as e:
                        # Machine failed
                        print(f"Warning: Machine {i} failed: {e}")
                        failed_call_indices.add(i)

            # Calculate progress from leading machines for smooth display
            all_progress = []
            for i in range(total_machines):
                if i in failed_call_indices:
                    continue
                value = progress_dict.get(i)
                if value is not None:
                    all_progress.append((i, value))

            # Sort by progress, take top threshold machines
            all_progress.sort(key=lambda x: x[1], reverse=True)
            leading_machines = all_progress[:threshold]
            completed_timesteps = sum(val for _, val in leading_machines)

            # Count completed sessions from leading machines
            completed_sessions = sum(1 for _, val in leading_machines if val >= n_timesteps)

            bar.update_completed_sessions(completed_sessions)
            bar.update_value(completed_timesteps)

            # Once threshold reached, cancel remaining machines
            if len(completed_call_indices) >= threshold:
                for i in range(total_machines):
                    if i not in completed_call_indices and i not in failed_call_indices:
                        try:
                            calls[i].cancel(terminate_containers=True)
                            cancelled_call_indices.add(i)
                        except Exception as e:
                            print(f"Warning: Failed to cancel machine {i}: {e}")
                break

            # Check if threshold is unreachable
            active_or_completed = total_machines - len(failed_call_indices)
            if active_or_completed < threshold:
                raise RuntimeError(
                    f"Cannot reach threshold of {threshold} machines. "
                    f"Completed: {len(completed_call_indices)}, "
                    f"Failed: {len(failed_call_indices)}"
                )

    # Gather results only from first threshold completed machines
    completed_sorted = sorted(completed_call_indices)[:threshold]
    completed_calls = [calls[i] for i in completed_sorted]

    results = modal.FunctionCall.gather(*completed_calls)

    return results

def train_session_on_modal_with_func(train_func, session_id, n_timesteps, progress_dict):
    def progress_callback(current_timestep):
        progress_dict[session_id] = current_timestep

    train_func_return = train_func(progress_callback=progress_callback)
    progress_callback(n_timesteps)

    return train_func_return

def create_modal_train_function(app, image, timeout=3600):
    @app.function(image=image, timeout=timeout, serialized=True)
    def _modal_train(n_timesteps, setup_func, session_id=None, progress_dict=None):
        agent, env_name, continuous, params = setup_func()

        training_func = partial(trainer.train, agent, env_name, continuous=continuous, 
                               n_timesteps=n_timesteps, should_print=False)
        
        train_results = train_session_on_modal_with_func(training_func, session_id, 
                                                          n_timesteps, progress_dict)
        
        return (*train_results, agent, env_name, continuous, params)
    
    return _modal_train

def train(n_sessions, n_timesteps, setup_func, app, image, timeout=3600, overflow_machines=0):
    """
    Train using Modal with optional overflow machines for robustness.

    Args:
        n_sessions: Number of training sessions needed (threshold)
        n_timesteps: Timesteps per session
        setup_func: Function to create agent and environment
        app: Modal app instance
        image: Modal image with dependencies
        timeout: Max time per machine in seconds
        overflow_machines: Extra machines to spawn as buffer (default 0)

    Returns:
        List of results from the fastest n_sessions machines
    """
    total_machines = n_sessions + overflow_machines
    # Create the modal function before entering app.run() so it gets registered
    modal_train = create_modal_train_function(app, image, timeout)

    with app.run():
        progress_dict = modal.Dict.from_name("training-progress", create_if_missing=True)
        progress_dict.clear()

        # Spawn total_machines instead of just n_sessions
        calls = [modal_train.spawn(n_timesteps, setup_func=setup_func, session_id=i, progress_dict=progress_dict
        ) for i in range(total_machines)]

        results = gather_modal_results_for_calls(calls, n_timesteps, n_sessions, progress_dict, overflow_machines=overflow_machines)

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