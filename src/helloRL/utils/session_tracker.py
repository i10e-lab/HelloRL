import numpy as np

class SessionTracker():
    def __init__(self, n_timesteps, should_print=True, print_interval=1000, window_length=64, progress_callback=None):
        self.current_timestep = 0
        self.n_timesteps = n_timesteps
        self.should_print = should_print
        self.print_interval = print_interval
        self.window_length = window_length
        self.progress_callback = progress_callback

        self.all_returns = []
        self.all_lengths = []

        if self.should_print:
            import helloRL.utils.progress as progress

            self.bar = progress.StepProgressBar(
                "Training", n_steps=n_timesteps, increments=print_interval, value_title="Score", minigraph=True)
        else:
            self.bar = None

    def is_session_complete(self):
        return self.current_timestep >= self.n_timesteps

    def increment_timestep(self, n=1):
        previous_timestep = self.current_timestep

        self.current_timestep += n

        if self.print_interval and\
            previous_timestep // self.print_interval != self.current_timestep // self.print_interval:
            recent_mean = np.mean(self.all_returns[-self.window_length:])

            if self.should_print:
                self.bar.update(recent_mean)

            # Call progress callback if provided
            if self.progress_callback is not None:
                self.progress_callback(self.current_timestep)

        return self.is_session_complete()
    
    def finish_episode(self, episode_return, episode_length):
        self.all_returns.append(episode_return)
        self.all_lengths.append(episode_length)

    def finish_episodes(self, episode_returns, episode_lengths):
        self.all_returns.extend(episode_returns)
        self.all_lengths.extend(episode_lengths)

    def close(self):
        if self.should_print:
            self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.should_print:
            self.bar.close()