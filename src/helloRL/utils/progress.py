from tqdm import tqdm
import numpy as np

def color_text(text, hex_color):
    """Convert hex like '#9CDCFE' to colored text using truecolor ANSI"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'\033[38;2;{r};{g};{b}m{text}\033[0m'

def generate_sparkline_graph(data, data_width=20, max_graph_width=20):
    blocks = " ▁▂▃▄▅▆▇█"

    graph_width = min(data_width, max_graph_width)

    if type(data) is list:
        data = np.array(data)

    current_data_width = len(data)
    # how much of the bar will it take up
    current_data_visual_width = int((current_data_width / data_width) * graph_width)
    remaining_visual_width = graph_width - current_data_visual_width

    # the graph length should be based on the number of data points
    # if more than width, the data points should be downsampled to width
    # if less, it should use all points and leave space

    # data is a np array of loss values
    # it should be extended with zeros if less than width
    if len(data) < data_width:
        data = np.concatenate((data, np.zeros(data_width - len(data))))
        
    # downsample to width
    if len(data) > graph_width:
        step = len(data) / graph_width
        data = [data[int(i * step)] for i in range(graph_width)]

    min_d = min(data)
    max_d = max(data)
    span = max_d - min_d if max_d > min_d else 1
    normalized = [(d - min_d) / span for d in data]

    sparkline = "".join(blocks[int(round(v * (len(blocks) - 1)))] for v in normalized)

    # fill all the characters between current data width and graph width with '.'
    sparkline = color_text(sparkline[0:current_data_visual_width], "#80C4ED")
    sparkline += color_text('░' * remaining_visual_width, "#586A75")

    return sparkline
    
pb_progress_color = "#9BD5D0"
pb_variable_color = "#F5CB83"

class StepProgressBar:
    def __init__(self, title, n_steps, increments=1, value_title="Value", minigraph=False):
        # remove any existing instances that haven't been cleaned up
        instances = list(tqdm._instances)

        for instance in instances:
            instance.leave = False
            _ = instance.close()

        self.title = title
        self.n_steps = n_steps
        self.increments = increments
        self.minigraph = minigraph
        self.value_title = value_title

        custom_format = '{l_bar}{bar:20}| {n_fmt}/{total_fmt}, ⏱️={elapsed}, est=<{remaining}{postfix}]'
        self.pbar = tqdm(
            total=self.n_steps,
            desc=self.title,
            bar_format=custom_format,
            ascii='░█',  # [filled character, empty character]
        )

        self.values = []
    
    def update(self, value):
        if np.isnan(value):
            value = 0.0

        self.values.append(value)

        value_str = f"{value:.4f}"
        value_str = color_text(value_str, pb_variable_color)

        if self.minigraph:
            sparkline = generate_sparkline_graph(self.values, data_width=self.n_steps // self.increments)
            self.pbar.set_postfix({
                self.value_title: value_str,
                "Trend": sparkline
            })
        else:
            self.pbar.set_postfix({
                self.value_title: value_str
            })
        self.pbar.update(self.increments)

        if len(self.values) * self.increments == self.n_steps:
            self.close()

    def close(self):
        self.pbar.colour = 'GREEN'
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class EpochProgressBar(StepProgressBar):
    def __init__(self, title, n_epochs, minigraph=True):
        super().__init__(title, n_steps=n_epochs, increments=1, value_title="Loss", minigraph=minigraph)

class ProgressBar:
    def __init__(self, title, n_steps):
        # remove any existing instances that haven't been cleaned up
        instances = list(tqdm._instances)

        for instance in instances:
            instance.leave = False
            _ = instance.close()

        self.title = title
        self.n_steps = n_steps

        custom_format = '{l_bar}{bar:20}| {n_fmt}/{total_fmt}, ⏱️={elapsed}, est=<{remaining}{postfix}]'
        self.pbar = tqdm(
            total=self.n_steps,
            desc=self.title,
            bar_format=custom_format,
            ascii='░█',  # [filled character, empty character]
        )
    
    def increment(self, increment):
        self.pbar.update(increment)

        if self.pbar.n >= self.n_steps:
            self.close()

    def update_value(self, value):
        self.pbar.n = value
        self.pbar.refresh()

        if self.pbar.n >= self.n_steps:
            self.close()

    def close(self):
        self.pbar.colour = 'GREEN'
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class RemoteProgressBar(ProgressBar):
    def __init__(self, title, n_steps, n_sessions, total_machines=None):
        super().__init__(title, n_steps)

        self.n_sessions = n_sessions
        self.total_machines = total_machines if total_machines is not None else n_sessions
        self.completed_sessions = 0
        self._update_format()

    def _update_format(self):
        completed_steps_str = f"{self.pbar.n}/{self.n_steps}"
        completed_steps_str = color_text(completed_steps_str, pb_progress_color)

        if self.total_machines > self.n_sessions:
            # Show overflow context: "Leading X of Y machines"
            completed_sessions_str = f"{self.completed_sessions}/{self.n_sessions}"
            completed_sessions_str = color_text(completed_sessions_str, pb_progress_color)

            total_machines_str = str(self.total_machines)
            total_machines_str = color_text(total_machines_str, pb_variable_color)

            custom_format = (
                f'{{l_bar}}{{bar:20}}| Steps={completed_steps_str}, '
                f'Leading={completed_sessions_str} of {total_machines_str} machines, '
                f'⏱️={{elapsed}}, est=<{{remaining}}{{postfix}}]'
            )
        else:
            # Original format when no overflow
            completed_sessions_str = f"{self.completed_sessions}/{self.n_sessions}"
            completed_sessions_str = color_text(completed_sessions_str, pb_progress_color)
            custom_format = (
                f'{{l_bar}}{{bar:20}}| Steps={completed_steps_str}, '
                f'Sessions={completed_sessions_str}, '
                f'⏱️={{elapsed}}, est=<{{remaining}}{{postfix}}]'
            )

        self.pbar.bar_format = custom_format
        self.pbar.refresh()

    def update_completed_sessions(self, completed_sessions):
        self.completed_sessions = completed_sessions
        self._update_format()
        self.pbar.refresh()