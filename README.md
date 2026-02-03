# HelloRL
A fully modular framework to make Reinforcement Learning quick and easy

## Modal
Modal is a service for running batches of code remotely, useful for when a function needs to be run many times independently, as many workers can be spun up quickly, and return much faster than running the function x times locally.

Modal simply needs to setup auth, one time. Run `modal setup` or `python -m modal setup` on the command line, within the project. [Here is more info](https://modal.com/docs/guide).