# BehaviouralPsychologyRewardSchemesRL

@Author Trang Thanh Mai Duyen
This project implements some of the classic control environment of Gymnasium (Cart pole and Mountain car) using behavioural psychology inspired reward schemes (fixed ratio, fixed interval, variable ratio, variable interval ) to Q-learning, SARSA and 3-step Expected SARSA algorithms.

Requires to install some additional python packages:

- gymnasium
- numpy
- matplotlib
- pickle

Some files' description:

- main.py: every algorithms, variable reward schedules and environments can be run from here. However, they are put as comments, to use it please uncomment (can use find tool to search). Only necessary environments setup (cart pole, mountain car and their noises) are not put as comments.
  - Environments include: Cart Pole and Mountain Car.
  - Noisy types: Normal (None), Random Uniform Noisy Reward, and Random Normal Noisy Observation.
  - Algorithms: Q-learning, SARSA, and 3-step Expected SARSA.
  - Variable reward schedules: fixed-ratio, fixed-interval, variable-interval, and variable-ratio).
- auxFunctions.py: variable reward schedules implementation.
- VisualizeResults.py: sketching graphs implementation for evaluation purposes.

* Both cart pole and mountain car environments has parameters: noisy_environment default is normal, scaling_factor default is 0.1, reward_schedule default is variable ratio. Those parameters can be set and executed in main.py

For files' structure, please prefer to the provided Class Diagram.
