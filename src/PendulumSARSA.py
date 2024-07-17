# Adapted from "https://github.com/BhanuPrakashPebbeti/Q-Learning_and_Double-Q-Learning/blob/main/Q-Learning/Pendulum-qlearning.py"
import gymnasium as gym
import numpy as np
import random


env = gym.make('Pendulum-v1')
# env.metadata['render_fps'] = 250

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.95
SHOW_EVERY = 10
STEPS = 20000
EPSILON_RATE = 1
EPSILON_THRESHOLD = 0.1
epsilon_decay_value = 0.999

# x = cos(theta) ranges between -1 and 1
X_SPACE = np.linspace(
    env.observation_space.low[0], env.observation_space.high[0], 21)
# y = sin(theta) ranges between -1 and 1
Y_SPACE = np.linspace(
    env.observation_space.low[1], env.observation_space.high[1], 21)
# angular velocity ranges from -8 and 8
ANGULAR_VELOCITY_SPACE = np.linspace(
    env.observation_space.low[2], env.observation_space.high[2], 65)

# Since pendulum's action space is a Box, so we have to discrete it
DISCRETE_ACTION_SPACE_SIZE = 17
discrete_action_space_win_size = (
    env.action_space.high - env.action_space.low) / (DISCRETE_ACTION_SPACE_SIZE - 1)
action_space = {}
for i in range(DISCRETE_ACTION_SPACE_SIZE):
    action_space[i] = [env.action_space.low[0] +
                       (i * discrete_action_space_win_size[0])]


def run(episodes, is_training, render):
    print("RUN")
    print("is_training = ", is_training)
    env = gym.make('Pendulum-v1', render_mode='human' if render else None)

    if (is_training):
        # Initialize Q table
        Q = np.random.uniform(low=-2, high=0, size=(
            [len(X_SPACE), len(Y_SPACE), len(ANGULAR_VELOCITY_SPACE)] + [DISCRETE_ACTION_SPACE_SIZE]))
        # Q = np.zeros((len(X_SPACE), len(Y_SPACE), len(
        # ANGULAR_VELOCITY_SPACE), env.action_space))
        # print(Q)

    rewards_per_episode = np.zeros(episodes)
    epsilon = EPSILON_RATE
    epsilon_decay_rate = 2/episodes

    # Initialize environment state
    state = env.reset()[0]
    terminated = False

    # Loop for each episode
    for i in range(episodes):
        # Reset the environment and obtain initial state
        state = env.reset()[0]
        state = discretize_state(state)
        terminated = False
        rewards = 0
        step = 1
        # Take action
        action = epsilon_greedy_policy(is_training, epsilon, Q, state)
        print("Episode %d" % i)
        # Loop for each step of an episode
        while (not terminated and step < STEPS):
            torque = action_space[action]
            # Observe next state
            next_state, reward, terminated, _, _ = env.step(torque)
            # Return the next state
            next_state_x = next_state[0]
            next_state_y = next_state[1]
            next_state_velocity = next_state[2]

            # Take next action
            next_state = discretize_state(next_state)
            next_action = epsilon_greedy_policy(
                is_training, epsilon, Q, next_state)

            # Update Q table during training process
            if is_training:
                td_error = reward + DISCOUNT_RATE * \
                    np.max(Q[state][action]) - Q[state][action]
                Q[state][action] += LEARNING_RATE*td_error

            if next_state_x == 0 and next_state_y == 1:
                # Reward for finish
                print("We made it on episode %d" % i)
                Q[state][action] = 0

            # Assign current state and action
            state, action = next_state, next_action
            step += 1
            rewards += reward

        # epsilion decay
        if epsilon >= EPSILON_THRESHOLD:
            epsilon *= epsilon_decay_value

        if i % SHOW_EVERY == 0 and render == False:
            env.render()

        rewards_per_episode[i] = rewards

    env.close()


def discretize_state(observation):
    observation_x = np.digitize(observation[0], X_SPACE)
    observation_y = np.digitize(observation[1], Y_SPACE)
    observation_angular_velocity = np.digitize(
        observation[2], ANGULAR_VELOCITY_SPACE)
    return tuple([observation_x-1, observation_y-1, observation_angular_velocity-1])


def epsilon_greedy_policy(is_training, epsilon, q, state):
    # e-Greedy strategy
    # Explore random action with probability epsilon

    if is_training and random.uniform(0, 1) < epsilon:
        # Select one action's index in defined action_space
        action = np.random.randint(0, DISCRETE_ACTION_SPACE_SIZE)

    # Take best action with probability 1-epsilon
    else:
        action = np.argmax(q[state])
    return action


if __name__ == '__main__':
    # print(X_SPACE)
    # Q = np.random.uniform(low=-2, high=0, size=(
    # [len(X_SPACE), len(Y_SPACE), len(ANGULAR_VELOCITY_SPACE)] + [DISCRETE_ACTION_SPACE_SIZE]))
    # print(Q[0, 0, 0])
    # print(env.action_space.high)
    # print(env.action_space.low)
    # print(discrete_action_space_win_size)
    # print(env.action_space.sample())
    # state = env.reset()[0]

    # # print(env.action_space.sample())
    # terminated = False
    # while (terminated == False):
    #     action = epsilon_greedy_policy(True, EPSILON_RATE, Q, state)
    #     torque = action_space[action]
    #     next_state, reward, terminate, _, _ = env.step(torque)
    #     next_state_x = next_state[0]
    #     next_state_y = next_state[1]
    #     print("Next_state_x = ", next_state_x)
    #     print("next_state_y = ", next_state_y)
    #     next_state = discretize_state(next_state)
    #     if next_state_x == 0 and next_state_y == 0:
    #         print("yeayyyy")
    #         terminated = True
    # print(next_state)
    # print(Q[next_state])

    # if next_state > (1, 1, 0):
    # print("win")
    # else:
    # print(next_state)
    run(1000, is_training=True, render=False)
    # run(5000, is_training=True, render=False)
    # run(10, is_training=False, render=True)
