# Adapted from https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_q.py
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import sys

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
# env.metadata['render_fps'] = 250

LEARNING_RATE = 0.9
DISCOUNT_RATE = 0.9
EPSILON_RATE = 1
STEPS = 1000

ROWS = env.nrow
COLUMNS = env.ncol
GOAL_STATE = ROWS*COLUMNS - 1


def run(episodes, is_training, render):
    print("RUN")
    print("is_training = ", is_training)
    env = gym.make('FrozenLake-v1', render_mode='human' if render else None, desc=None, map_name="8x8",
                   is_slippery=True)
    # Initialize path
    path = 'Frozen_Lake_SARSA'
    q_path = path + '.pkl'

    if (is_training):
        # Q = np.random.uniform(low=0, high=1, size=(
        # [env.observation_space.n] + [env.action_space.n]))
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        # Q = np.zeros((COLUMNS, ROWS, env.action_space.n))
        # Q = np.random.uniform(low=0, high=1, size=(
        # [COLUMNS, ROWS] + [env.action_space.n]))
    else:
        print("Testing....")
        f = open(q_path, 'rb')
        Q = pickle.load(f)
        f.close()

    epsilon = EPSILON_RATE
    epsilon_decay_rate = 0.0001
    rewards_per_episode = np.zeros(episodes)
    learning_rate = LEARNING_RATE
    state = env.reset()[0]
    terminated = False
    for i in range(episodes):
        # Reset the environment / Initial state
        state = env.reset()[0]
        # state = discretize_state(state)

        terminated = False
        rewards = 0
        step = 1
        # Take action
        action = epsilon_greedy_policy(is_training, epsilon, Q, state)

        print("Episode %d" % i)
        while (not terminated and step <= STEPS):
            # Observe next state
            next_state, reward, terminated, _, _ = env.step(action)

            # next_state = discretize_state(next_state)

            next_action = epsilon_greedy_policy(
                is_training, epsilon, Q, next_state)

            if is_training:
                td_error = reward + DISCOUNT_RATE * \
                    Q[next_state][next_action] - Q[state][action]
                Q[state][action] += learning_rate*td_error
                # print(Q[state][action])

            if next_state == GOAL_STATE:
                print("We made it on episode %d" % i)
                Q[state][action] = 1

            state, action = next_state, next_action
            step += 1

        # print("Episode:", i, "Reward:", rewards, "Steps:", step)

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if (epsilon == 0):
            learning_rate = 0.0001
        if reward == 1:
            rewards_per_episode[i] = 1

    if is_training:
        f = open(q_path, 'wb')
        pickle.dump(Q, f)
        f.close()
        # Plot the rewards receive per 100 episodes graph
        mean_rewards = np.zeros(episodes)
        for t in range(episodes):
            mean_rewards[t] = np.mean(
                rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(mean_rewards)
        # reward_schedule_name = '_' +  reward_schedule
        # other = reward_schedule_name + '_' + environment_name
        # plt.suptitle('Mountain car SARSA applying VRS %s, Noisy = %s, Time = %f' % (self.reward_schedule, self.environment_name,
        #  end_time_running_all_episodes))
        # png_file = auxFunctions.create_filename(
        # 'mountain_car', 'SARSA', 'png', other=other)
        # plt.savefig(png_file)
        plt.show()
    return rewards_per_episode


def epsilon_greedy_policy(is_training, epsilon, Q, state):
    if is_training and random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

    return action


if __name__ == '__main__':
    run(20000, is_training=True, render=False)
    run(10, is_training=False, render=False)
    # Q = np.zeros((ROW, COLUMN, env.action_space.n))
    # state = env.reset()[0]
    # print(discretize_state(state))
