# Adapted from “https://github.com/johnnycode8/gym_solutions”
import gymnasium as gym
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import pickle
import sys
import auxFunctions

'''
- Add noisy environment option
- Implement SARSA algorithm
@author: Trang Thanh Mai Duyen
'''

env = gym.make('MountainCar-v0')
env.metadata['render_fps'] = 250

LEARNING_RATE = 0.9
DISCOUNT_RATE = 0.9
EPSILON_RATE = 1
STEPS = 1000

# Divide position and velocity into 20 segments
# bet -1.2 and 0.6 for position
POS_SPACE = np.linspace(
    env.observation_space.low[0], env.observation_space.high[0], 20)
# bet -0.7 and 0.08 for velocity
VEL_SPACE = np.linspace(
    env.observation_space.low[1], env.observation_space.high[1], 20)


class MountainCarSARSANoisyEnvironment:
    def __init__(self, noisy_environment=None):
        self.noisy_environment = noisy_environment
        # self.run(self.training_episode, is_training=True, render=False)
        # self.run(self.testing_episode, is_training=False, render=True)

    def run(self, episodes, is_training, render):
        print("RUN")
        print("is_training = ", is_training)
        env = gym.make('MountainCar-v0',
                       render_mode='human' if render else None)
        self.environment_name = 'None'
        if (self.noisy_environment != None):
            self.env = self.noisy_environment
            # Get noisy environment name
            self.environment_name = self.noisy_environment.__class__.__name__

        # Initialize path
        path = 'mountain_car_SARSA_' + self.environment_name
        q_path = path + '.pkl'

        # Initialize V(s)
        if (is_training):
            print('Training ...')
            # Q = np.zeros((len(POS_SPACE), len(VEL_SPACE), env.action_space.n))
            Q = np.random.uniform(
                low=-2, high=0, size=([len(POS_SPACE), len(VEL_SPACE)] + [env.action_space.n]))
        else:
            print('Testing ...')
            f = open(q_path, 'rb')
            Q = pickle.load(f)
            f.close()

        epsilon = EPSILON_RATE  # 100% of random actions
        epsilon_decay_rate = 2/episodes  # epsilon decay rate

        episode_durations = []
        rewards_per_episode = np.zeros(episodes)

        state = env.reset()[0]
        terminated = False

        start_time_running_all_episodes = time.time()

        # Loop for each episode:
        for i in range(episodes):
            # Initialize S / Reset the environment
            state = env.reset()[0]
            state = self.discretize_state(state)
            terminated = False  # True when reached goal
            rewards = 0
            step = 1
            # Take action (1)
            action = self.epsilon_greedy_policy(
                is_training, epsilon, Q, state)

            print("Episode %d" % i)
            # Loop for each step of episode:
            while (not terminated and step <= STEPS):

                # Observe next_state:
                next_state, reward, terminated, _, _ = env.step(action)
                # return the next position to check if the next position != goal position
                next_state_position = next_state[0]
                next_state = self.discretize_state(next_state)
                next_action = self.epsilon_greedy_policy(
                    is_training, epsilon, Q, next_state)

                if is_training:
                    td_error = reward + DISCOUNT_RATE * \
                        Q[next_state][next_action] - Q[state][action]
                    Q[state][action] += LEARNING_RATE*td_error
                    # print(Q[state][action])

                if next_state_position >= env.goal_position:
                    # reward for finish things
                    print("We made it on episode %d" % i)
                    Q[state][action] = 0
                    break

                state, action = next_state, next_action
                step += 1
                rewards += reward

            # epsilon = epsilon - 2/episodes if epsilon > 0.01 else 0.01
            epsilon = max(epsilon - epsilon_decay_rate, 0)
            rewards_per_episode[i] = rewards
            episode_durations.append(step-1)
        end_time_running_all_episodes = time.time() - start_time_running_all_episodes
        env.close()

        # Save V table to file
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
            plt.suptitle('Mountain car SARSA, Noisy = %s,Time = %f' % (self.environment_name,
                         end_time_running_all_episodes))
            png_file = auxFunctions.create_filename(
                'mountain_car', 'SARSA', 'png', other=self.environment_name)
            plt.savefig(png_file)
        return episode_durations, rewards_per_episode

    def discretize_state(self, observation):
        observation_p = np.digitize(observation[0], POS_SPACE)
        observation_v = np.digitize(observation[1], VEL_SPACE)
        return tuple([observation_p, observation_v])

    def epsilon_greedy_policy(self, is_training, epsilon, q, state):
        # e-Greedy strategy
        # Explore random action with probability epsilon

        if is_training and random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()

        # Take best action with probability 1-epsilon
        else:
            action = np.argmax(q[state])
        return action


# if __name__ == '__main__':
#     run(100, is_training=True, render=False)
#     run(10, is_training=False, render=True)
    # Q = np.random.uniform(
    # low=-2, high=0, size=([len(POS_SPACE), len(VEL_SPACE)] + [env.action_space.n]))
    # state = env.reset()[0]
    # state_p = np.digitize(state[0], POS_SPACE)
    # # state_v = np.digitize(state[1], VEL_SPACE)
    # state = discretize_state(state)
    # action = epsilon_greedy_policy(False, 1, Q, state)

    # print(np.max(Q[state][:]))
    # print(np.argmax(Q[state]))
    # print(epsilon_greedy_policy(False, 1, Q, state))
