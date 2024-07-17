# Adapted from: https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
import gymnasium as gym
import numpy as np
import random
import math
import time
# from time import sleep
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pickle
from time import sleep
import NoisyEnvironment
import auxFunctions

'''
- Add noisy environment option
- Add plot graph parts which are adapted from Mountain Car
@author: Trang Thanh Mai Duyen
'''

# Initialize the "Cart-Pole" environment
env = gym.make("CartPole-v1", render_mode="human")
# env.metadata['render_fps'] = 240
env._max_episode_steps = 200

# Initializing the random number generator
np.random.seed(int(time.time()))

# Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n  # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = (-0.5, 0.5)
STATE_BOUNDS[3] = (-math.radians(50), math.radians(50))
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

# Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

# Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
TEST_RAND_PROB = 0.2

# Defining the simulation related constants
NUM_TRAIN_EPISODES = 1000  # 1000
NUM_TEST_EPISODES = 1
MAX_TRAIN_T = 250
MAX_TEST_T = 250
STREAK_TO_END = 120  # 120
SOLVED_T = 199
VERBOSE = False


class CartPoleQLEARNINGNoisyEnvironment:
    def __init__(self, noisy_environment=None):
        self.noisy_environment = noisy_environment

    def train(self, episodes, render):
        print('Training ...')

        '''Adding noisy environment option by Trang Thanh Mai Duyen'''
        env = gym.make('CartPole-v1', render_mode='human' if render else None)
        self.environment_name = 'None'
        if (self.noisy_environment != None):
            self.env = self.noisy_environment
            # Get noisy environment name
            self.environment_name = self.noisy_environment.__class__.__name__
        # Instantiating the learning related parameters
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        discount_factor = 0.99  # since the world is unchanging
        q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
        num_train_streaks = 0
        episode_durations = []
        rewards_per_episode = np.zeros(episodes)
        start_time_running_all_episodes = time.time()
        for episode in range(episodes):
            # Reset the environment
            state, info = env.reset()
            rewards = 0
            # the initial state
            state = self.state_to_bucket(state)

            for t in range(MAX_TRAIN_T):
                env.render()

                # Select an action
                action = self.select_action(state, explore_rate, q_table)

                # Execute the action
                next_state, reward, terminated, truncated, info = env.step(
                    action)
                # Observe the result
                next_state = self.state_to_bucket(next_state)

                # Update the Q based on the result
                best_q = np.amax(q_table[next_state])
                q_table[state + (action,)] += learning_rate*(reward +
                                                             discount_factor*(best_q) - q_table[state + (action,)])
                # print(q_table[state + (action, )])
                # Setting up for the next iteration
                state = next_state
                rewards += reward
                # Print data
                if (VERBOSE):
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Action: %d" % action)
                    print("State: %s" % str(state))
                    print("Reward: %f" % reward)
                    print("Best Q: %f" % best_q)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_train_streaks)
                    print("")

                if terminated or truncated:
                    print("Episode %d finished after %f time steps" %
                          (episode, t))
                    if (t >= SOLVED_T):
                        num_train_streaks += 1
                    else:
                        num_train_streaks = 0
                    break

                # sleep(0.25)

            # It's considered done when it's solved over 120 times consecutively
            # if num_train_streaks > STREAK_TO_END:
            #     break

            # Update parameters
            explore_rate = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate(episode)
            episode_durations.append(t)
            rewards_per_episode[episode] = rewards

        end_time_running_all_episodes = time.time() - start_time_running_all_episodes

        # Plot the results: rewards versus episode
        path = 'cart_pole_Q-LEARNING_' + self.environment_name
        q_path = path + '.pkl'
        f = open(q_path, 'wb')
        pickle.dump(q_table, f)
        f.close()
        # Plot the rewards receive per 100 episodes graph
        mean_rewards = np.zeros(episodes)
        for t in range(episodes):
            mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(mean_rewards)
        plt.suptitle('Cart pole Q-learning, Noisy = %s, Time = %f' % (self.environment_name,
                     end_time_running_all_episodes))
        png_file = auxFunctions.create_filename(
            'cart_pole', 'Q-learning', 'png', other=self.environment_name)
        plt.savefig(png_file)
        # plt.clf()
        return episode_durations, rewards_per_episode

    def test(self):
        print('Testing ...')
        path = 'cart_pole_Q-learning_' + self.environment_name
        q_path = path + '.pkl'
        f = open(q_path, 'rb')
        Q = pickle.load(f)
        f.close()

        num_test_streaks = 0

        for episode in range(NUM_TEST_EPISODES):

            # Reset the environment
            obv, info = env.reset()

            # the initial state
            state_0 = self.state_to_bucket(obv)

            # basic initializations
            tt = 0
            terminated = False

            while ((abs(obv[0]) < 2.4) & (abs(obv[2]) < 45) and not terminated):
                while not (terminated):
                    tt += 1
                    env.render()

                    # Select an action
                    action = self.select_action(state_0, 0, Q)
                    # action = select_action(state_0, TEST_RAND_PROB)
                    # action = select_action(state_0, 0.01)

                    # Execute the action
                    obv, reward, terminated, truncated, info = env.step(action)

                    # Observe the result
                    state_0 = self.state_to_bucket(obv)

                    print("Test episode %d; time step %f." % (episode, tt))

    def select_action(self, state, explore_rate, Q):
        # Select a random action
        if random.random() < explore_rate:
            action = env.action_space.sample()
        # Select the action with the highest q
        else:
            action = np.argmax(Q[state])
        return action

    def get_explore_rate(self, t):
        return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

    def get_learning_rate(self, t):
        return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

    def state_to_bucket(self, state):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= STATE_BOUNDS[i][0]:
                bucket_index = 0
            elif state[i] >= STATE_BOUNDS[i][1]:
                bucket_index = NUM_BUCKETS[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
                offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0] / \
                    bound_width  # "index per boudns" * min_bound
                # how much index per width
                scaling = (NUM_BUCKETS[i]-1)/bound_width
                bucket_index = int(round(scaling*state[i] - offset))
                # For easier visualization of the above, you might want to use
                # pen and paper and apply some basic algebraic manipulations.
                # If you do so, you will obtaint (B-1)*[(S-MIN)]/W], where
                # B = NUM_BUCKETS, S = state, MIN = STATE_BOUNDS[i][0], and
                # W = bound_width. This simplification is very easy to
                # to visualize, i.e. num_buckets x percentage in width.
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)


# if __name__ == "__main__":
#     print('Training ...')
#     train(render=False)
#     print('Testing ...')
#     test()
