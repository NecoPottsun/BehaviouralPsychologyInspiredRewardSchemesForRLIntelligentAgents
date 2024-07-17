# Adapted some of the plot functions from the Reinforcement Learning (DQN) Tutorial by
# [Adam Paszke](https://github.com/apaszke) and [Mark Towers](https://github.com/pseudo-rnd-thoughts)

from MountainCar.MountainCarQLEARNINGNoisyEnvironment import MountainCarQLEARNINGNoisyEnvironment
from MountainCar.MountainCarQLEARNINGVRSNoisyEnvironment import MountainCarQLEARNINGVRSNoisyEnvironment
from MountainCar.MountainCarSARSANoisyEnvironment import MountainCarSARSANoisyEnvironment
from MountainCar.MountainCarSARSAVRSNoisyEnvironment import MountainCarSARSAVRSNoisyEnvironment
from MountainCar.MountainCarTD3NoisyEnvironment import MountainCarTD3NoisyEnvironment
from MountainCar.MountainCarTD3VRSNoisyEnvironment import MountainCarTD3VRSNoisyEnvironment
from CartPole.CartPoleQLEARNINGNoisyEnvironment import CartPoleQLEARNINGNoisyEnvironment
from CartPole.CartPoleQLEARNINGVRSNoisyEnvironment import CartPoleQLEARNINGVRSNoisyEnvironment
from CartPole.CartPoleSARSANoisyEnvironment import CartPoleSARSANoisyEnvironment
from CartPole.CartPoleSARSAVRSNoisyEnvironment import CartPoleSARSAVRSNoisyEnvironment
from CartPole.CartPoleTD3NoisyEnvironment import CartPoleTD3NoisyEnvironment
from CartPole.CartPoleTD3VRSNoisyEnvironment import CartPoleTD3VRSNoisyEnvironment

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import numpy as np
from scipy import stats
import time
from NoisyEnvironment import RandomNormalNoisyObservation

TEST_TIMES = 1

# Get index of first value under a threshold
# For mountain car, the agent will learn overtime, so at first the steps will be very large, after every step learning it will try to optimize the step


def get_index_under(list, threshold_value):
    array = np.array(list)
    # print(array)
    mask = array < threshold_value
    if np.any(mask):
        return np.argmax(mask)
    else:
        return -1

# For cart pole, the agent will learn overtime, trying to balance the pole a long as possible, so the steps will increase overtime


def get_index_exceed(list, threshold_value):
    array = np.array(list)
    # print(array)
    mask = array > threshold_value
    if np.any(mask):
        return np.argmax(mask)
    else:
        return -1


def Calculate_Mean_Std(all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title):
    # Calculating the mean & std of all the durations and received rewards of the agent on 5 times training
    all_mean_durations = all_durations.mean(axis=0)
    all_std_durations = all_durations.std(axis=0)
    all_mean_rewards_received = all_rewards_received.mean(axis=0)
    all_std_rewards_received = all_rewards_received.std(axis=0)

    # rewards of cart pole will be > 0, rewards of mountain car will be <= 0
    if (all_mean_rewards_received[0] > 0):
        solved_at = get_index_exceed(all_durations, length_solved)
    else:
        solved_at = get_index_under(all_durations, length_solved)
    mean_cumulative_durations = cumulative_durations.mean(axis=0)
    std_cumulative_durations = cumulative_durations.std(axis=0)
    mean_cumulative_tot_rewards = cumulative_tot_rewards.mean(axis=0)
    std_cumulative_tot_rewards = cumulative_tot_rewards.std(axis=0)

    #   Print out results
    print(f"*********{a_title}*********")
    print(f"all_mean_durations = {all_durations.mean()}")
    print(f"all_std_durations = {all_durations.std()}")
    print(f"all_mean_rewards_received = {all_rewards_received.mean()}")
    print(f"all_std_rewards_received = {all_rewards_received.std()}")
    print(f"solved_at = {solved_at}")
    # print("Cumulative_durations: ", cumulative_durations)
    print(f"mean_cumulative_durations = {mean_cumulative_durations}")
    print(f"std_cumulative_durations = {std_cumulative_durations}")
    # print("Cumulative total rewards: ", cumulative_tot_rewards)
    print(f"mean_cumulative_tot_rewards = {mean_cumulative_tot_rewards}")
    print(f"std_cumulative_tot_rewards = {std_cumulative_tot_rewards}")

    # Plot
    performs = {}
    performs['solved_at'] = solved_at
    performs['mean_cumulative_durations'] = mean_cumulative_durations
    performs['std_cumulative_durations'] = std_cumulative_durations
    performs['mean_cumulative_tot_rewards_received'] = mean_cumulative_tot_rewards
    performs['std_cumulative_tot_rewards_received'] = std_cumulative_tot_rewards

    return all_mean_durations, all_std_durations, all_mean_rewards_received, all_std_rewards_received, performs


def plot_duration(mean_durations, std_durations, mean_tot_rewards_received, std_tot_rewards_received, performs, a_title):
    num_trials = mean_durations.shape[0]
    # print("Length of mean_durations= ", len(mean_durations))
    num_episode = len(mean_durations)
    plt.title(a_title)
    plt.xlabel('Episode')
    plt.ylabel('Duration/Reward')
    xs = np.arange(0, num_trials)
    # Plotting mean curves with error bars
    # plt.errorbar(range(1, num_episode + 1), mean_durations,
    #              yerr=all_std_durations, color='blue', label='DR')
    # plt.errorbar(range(1, num_episode + 1), mean_tot_rewards_received,
    #              yerr=np.abs(all_std_rewards_received), color='green', label='TR')
    plt.plot(xs, mean_durations, color='blue', label='DR')
    plt.plot(xs, mean_tot_rewards_received, color='red', label='TR')

    # # Plotting filled areas
    plt.fill_between(range(1, num_episode+1), mean_durations - std_durations,
                     mean_durations + std_durations, color='blue', alpha=0.2)
    plt.fill_between(range(1, num_episode+1), mean_tot_rewards_received - std_tot_rewards_received,
                     mean_tot_rewards_received + std_tot_rewards_received, color='green', alpha=0.2)
    plt.legend()

    # Plot extra performance metrics
    res_string = 'Solved at: {0}\n'.format(performs['solved_at'])
    # total_time running n experiments
    res_string += 'Total time: {0}\n'.format(performs['total_time'])
    # durations
    res_string += 'M. cumul. DR: {0}\n'.format(
        performs['mean_cumulative_durations'])
    # total reward
    res_string += 'M.cumul. TR: {0}\n'.format(
        performs['mean_cumulative_tot_rewards_received'])
    res_string += 'M. std. DR: {0}\n'.format(
        np.round(performs['std_cumulative_durations'], 2))
    res_string += 'M. std. TR: {0}\n'.format(
        np.round(performs['std_cumulative_tot_rewards_received'], 2))
    x_pos = int(0.6*num_trials)
    plt.text(x_pos, 30, res_string, color='black', fontsize=12)


# Mountain Car Training & Plotting
def MountainCar_QLearning(num_trials, num_episode, length_solved, noisy_environment=None):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))
    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        MountainCar_QLearning = MountainCarQLEARNINGNoisyEnvironment(
            noisy_environment=noisy_environment)
        # Training
        episode_durations, rewards_per_episode = MountainCar_QLearning.run(
            episodes=num_episode, is_training=True, render=False)

        # Testing
        MountainCar_QLearning.run(TEST_TIMES, is_training=False, render=True)
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    # Calculating the mean & std of all the durations and received rewards of the agent on 5 times training
    a_title = "Mountain Car Q-Learning"
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time

    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def MountainCar_QLearning_VRS(num_trials, num_episode, length_solved, noisy_environment=None, scaling_factor=0.1, reward_schedule='variable_ratio'):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))
    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        MountainCar_QLearning_VRS = MountainCarQLEARNINGVRSNoisyEnvironment(noisy_environment=noisy_environment, scaling_factor=scaling_factor,
                                                                            reward_schedule=reward_schedule)
        # Training
        episode_durations, rewards_per_episode = MountainCar_QLearning_VRS.run(
            num_episode, is_training=True, render=False)
        # Testing
        MountainCar_QLearning_VRS.run(
            TEST_TIMES, is_training=False, render=True)
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    a_title = "Mountain Car Q-Learning applying VRS " + reward_schedule
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time

    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def MountainCar_SARSA(num_trials, num_episode, length_solved, noisy_environment=None):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))
    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        MountainCar_SARSA = MountainCarSARSANoisyEnvironment(
            noisy_environment=noisy_environment)
        # Training
        episode_durations, rewards_per_episode = MountainCar_SARSA.run(
            episodes=num_episode, is_training=True, render=False)

        # Testing
        MountainCar_SARSA.run(TEST_TIMES, is_training=False, render=True)
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    # Calculating the mean & std of all the durations and received rewards of the agent on 5 times training
    a_title = "Mountain Car SARSA"
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time

    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def MountainCar_SARSA_VRS(num_trials, num_episode, length_solved, noisy_environment=None, scaling_factor=0.1, reward_schedule='variable_ratio'):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))
    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        MountainCar_SARSA_VRS = MountainCarSARSAVRSNoisyEnvironment(noisy_environment=noisy_environment, scaling_factor=scaling_factor,
                                                                    reward_schedule=reward_schedule)
        # Training
        episode_durations, rewards_per_episode = MountainCar_SARSA_VRS.run(
            num_episode, is_training=True, render=False)
        # Testing
        MountainCar_SARSA_VRS.run(
            TEST_TIMES, is_training=False, render=True)
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    a_title = "Mountain Car SARSA applying VRS " + reward_schedule
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time

    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def MountainCar_TD3(num_trials, num_episode, length_solved, noisy_environment=None):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))
    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        MountainCar_TD3 = MountainCarTD3NoisyEnvironment(
            noisy_environment=noisy_environment)
        # Training
        episode_durations, rewards_per_episode = MountainCar_TD3.run(
            num_episode, is_training=True, render=False)
        # Testing
        MountainCar_TD3.run(TEST_TIMES, is_training=False, render=True)
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    a_title = "Mountain Car TD(3)"
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time
    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def MountainCar_TD3_VRS(num_trials, num_episode, length_solved, noisy_environment=None, scaling_factor=0.1, reward_schedule='variable_ratio'):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))
    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        MountainCar_TD3_VRS = MountainCarTD3VRSNoisyEnvironment(
            noisy_environment=noisy_environment, scaling_factor=scaling_factor, reward_schedule=reward_schedule)
        # Training
        episode_durations, rewards_per_episode = MountainCar_TD3_VRS.run(
            num_episode, is_training=True, render=False)
        # Testing
        MountainCar_TD3_VRS.run(TEST_TIMES, is_training=False, render=True)
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    a_title = "Mountain Car TD(3) applying VRS " + reward_schedule
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time
    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


# Cart Pole Training & Plotting
def CartPole_QLearning(num_trials, num_episode, length_solved, noisy_environment=None):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))

    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        CartPole_QLearning = CartPoleQLEARNINGNoisyEnvironment(
            noisy_environment=noisy_environment)
        episode_durations, rewards_per_episode = CartPole_QLearning.train(
            num_episode, render=False)
        CartPole_QLearning.test()
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    # Calculating the mean & std of all the durations and received rewards of the agent on 5 times training
    a_title = "Cart Pole Q-Learning"
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time
    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def CartPole_QLearning_VRS(num_trials, num_episode, length_solved, noisy_environment=None, scaling_factor=0.1, reward_schedule='variable_ratio'):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))

    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        CartPole_QLearning_VRS = CartPoleQLEARNINGVRSNoisyEnvironment(noisy_environment=noisy_environment, scaling_factor=scaling_factor,
                                                                      reward_schedule=reward_schedule)
        episode_durations, rewards_per_episode = CartPole_QLearning_VRS.train(
            num_episode, render=False)
        CartPole_QLearning_VRS.test()
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    # Calculating the mean & std of all the durations and received rewards of the agent on 5 times training
    a_title = "Cart Pole Q-Learning applying VRS"
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time
    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def CartPole_SARSA(num_trials, num_episode, length_solved, noisy_environment=None):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))

    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        CartPole_SARSA = CartPoleSARSANoisyEnvironment(
            noisy_environment=noisy_environment)
        episode_durations, rewards_per_episode = CartPole_SARSA.train(
            num_episode, render=False)
        CartPole_SARSA.test()
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    # Calculating the mean & std of all the durations and received rewards of the agent on 5 times training
    a_title = "Cart Pole SARSA"
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time
    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def CartPole_SARSA_VRS(num_trials, num_episode, length_solved, noisy_environment=None, scaling_factor=0.1, reward_schedule='variable_ratio'):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))
    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        CartPole_SARSA_VRS = CartPoleSARSAVRSNoisyEnvironment(noisy_environment=noisy_environment, scaling_factor=scaling_factor,
                                                              reward_schedule=reward_schedule)
        episode_durations, rewards_per_episode = CartPole_SARSA_VRS.train(
            num_episode, render=False)
        CartPole_SARSA_VRS.test()
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    # Calculating the mean & std of all the durations and received rewards of the agent on 5 times training
    a_title = "Cart Pole SARSA applying VRS " + reward_schedule
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time
    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def CartPole_TD3(num_trials, num_episode, length_solved, noisy_environment=None):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))
    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        CartPole_TD3 = CartPoleTD3NoisyEnvironment(
            noisy_environment=noisy_environment)
        episode_durations, rewards_per_episode = CartPole_TD3.train(
            num_episode, render=False)
        CartPole_TD3.test()
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    a_title = "Cart Pole TD(3)"
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time
    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


def CartPole_TD3_VRS(num_trials, num_episode, length_solved, noisy_environment=None, scaling_factor=0.1, reward_schedule='variable_ratio'):
    all_durations = np.zeros((num_trials, num_episode))
    all_rewards_received = np.zeros((num_trials, num_episode))
    cumulative_durations = np.zeros((num_trials))
    cumulative_tot_rewards = np.zeros((num_trials))
    start_time = time.time()
    for trial in range(num_trials):
        print(f"_________Trial {trial + 1}_________")
        CartPole_TD3_VRS = CartPoleTD3VRSNoisyEnvironment(noisy_environment=noisy_environment, scaling_factor=scaling_factor,
                                                          reward_schedule=reward_schedule)
        episode_durations, rewards_per_episode = CartPole_TD3_VRS.train(
            num_episode, render=False)
        CartPole_TD3_VRS.test()
        all_durations[trial] = episode_durations
        all_rewards_received[trial] = rewards_per_episode
        cumulative_durations[trial] = np.sum(episode_durations)
        cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)
    total_time = time.time() - start_time
    a_title = "Cart Pole TD(3) applying VRS " + reward_schedule
    mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
        all_durations, all_rewards_received, cumulative_durations, cumulative_tot_rewards, length_solved, a_title)
    # add total_time to performs
    performs['total_time'] = total_time
    plot_duration(mean_durations, std_durations,
                  mean_rewards_received, std_rewards_received, performs, a_title)

    plt.show()


if __name__ == '__main__':

    num_trials = 5
    num_episode = 300

    # all_durations = np.zeros((num_trials, num_episode))
    # # all_rewards_received = np.zeros((num_trials, num_episode))
    # # cumulative_durations = np.zeros((num_trials))
    # # cumulative_tot_rewards = np.zeros((num_trials))

    # print(all_durations.shape[1])
    # cart_pole_env = gym.make("CartPole-v1", render_mode="human")
    # cart_pole_noisy_env = RandomNormalNoisyObservation(cart_pole_env)

    # CartPole_SARSA_VRS(
    #     2, 1000, 200, cart_pole_noisy_env)
    # for trial in range(num_trials):
    #     print(f"_________Trial {trial + 1}_________")
    #     MountainCarTD3VRS = MountainCarTD3VRSNoisyEnvironment()
    #     # Training
    #     episode_durations, rewards_per_episode = MountainCarTD3VRS.run(
    #         num_episode, is_training=True, render=False)
    #     # Testing
    #     MountainCarTD3VRS.run(5, is_training=False, render=True)
    #     all_durations[trial] = episode_durations
    #     all_rewards_received[trial] = rewards_per_episode
    #     cumulative_durations[trial] = np.sum(episode_durations)
    #     cumulative_tot_rewards[trial] = np.sum(rewards_per_episode)

    # a_title = "Mountain Car TD(3) applying VRS "
    # mean_durations, std_durations, mean_rewards_received, std_rewards_received, performs = Calculate_Mean_Std(
    #     all_durations, all_rewards_received, LENG_SOLVED, a_title)

    # plot_duration(mean_durations, std_durations,
    #               mean_rewards_received, std_rewards_received, performs, a_title)

    # plt.show()
