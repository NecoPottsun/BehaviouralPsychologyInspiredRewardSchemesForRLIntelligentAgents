import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from NoisyEnvironment import RandomNormalNoisyObservation, RandomUniformNoisyReward, RandomDropoutObservation, RandomNormalNoisyReward
import VisualizeResults

'''
@author: Trang Thanh Mai Duyen
'''

NUM_TRIALS = 15
NUM_EPISODE_CART_POLE = 1000
NUM_EPISODE_MOUNTAIN_CAR = 1000
LENGTH_SOLVED_CART_POLE = 200
LENGTH_SOLVED_MOUNTAIN_CAR = 300

# 0.1 is default
SCALING_FACTOR = 0.1


if __name__ == "__main__":
    cart_pole_env = gym.make("CartPole-v1", render_mode="human")
    cart_pole_noisy_env = RandomUniformNoisyReward(
        cart_pole_env, noise_rate=0.9, low=-0.2, high=0.2)
    cart_pole_noisy_env1 = RandomNormalNoisyObservation(cart_pole_env)

    mountain_car_env = gym.make("MountainCar-v0", render_mode="human")
    mountain_car_noisy_env = RandomUniformNoisyReward(
        mountain_car_env, noise_rate=0.9, low=-0.2, high=0.2)
    mountain_car_noisy_env1 = RandomNormalNoisyObservation(mountain_car_env)

    # mountain_car_noisy_env = RandomDropoutObservation(
    # mountain_car_env, noise_rate=0.09, p=0.5)
    # mountain_car_noisy_env = RandomNormalNoisyReward(
    # mountain_car_env, noise_rate=0.06, loc=0, scale=0.05)
    # mountain_car_noisy_env = RandomUniformNoisyReward(
    # mountain_car_env, noise_rate=0.5, low=-0.5, high=0.5)

    reward_schedule = 'variable_ratio'

    # Running Cart pole - Noisy Environment: None
    # Q-Learning
    # Continous Reward
    # VisualizeResults.CartPole_QLearning(
    # NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, None)
    # # Partial Reward Schedules
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                         None, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # None, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # None, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # None, SCALING_FACTOR, reward_schedule="variable-ratio")

    # SARSA
    # Continous Reward
    # VisualizeResults.CartPole_SARSA(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, None)
    # # Partial Reward Schedules
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                     None, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                     None, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                     None, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                     None, SCALING_FACTOR, reward_schedule="variable-ratio")

    # TD(3)
    # # Continous Reward
    # VisualizeResults.CartPole_TD3(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, None)
    # # Partial Reward Schedules
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                   None, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                   None, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                   None, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                   None, SCALING_FACTOR, reward_schedule="variable-ratio")

    # Running Cart pole - Noisy Environment: RandomUniformNoisyReward
    # Q-Learning
    # Continous Reward
    # VisualizeResults.CartPole_QLearning(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, cart_pole_noisy_env)
    # # Partial Reward Schedules
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="variable-ratio")

    # SARSA
    # Continous Reward
    # VisualizeResults.CartPole_SARSA(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, cart_pole_noisy_env)
    # # Partial Reward Schedules
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                     cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                     cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                     cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="variable-ratio")

    # TD(3)
    # Continous Reward
    # VisualizeResults.CartPole_TD3(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, cart_pole_noisy_env)
    # Partial Reward Schedules
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                   cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                   cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                   cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #                                   cart_pole_noisy_env, SCALING_FACTOR, reward_schedule="variable-ratio")

   # Running Cart pole - Noisy Environment: RandomNormalNoisyObservation
    # Q-Learning
    # Continous Reward
    # VisualizeResults.CartPole_QLearning(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, cart_pole_noisy_env)
    # # Partial Reward Schedules
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.CartPole_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="variable-ratio")

    # SARSA
    # Continous Reward
    # VisualizeResults.CartPole_SARSA(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, cart_pole_noisy_env)
    # # Partial Reward Schedules
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.CartPole_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    # cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="variable-ratio")

    # TD(3)
    # # Continous Reward
    # VisualizeResults.CartPole_TD3(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, None)
    # # Partial Reward Schedules
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #   cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #   cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #   cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.CartPole_TD3_VRS(NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE,
    #   cart_pole_noisy_env1, SCALING_FACTOR, reward_schedule="variable-ratio")

    # Running Mountain Car - Noisy Environment: None
    # Q-Learning
    # Continous Reward
    # VisualizeResults.MountainCar_QLearning(
    # NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR, None)
    # # Partial Reward Schedules
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    None, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    None, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    None, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    None, SCALING_FACTOR, reward_schedule="variable-ratio")

    # SARSA
    # Continous Reward
    # VisualizeResults.MountainCar_SARSA(
    # NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR, None)
    # # Partial Reward Schedules
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    None, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    None, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    None, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                        None, SCALING_FACTOR, reward_schedule="variable-ratio")

    # TD(3)
    # Continous Reward
    # VisualizeResults.MountainCar_TD3(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, None)
    # # Partial Reward Schedules
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                      None, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                      None, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                      None, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                      None, SCALING_FACTOR, reward_schedule="variable-ratio")

    # Running Mountain Car - Noisy Environment: RandomUniformNoisyReward
    # Q-Learning
    # Continous Reward
    # VisualizeResults.MountainCar_QLearning(
    # NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR, mountain_car_noisy_env)
    # # Partial Reward Schedules
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="variable-ratio")

    # SARSA
    # # Continous Reward
    # VisualizeResults.MountainCar_SARSA(
    #     NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR, mountain_car_noisy_env)
    # # Partial Reward Schedules
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                        mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                        mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                        mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                        mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="variable-ratio")

    # TD(3)
    # Continous Reward
    # VisualizeResults.MountainCar_TD3(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, mountain_car_noisy_env)
    # # Partial Reward Schedules
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                      mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                      mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                      mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                      mountain_car_noisy_env, SCALING_FACTOR, reward_schedule="variable-ratio")

 # Running Mountain Car - Noisy Environment: RandomNormalNoisyObservation
    # Q-Learning
    # Continous Reward
    # VisualizeResults.MountainCar_QLearning(
    # NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR, mountain_car_noisy_env1)
    # # Partial Reward Schedules
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.MountainCar_QLearning_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="variable-ratio")

    # SARSA
    # # Continous Reward
    # VisualizeResults.MountainCar_SARSA(
    #     NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR, mountain_car_noisy_env)
    # # Partial Reward Schedules
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.MountainCar_SARSA_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #    mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="variable-ratio")

    # TD(3)
    # Continous Reward
    # VisualizeResults.MountainCar_TD3(
    #     NUM_TRIALS, NUM_EPISODE_CART_POLE, LENGTH_SOLVED_CART_POLE, mountain_car_noisy_env)
    # # Partial Reward Schedules
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #  mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-ratio")
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #  mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="fixed-interval")
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #  mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="variable-interval")
    # VisualizeResults.MountainCar_TD3_VRS(NUM_TRIALS, NUM_EPISODE_MOUNTAIN_CAR, LENGTH_SOLVED_MOUNTAIN_CAR,
    #                                      mountain_car_noisy_env1, SCALING_FACTOR, reward_schedule="variable-ratio")
