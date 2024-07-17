import numpy as np
import random
import math
import time
from datetime import datetime

'''
@author: Trang Thanh Mai Duyen
'''

FIXED_OPTIMAL_STEPS = 300
FIXED_NUMBER_STEPS = 5
INTERVAL_DURATION = 0.002
PROBABILITY = 0.5
a = 1
AVERAGE_RATIO = 1


def select_reward_schedule(schedule, time_step, last_reinforcement_time):
    if schedule == 'fixed_ratio':
        return fixed_Ratio_Function(time_step)
    elif schedule == 'fixed_interval':
        return fixed_Ratio_Function(last_reinforcement_time)
    elif schedule == 'variable_interval':
        return variable_Interval_Function(last_reinforcement_time)
    else:
        return variable_Ratio_Function(time_step)


def fixed_Ratio_Function(time_step, optimal_steps=FIXED_OPTIMAL_STEPS, fixed_number_steps=FIXED_NUMBER_STEPS):
    if time_step >= optimal_steps or (time_step % fixed_number_steps) == 0:
        return 1
    else:
        return 0


def fixed_Interval_Function(last_reinforcement_time, interval_duration=INTERVAL_DURATION):
    current_time = time.time()
    duration = current_time - last_reinforcement_time
    duration = round(duration, 8)
    # print("duration = ", duration)
    if (duration % interval_duration) == 0:
        return 1
    else:
        return 0


def variable_Interval_Function(last_reinforcement_time, a=1, p=PROBABILITY):
    current_time = time.time()
    duration = current_time - last_reinforcement_time
    generated_interval = np.random.binomial(a, p) + 1
    # print(duration % generated_interval)

    # predictable - predict based on generated_interval
    if (duration % generated_interval) == 0:
        # print(1)
        return 1
    else:
        return 0


def variable_Ratio_Function(time_step, p=PROBABILITY, average_ratio=AVERAGE_RATIO):
    correct_responses = np.random.binomial(time_step, p)

    # unpredictable - correct-responses randomly generated based on given probability
    if correct_responses >= average_ratio:
        return 1
    else:
        return 0


def bias_correction(scaling_factor, reward_avg, reward_prev):
    return scaling_factor*(reward_avg - reward_prev)

# Generate date/time string


def get_date_time_str():
    now = datetime.now()  # current date and time
    return now.strftime("%d%m%Y_%H%M")


def create_filename(environment_name, algorithm_name, format, other=''):
    # For example: 'mountain_car_SARSA_02122023_0121.pkl' or 'mountain_car_SARSA_02122023_0121.png'
    filename = environment_name + '_' + algorithm_name + other + \
        '_' + get_date_time_str() + '.' + format
    return filename


if __name__ == '__main__':
    # variable_Interval_Function(5, a=1, p=0.5)
    count = 0
    for i in range(1, 100):
        # print(variable_Ratio_Function(, p=0.7, variable_ratio=5))
        correct_response = np.random.binomial(i, (1/i))
        print("Correct response = ", correct_response)
        if correct_response == 1:
            print("Correct response == 1, hooray!")
            count += 1

    print("Number of correct responses = ", count)
