import math
import numpy as np


def calculate_distance(user_x, user_y, bs_x, bs_y):
    """
    Calculate Euclidean distance between
    the mobile user and a base station.
    """

    dx = user_x - bs_x
    dy = user_y - bs_y

    distance = math.sqrt(dx**2 + dy**2)

    return distance


def calculate_path_loss(
    distance,
    reference_distance=1.0,
    reference_path_loss=40.0,
    path_loss_exponent=3.0
):
    """
    Calculate path loss using the log-distance model.
    """

    distance = max(distance, reference_distance)

    path_loss = (
        reference_path_loss
        + 10 * path_loss_exponent
        * math.log10(distance / reference_distance)
    )

    return path_loss


def calculate_shadowing(std_dev=6.0):
    """
    Generate log-normal shadowing in dB.
    """

    return np.random.normal(0, std_dev)


def calculate_rayleigh_fading():
    """
    Generate Rayleigh small-scale fading
    in dB.
    """

    power_gain = np.random.exponential(scale=1.0)

    fading_db = 10 * math.log10(power_gain)

    return fading_db


def calculate_received_power(
    transmit_power,
    path_loss,
    shadowing=0.0,
    fading=0.0
):
    """
    Calculate received signal power in dBm.
    """

    received_power = (
        transmit_power
        - path_loss
        + shadowing
        + fading
    )

    return received_power

# Example usage

if __name__ == "__main__":

    distance = 100

    path_loss = calculate_path_loss(distance)

    print("Distance:", distance, "m")
    print("Path Loss:", path_loss, "dB")