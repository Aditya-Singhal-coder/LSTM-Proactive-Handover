
import random

import pandas as pd

from base_stations import create_base_stations
from mobility import MobileUser

from channel import (
    calculate_distance,
    calculate_path_loss,
    calculate_shadowing,
    calculate_rayleigh_fading,
    calculate_received_power
)

from handover import reactive_handover

from metrics import (
    count_handovers,
    count_ping_pong,
    calculate_average_serving_rssi,
    count_outage_samples
)


def run_simulation(
    simulation_time=100,
    transmit_power=30,
    shadowing_std=6
):

    # Create base stations
    base_stations = create_base_stations()

    # Random starting position
    start_x = random.uniform(-50, 50)
    start_y = random.uniform(20, 80)

    # Random speed
    speed = random.uniform(5, 15)

    # Random direction
    direction = random.uniform(0, 360)

    # Create mobile user
    user = MobileUser(
        x=start_x,
        y=start_y,
        speed=speed,
        direction=direction
    )

    user_data = []

    # ----------------------------------------------
    # Run simulation
    # ----------------------------------------------

    for t in range(simulation_time):

        row = {
            "time": t,
            "x": user.x,
            "y": user.y,
            "speed": user.speed,
            "direction": user.direction
        }

        # Calculate RSSI from every BS
        for bs in base_stations:

            distance = calculate_distance(
                user.x,
                user.y,
                bs.x,
                bs.y
            )

            path_loss = calculate_path_loss(
                distance
            )

            shadowing = calculate_shadowing(
                shadowing_std
            )

            fading = calculate_rayleigh_fading()

            received_power = calculate_received_power(
                transmit_power,
                path_loss,
                shadowing,
                fading
            )

            row[f"{bs.bs_id}_distance"] = distance
            row[f"{bs.bs_id}_path_loss"] = path_loss
            row[f"{bs.bs_id}_shadowing"] = shadowing
            row[f"{bs.bs_id}_fading"] = fading
            row[f"{bs.bs_id}_power"] = received_power

        user_data.append(row)

        # Move user
        user.move(time_step=1)

    # ----------------------------------------------
    # Create DataFrame
    # ----------------------------------------------

    df = pd.DataFrame(user_data)

    # ----------------------------------------------
    # Reactive handover
    # ----------------------------------------------

    bs_ids = [
        bs.bs_id
        for bs in base_stations
    ]

    serving_history, handovers = reactive_handover(
        df,
        bs_ids,
        hysteresis=3.0,
        ttt=3
    )

    df["serving_bs"] = serving_history

    # ----------------------------------------------
    # Calculate metrics
    # ----------------------------------------------

    total_handovers = count_handovers(
        handovers
    )

    ping_pong = count_ping_pong(
        handovers,
        window=10
    )

    average_rssi = calculate_average_serving_rssi(
        df
    )

    outage_samples = count_outage_samples(
        df,
        threshold=-100
    )

    # ----------------------------------------------
    # Return simulation results
    # ----------------------------------------------

    metrics = {
        "total_handovers": total_handovers,
        "ping_pong": ping_pong,
        "average_rssi": average_rssi,
        "outage_samples": outage_samples
    }

    return df, metrics


if __name__ == "__main__":

    # Number of simulations
    num_simulations = 1000

    # Store all time-series data
    all_data = []

    # Store metrics from every simulation
    all_metrics = []

    print("Generating simulations...")

    for sim in range(num_simulations):

        df, metrics = run_simulation()

        # Add simulation ID
        df["simulation_id"] = sim

        # Store time-series data
        all_data.append(df)

        # Store metrics
        metrics["simulation_id"] = sim
        all_metrics.append(metrics)

        # Show progress
        if (sim + 1) % 100 == 0:
            print(
                f"Completed {sim + 1}/{num_simulations}"
            )

    # ----------------------------------------------
    # Combine all simulations
    # ----------------------------------------------

    dataset = pd.concat(
        all_data,
        ignore_index=True
    )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    # ----------------------------------------------
    # Save datasets
    # ----------------------------------------------

    dataset.to_csv(
        "../data/rssi_dataset.csv",
        index=False
    )

    metrics_df.to_csv(
        "../data/simulation_metrics.csv",
        index=False
    )

    # ----------------------------------------------
    # Print information
    # ----------------------------------------------

    print("\nDataset generation completed.")

    print(
        "RSSI dataset shape:",
        dataset.shape
    )

    print(
        "Metrics dataset shape:",
        metrics_df.shape
    )

    print("\nRSSI dataset saved to:")
    print("../data/rssi_dataset.csv")

    print("\nMetrics saved to:")
    print("../data/simulation_metrics.csv")
