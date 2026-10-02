
import matplotlib.pyplot as plt
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

def main():

    # --------------------------------------------------
    # 1. Simulation parameters
    # --------------------------------------------------

    simulation_time = 100
    transmit_power = 30
    shadowing_std = 6

    # --------------------------------------------------
    # 2. Create base stations
    # --------------------------------------------------

    base_stations = create_base_stations()

    # --------------------------------------------------
    # 3. Create mobile user
    # --------------------------------------------------

    user = MobileUser(
        x=-50,
        y=50,
        speed=10,
        direction=0
    )

    # --------------------------------------------------
    # 4. Store simulation data
    # --------------------------------------------------

    user_data = []

    # --------------------------------------------------
    # 5. Run simulation
    # --------------------------------------------------

    for t in range(simulation_time):

        # Store current user state
        row = {
            "time": t,
            "x": user.x,
            "y": user.y,
            "speed": user.speed,
            "direction": user.direction
        }

        # --------------------------------------------------
        # Calculate channel information for every BS
        # --------------------------------------------------

        for bs in base_stations:

            # Distance between user and BS
            distance = calculate_distance(
                user.x,
                user.y,
                bs.x,
                bs.y
            )

            # Path loss
            path_loss = calculate_path_loss(
                distance
            )

            # Shadowing
            shadowing = calculate_shadowing(
                shadowing_std
            )

            # Rayleigh fading
            fading = calculate_rayleigh_fading()

            # Final received signal power
            received_power = calculate_received_power(
                transmit_power,
                path_loss,
                shadowing,
                fading
            )

            # Store channel information
            row[f"{bs.bs_id}_distance"] = distance
            row[f"{bs.bs_id}_path_loss"] = path_loss
            row[f"{bs.bs_id}_shadowing"] = shadowing
            row[f"{bs.bs_id}_fading"] = fading
            row[f"{bs.bs_id}_power"] = received_power

        # Store current timestep
        user_data.append(row)

        # Move user to next position
        user.move(time_step=1)

    # --------------------------------------------------
    # 6. Create DataFrame
    # --------------------------------------------------

    df = pd.DataFrame(user_data)

    print("\nSimulation Data:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    # --------------------------------------------------
    # 7. Reactive handover simulation
    # --------------------------------------------------

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

    # Store serving BS for every timestep
    df["serving_bs"] = serving_history

    # --------------------------------------------------
    # Performance metrics
    # --------------------------------------------------

    total_handovers = count_handovers(handovers)

    ping_pong = count_ping_pong(
        handovers,
        window=10
    )

    average_rssi = calculate_average_serving_rssi(df)

    outage_samples = count_outage_samples(
        df,
        threshold=-100
    )

    print("\nPerformance Metrics")
    print("-------------------")

    print(
        "Total handovers:",
        total_handovers
    )

    print(
        "Ping-pong handovers:",
        ping_pong
    )

    print(
        "Average serving RSSI:",
        round(average_rssi, 2),
        "dBm"
    )

    print(
        "Outage samples:",
        outage_samples
    )

    # --------------------------------------------------
    # 8. Print handover events
    # --------------------------------------------------

    print("\nReactive Handover Events:")

    if len(handovers) == 0:

        print("No handovers occurred.")

    else:

        for event in handovers:

            print(
                f"Time {event['time']}s: "
                f"{event['from']} -> {event['to']} "
                f"| "
                f"{event['serving_power']:.2f} dBm -> "
                f"{event['target_power']:.2f} dBm"
            )

    print(
        "\nTotal handovers:",
        len(handovers)
    )

    # --------------------------------------------------
    # 9. Plot RSSI / received power
    # --------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["time"],
        df["BS1_power"],
        label="BS1"
    )

    plt.plot(
        df["time"],
        df["BS2_power"],
        label="BS2"
    )

    plt.plot(
        df["time"],
        df["BS3_power"],
        label="BS3"
    )

    # Mark handover events
    for event in handovers:

        plt.axvline(
            event["time"],
            linestyle="--"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Received Power (dBm)")
    plt.title("Received Signal Power vs Time")

    plt.legend()
    plt.grid(True)


    # --------------------------------------------------
    # 10. Plot user trajectory
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    # Plot base stations
    for bs in base_stations:

        plt.scatter(
            bs.x,
            bs.y,
            s=100
        )

        plt.text(
            bs.x,
            bs.y + 15,
            bs.bs_id
        )

    # Plot user trajectory
    plt.plot(
        df["x"],
        df["y"],
        label="User trajectory"
    )

    # Starting point
    plt.scatter(
        df["x"].iloc[0],
        df["y"].iloc[0],
        marker="o",
        s=80,
        label="Start"
    )

    # Ending point
    plt.scatter(
        df["x"].iloc[-1],
        df["y"].iloc[-1],
        marker="x",
        s=100,
        label="End"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Mobile User Trajectory")

    plt.legend()
    plt.grid(True)


    # --------------------------------------------------
    # 11. Plot reactive handover decisions
    # --------------------------------------------------

    bs_map = {
        "BS1": 1,
        "BS2": 2,
        "BS3": 3
    }

    df["serving_id"] = df["serving_bs"].map(bs_map)

    plt.figure(figsize=(10, 4))

    plt.step(
        df["time"],
        df["serving_id"],
        where="post"
    )

    plt.yticks(
        [1, 2, 3],
        ["BS1", "BS2", "BS3"]
    )

    # Mark handover events
    for event in handovers:

        plt.axvline(
            event["time"],
            linestyle="--"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Serving Base Station")
    plt.title("Reactive Handover Decisions")

    plt.grid(True)


    # --------------------------------------------------
    # Show all three figures together
    # --------------------------------------------------

    plt.show()


    


if __name__ == "__main__":
    main()

   
