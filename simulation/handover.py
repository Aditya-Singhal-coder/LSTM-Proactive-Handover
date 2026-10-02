def reactive_handover(
    df,
    bs_ids,
    hysteresis=3.0,
    ttt=3
):
    """
    Simulate a reactive handover using an A3-style rule.

    Handover condition:

        neighbor RSSI > serving RSSI + hysteresis

    The condition must remain true for TTT
    consecutive samples before handover.
    """

    # Start with the strongest BS
    first_powers = {
        bs: df.loc[0, f"{bs}_power"]
        for bs in bs_ids
    }

    serving_bs = max(
        first_powers,
        key=first_powers.get
    )

    handovers = []

    candidate_bs = None
    candidate_count = 0

    serving_history = []

    for i in range(len(df)):

        serving_power = df.loc[
            i,
            f"{serving_bs}_power"
        ]

        # Find strongest neighbor
        neighbors = [
            bs for bs in bs_ids
            if bs != serving_bs
        ]

        best_neighbor = max(
            neighbors,
            key=lambda bs: df.loc[i, f"{bs}_power"]
        )

        neighbor_power = df.loc[
            i,
            f"{best_neighbor}_power"
        ]

        # Check A3-style condition
        if neighbor_power > serving_power + hysteresis:

            # New candidate
            if candidate_bs != best_neighbor:
                candidate_bs = best_neighbor
                candidate_count = 1

            else:
                candidate_count += 1

            # Trigger handover
            if candidate_count >= ttt:

                handovers.append({
                    "time": df.loc[i, "time"],
                    "from": serving_bs,
                    "to": candidate_bs,
                    "serving_power": serving_power,
                    "target_power": neighbor_power
                })

                serving_bs = candidate_bs

                candidate_bs = None
                candidate_count = 0

        else:

            candidate_bs = None
            candidate_count = 0

        serving_history.append(serving_bs)

    return serving_history, handovers

