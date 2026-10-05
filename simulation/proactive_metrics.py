def count_handovers(handovers):
    return len(handovers)


def count_ping_pong(handovers, window=10):

    count = 0

    for i in range(1, len(handovers)):

        previous = handovers[i - 1]
        current = handovers[i]

        time_difference = (
            current["time"] - previous["time"]
        )

        if (
            time_difference <= window
            and current["to"] == previous["from"]
            and current["from"] == previous["to"]
        ):
            count += 1

    return count


def calculate_average_serving_rssi(
    actual_rssi,
    serving_history,
    bs_ids
):

    values = []

    for i in range(len(serving_history)):

        serving_bs = serving_history[i]

        bs_index = bs_ids.index(serving_bs)

        power = actual_rssi[i][bs_index]

        values.append(power)

    if not values:
        return 0

    return sum(values) / len(values)


def count_outage_samples(
    actual_rssi,
    serving_history,
    bs_ids,
    threshold=-100
):

    count = 0

    for i in range(len(serving_history)):

        serving_bs = serving_history[i]

        bs_index = bs_ids.index(serving_bs)

        power = actual_rssi[i][bs_index]

        if power < threshold:
            count += 1

    return count