
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


def calculate_average_serving_rssi(df):

    values = []

    for _, row in df.iterrows():

        serving_bs = row["serving_bs"]

        power = row[
            f"{serving_bs}_power"
        ]

        values.append(power)

    if not values:
        return 0

    return sum(values) / len(values)


def count_outage_samples(
    df,
    threshold=-100
):

    count = 0

    for _, row in df.iterrows():

        serving_bs = row["serving_bs"]

        power = row[
            f"{serving_bs}_power"
        ]

        if power < threshold:
            count += 1

    return count

