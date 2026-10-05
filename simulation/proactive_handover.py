import numpy as np


def proactive_handover(
    predictions,
    bs_ids,
    threshold=-90,
    margin=2,
    ttt=2
):
    serving_history = []
    handovers = []

    current_bs = bs_ids[0]

    candidate_bs = None
    candidate_count = 0

    for i in range(len(predictions)):

        predicted = predictions[i]

        current_index = bs_ids.index(current_bs)
        current_power = predicted[current_index]

        best_index = np.argmax(predicted)
        best_bs = bs_ids[best_index]
        best_power = predicted[best_index]

        # Check proactive condition
        condition = (
            best_bs != current_bs
            and best_power > current_power + margin
            and current_power < threshold
        )

        if condition:

            # Start counting a new candidate
            if candidate_bs != best_bs:
                candidate_bs = best_bs
                candidate_count = 1

            else:
                candidate_count += 1

            # Handover only after TTT
            if candidate_count >= ttt:

                handovers.append({
                    "time": i,
                    "from": current_bs,
                    "to": candidate_bs,
                    "current_power": current_power,
                    "target_power": best_power
                })

                current_bs = candidate_bs

                candidate_bs = None
                candidate_count = 0

        else:

            # Condition was broken
            candidate_bs = None
            candidate_count = 0

        serving_history.append(current_bs)

    return serving_history, handovers