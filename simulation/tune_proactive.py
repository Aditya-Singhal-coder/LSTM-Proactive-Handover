import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from handover import reactive_handover
from proactive_handover import proactive_handover
from proactive_metrics import (
    count_handovers,
    count_ping_pong,
    calculate_average_serving_rssi,
    count_outage_samples
)


# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

X_test = np.load("../data/X_test_scaled.npy")
y_test = np.load("../data/y_test_scaled.npy")

target_scaler = joblib.load(
    "../data/target_scaler.pkl"
)


# --------------------------------------------------
# 2. Convert actual RSSI back to dBm
# --------------------------------------------------

actual_rssi = target_scaler.inverse_transform(y_test)


# --------------------------------------------------
# 3. Base stations
# --------------------------------------------------

bs_ids = ["BS1", "BS2", "BS3"]


# --------------------------------------------------
# 4. Create DataFrame
# --------------------------------------------------

df = pd.DataFrame({
    "time": np.arange(len(actual_rssi)),
    "BS1_power": actual_rssi[:, 0],
    "BS2_power": actual_rssi[:, 1],
    "BS3_power": actual_rssi[:, 2]
})


# --------------------------------------------------
# 5. Load LSTM
# --------------------------------------------------

model = load_model(
    "../models/lstm_rssi_model.keras"
)


# --------------------------------------------------
# 6. Generate predictions
# --------------------------------------------------

print("Generating LSTM predictions...")

predictions_scaled = model.predict(
    X_test,
    verbose=0
)

predictions = target_scaler.inverse_transform(
    predictions_scaled
)


# --------------------------------------------------
# 7. Parameter combinations
# --------------------------------------------------

thresholds = [-90, -92, -95, -98]
margins = [1, 2, 3, 4]
ttts = [2, 3, 4, 5]


results = []


# --------------------------------------------------
# 8. Test every combination
# --------------------------------------------------

for threshold in thresholds:

    for margin in margins:

        for ttt in ttts:

            serving, handovers = proactive_handover(
                predictions,
                bs_ids,
                threshold=threshold,
                margin=margin,
                ttt=ttt
            )

            handover_count = count_handovers(
                handovers
            )

            ping_pong = count_ping_pong(
                handovers,
                window=10
            )

            avg_rssi = calculate_average_serving_rssi(
                actual_rssi,
                serving,
                bs_ids
            )

            outages = count_outage_samples(
                actual_rssi,
                serving,
                bs_ids,
                threshold=-100
            )

            results.append({
                "threshold": threshold,
                "margin": margin,
                "ttt": ttt,
                "handovers": handover_count,
                "ping_pong": ping_pong,
                "avg_rssi": avg_rssi,
                "outages": outages
            })


# --------------------------------------------------
# 9. Convert to DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)


# --------------------------------------------------
# 10. Sort by outage first
# --------------------------------------------------

results_df = results_df.sort_values(
    by=["outages", "handovers"]
)


# --------------------------------------------------
# 11. Display best configurations
# --------------------------------------------------

print("\n")
print("=" * 80)
print("TOP PROACTIVE HANDOVER CONFIGURATIONS")
print("=" * 80)

print(
    results_df.head(15).to_string(
        index=False
    )
)


# --------------------------------------------------
# 12. Save results
# --------------------------------------------------

results_df.to_csv(
    "../data/proactive_tuning_results.csv",
    index=False
)

print("\nResults saved to:")
print("../data/proactive_tuning_results.csv")