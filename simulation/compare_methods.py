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

X_test = np.load(
    "../data/X_test_scaled.npy"
)

y_test = np.load(
    "../data/y_test_scaled.npy"
)

target_scaler = joblib.load(
    "../data/target_scaler.pkl"
)


# --------------------------------------------------
# 2. Convert actual RSSI back to dBm
# --------------------------------------------------

actual_rssi = target_scaler.inverse_transform(
    y_test
)


print("Test RSSI shape:")
print(actual_rssi.shape)


# --------------------------------------------------
# 3. Base station IDs
# --------------------------------------------------

bs_ids = [
    "BS1",
    "BS2",
    "BS3"
]


# --------------------------------------------------
# 4. Create DataFrame for reactive method
# --------------------------------------------------

df = pd.DataFrame({
    "time": np.arange(len(actual_rssi)),
    "BS1_power": actual_rssi[:, 0],
    "BS2_power": actual_rssi[:, 1],
    "BS3_power": actual_rssi[:, 2]
})


# --------------------------------------------------
# 5. Run reactive handover
# --------------------------------------------------

print("\nRunning reactive handover...")

reactive_serving, reactive_handovers = reactive_handover(
    df,
    bs_ids,
    hysteresis=3.0,
    ttt=3
)


print(
    "Reactive handovers:",
    len(reactive_handovers)
)


# --------------------------------------------------
# 6. Load LSTM model
# --------------------------------------------------

model = load_model(
    "../models/lstm_rssi_model.keras"
)


# --------------------------------------------------
# 7. Generate LSTM predictions
# --------------------------------------------------

print("\nGenerating LSTM predictions...")

predictions_scaled = model.predict(
    X_test,
    verbose=0
)


predictions = target_scaler.inverse_transform(
    predictions_scaled
)


# --------------------------------------------------
# 8. Run proactive handover
# --------------------------------------------------

print("\nRunning proactive handover...")

proactive_serving, proactive_handovers = proactive_handover(
    predictions,
    bs_ids,
    threshold=-95,
    margin=3,
    ttt=3
)


print(
    "Proactive handovers:",
    len(proactive_handovers)
)


# --------------------------------------------------
# 9. Reactive metrics
# --------------------------------------------------

reactive_handover_count = count_handovers(
    reactive_handovers
)

reactive_ping_pong = count_ping_pong(
    reactive_handovers,
    window=10
)

reactive_avg_rssi = calculate_average_serving_rssi(
    actual_rssi,
    reactive_serving,
    bs_ids
)

reactive_outage = count_outage_samples(
    actual_rssi,
    reactive_serving,
    bs_ids,
    threshold=-100
)


# --------------------------------------------------
# 10. Proactive metrics
# --------------------------------------------------

proactive_handover_count = count_handovers(
    proactive_handovers
)

proactive_ping_pong = count_ping_pong(
    proactive_handovers,
    window=10
)

proactive_avg_rssi = calculate_average_serving_rssi(
    actual_rssi,
    proactive_serving,
    bs_ids
)

proactive_outage = count_outage_samples(
    actual_rssi,
    proactive_serving,
    bs_ids,
    threshold=-100
)


# --------------------------------------------------
# 11. Display comparison
# --------------------------------------------------

print("\n")
print("=" * 60)
print("REACTIVE vs PROACTIVE COMPARISON")
print("=" * 60)

print(
    f"{'Metric':<30}"
    f"{'Reactive':>15}"
    f"{'Proactive':>15}"
)

print("-" * 60)

print(
    f"{'Total Handovers':<30}"
    f"{reactive_handover_count:>15}"
    f"{proactive_handover_count:>15}"
)

print(
    f"{'Ping-Pong Handovers':<30}"
    f"{reactive_ping_pong:>15}"
    f"{proactive_ping_pong:>15}"
)

print(
    f"{'Average Serving RSSI (dBm)':<30}"
    f"{reactive_avg_rssi:>15.2f}"
    f"{proactive_avg_rssi:>15.2f}"
)

print(
    f"{'Outage Samples':<30}"
    f"{reactive_outage:>15}"
    f"{proactive_outage:>15}"
)

print("=" * 60)