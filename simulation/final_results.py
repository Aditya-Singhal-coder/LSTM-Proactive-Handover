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


# -----------------------------
# Load test data
# -----------------------------

X_test = np.load("../data/X_test_scaled.npy")
y_test = np.load("../data/y_test_scaled.npy")

target_scaler = joblib.load(
    "../data/target_scaler.pkl"
)


# -----------------------------
# Convert actual RSSI
# -----------------------------

actual_rssi = target_scaler.inverse_transform(
    y_test
)

bs_ids = ["BS1", "BS2", "BS3"]


df = pd.DataFrame({
    "time": np.arange(len(actual_rssi)),
    "BS1_power": actual_rssi[:, 0],
    "BS2_power": actual_rssi[:, 1],
    "BS3_power": actual_rssi[:, 2]
})


# -----------------------------
# Reactive method
# -----------------------------

print("Running reactive handover...")

reactive_serving, reactive_handovers = reactive_handover(
    df,
    bs_ids,
    hysteresis=3.0,
    ttt=3
)


# -----------------------------
# Load LSTM
# -----------------------------

model = load_model(
    "../models/lstm_rssi_model.keras"
)


# -----------------------------
# Generate predictions
# -----------------------------

print("Generating LSTM predictions...")

pred_scaled = model.predict(
    X_test,
    verbose=0
)

predictions = target_scaler.inverse_transform(
    pred_scaled
)


# -----------------------------
# Final proactive configuration
# -----------------------------

threshold = -90
margin = 2
ttt = 2


print("\nFinal proactive configuration:")
print("Threshold:", threshold, "dBm")
print("Margin:", margin, "dB")
print("TTT:", ttt)


# -----------------------------
# Proactive method
# -----------------------------

proactive_serving, proactive_handovers = proactive_handover(
    predictions,
    bs_ids,
    threshold=threshold,
    margin=margin,
    ttt=ttt
)


# -----------------------------
# Reactive metrics
# -----------------------------

r_handover = count_handovers(
    reactive_handovers
)

r_ping = count_ping_pong(
    reactive_handovers,
    window=10
)

r_rssi = calculate_average_serving_rssi(
    actual_rssi,
    reactive_serving,
    bs_ids
)

r_outage = count_outage_samples(
    actual_rssi,
    reactive_serving,
    bs_ids,
    threshold=-100
)


# -----------------------------
# Proactive metrics
# -----------------------------

p_handover = count_handovers(
    proactive_handovers
)

p_ping = count_ping_pong(
    proactive_handovers,
    window=10
)

p_rssi = calculate_average_serving_rssi(
    actual_rssi,
    proactive_serving,
    bs_ids
)

p_outage = count_outage_samples(
    actual_rssi,
    proactive_serving,
    bs_ids,
    threshold=-100
)


# -----------------------------
# Improvement calculations
# -----------------------------

handover_reduction = (
    (r_handover - p_handover)
    / r_handover
) * 100

ping_reduction = (
    (r_ping - p_ping)
    / r_ping
) * 100

outage_reduction = (
    (r_outage - p_outage)
    / r_outage
) * 100


# -----------------------------
# Final results
# -----------------------------

print("\n")
print("=" * 70)
print("FINAL REACTIVE vs PROACTIVE RESULTS")
print("=" * 70)

print(
    f"{'Metric':<30}"
    f"{'Reactive':>15}"
    f"{'Proactive':>15}"
)

print("-" * 70)

print(
    f"{'Total Handovers':<30}"
    f"{r_handover:>15}"
    f"{p_handover:>15}"
)

print(
    f"{'Ping-Pong Handovers':<30}"
    f"{r_ping:>15}"
    f"{p_ping:>15}"
)

print(
    f"{'Average Serving RSSI (dBm)':<30}"
    f"{r_rssi:>15.2f}"
    f"{p_rssi:>15.2f}"
)

print(
    f"{'Outage Samples':<30}"
    f"{r_outage:>15}"
    f"{p_outage:>15}"
)

print("-" * 70)

print(
    f"{'Handover Reduction (%)':<30}"
    f"{handover_reduction:>15.2f}"
)

print(
    f"{'Ping-Pong Reduction (%)':<30}"
    f"{ping_reduction:>15.2f}"
)

print(
    f"{'Outage Reduction (%)':<30}"
    f"{outage_reduction:>15.2f}"
)

print("=" * 70)