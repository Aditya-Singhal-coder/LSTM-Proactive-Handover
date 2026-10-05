import numpy as np
import joblib
from tensorflow.keras.models import load_model

from proactive_handover import proactive_handover

from proactive_metrics import (
    count_handovers,
    count_ping_pong,
    calculate_average_serving_rssi,
    count_outage_samples
)

# --------------------------------------------------
# 1. Load trained LSTM model
# --------------------------------------------------

model = load_model(
    "../models/lstm_rssi_model.keras"
)


# --------------------------------------------------
# 2. Load scaled test data
# --------------------------------------------------

X_test = np.load(
    "../data/X_test_scaled.npy"
)

y_test = np.load(
    "../data/y_test_scaled.npy"
)


# --------------------------------------------------
# 3. Load target scaler
# --------------------------------------------------

target_scaler = joblib.load(
    "../data/target_scaler.pkl"
)


print("Test data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# 4. Generate LSTM predictions
# --------------------------------------------------

print("\nGenerating LSTM predictions...")

predictions_scaled = model.predict(
    X_test,
    verbose=0
)


# --------------------------------------------------
# 5. Convert predictions back to dBm
# --------------------------------------------------

predictions = target_scaler.inverse_transform(
    predictions_scaled
)


print("\nPrediction shape:")
print(predictions.shape)


# --------------------------------------------------
# 6. Base station IDs
# --------------------------------------------------

bs_ids = [
    "BS1",
    "BS2",
    "BS3"
]


# --------------------------------------------------
# 7. Run proactive handover
# --------------------------------------------------

serving_history, handovers = proactive_handover(
    predictions,
    bs_ids,
    threshold=-100,
    margin=2
)


# --------------------------------------------------
# 8. Print handover events
# --------------------------------------------------

print("\nProactive Handover Events:")
print("--------------------------------")

for event in handovers:

    print(
        f"Time {event['time']}s: "
        f"{event['from']} -> {event['to']} | "
        f"{event['current_power']:.2f} dBm -> "
        f"{event['target_power']:.2f} dBm"
    )


# --------------------------------------------------
# 9. Total handovers
# --------------------------------------------------

print("\nTotal proactive handovers:")
print(len(handovers))

# --------------------------------------------------
# 11. Convert actual test RSSI back to dBm
# --------------------------------------------------

actual_rssi = target_scaler.inverse_transform(
    y_test
)


# --------------------------------------------------
# 12. Calculate proactive metrics
# --------------------------------------------------

total_handovers = count_handovers(
    handovers
)

ping_pong = count_ping_pong(
    handovers,
    window=10
)

average_rssi = calculate_average_serving_rssi(
    actual_rssi,
    serving_history,
    bs_ids
)

outage_samples = count_outage_samples(
    actual_rssi,
    serving_history,
    bs_ids,
    threshold=-100
)


# --------------------------------------------------
# 13. Print metrics
# --------------------------------------------------

print("\nProactive Handover Metrics")
print("--------------------------------")

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
# 10. Show first few predictions
# --------------------------------------------------

print("\nSample predictions:")
print("--------------------------------")

for i in range(min(10, len(predictions))):

    print(
        f"Sample {i + 1}: "
        f"BS1={predictions[i][0]:.2f} dBm, "
        f"BS2={predictions[i][1]:.2f} dBm, "
        f"BS3={predictions[i][2]:.2f} dBm"
    )