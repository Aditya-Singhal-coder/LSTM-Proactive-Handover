
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model


# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

X_test = np.load(
    "../data/X_test_scaled.npy"
)

y_test_scaled = np.load(
    "../data/y_test_scaled.npy"
)


# --------------------------------------------------
# 2. Load model
# --------------------------------------------------

model = load_model(
    "../models/lstm_rssi_model.keras"
)


# --------------------------------------------------
# 3. Load scaler
# --------------------------------------------------

target_scaler = joblib.load(
    "../data/target_scaler.pkl"
)


# --------------------------------------------------
# 4. Predict
# --------------------------------------------------

y_pred_scaled = model.predict(
    X_test,
    verbose=0
)


# --------------------------------------------------
# 5. Convert to dBm
# --------------------------------------------------

y_test = target_scaler.inverse_transform(
    y_test_scaled
)

y_pred = target_scaler.inverse_transform(
    y_pred_scaled
)


# --------------------------------------------------
# 6. Plot BS1
# --------------------------------------------------

samples = 300

plt.figure(figsize=(12, 5))

plt.plot(
    y_test[:samples, 0],
    label="Actual BS1 RSSI"
)

plt.plot(
    y_pred[:samples, 0],
    label="Predicted BS1 RSSI"
)

plt.xlabel("Test Sample")
plt.ylabel("RSSI (dBm)")
plt.title("LSTM Prediction vs Actual RSSI - BS1")

plt.legend()
plt.grid(True)

plt.show()


# --------------------------------------------------
# 7. Plot BS2
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    y_test[:samples, 1],
    label="Actual BS2 RSSI"
)

plt.plot(
    y_pred[:samples, 1],
    label="Predicted BS2 RSSI"
)

plt.xlabel("Test Sample")
plt.ylabel("RSSI (dBm)")
plt.title("LSTM Prediction vs Actual RSSI - BS2")

plt.legend()
plt.grid(True)

plt.show()


# --------------------------------------------------
# 8. Plot BS3
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    y_test[:samples, 2],
    label="Actual BS3 RSSI"
)

plt.plot(
    y_pred[:samples, 2],
    label="Predicted BS3 RSSI"
)

plt.xlabel("Test Sample")
plt.ylabel("RSSI (dBm)")
plt.title("LSTM Prediction vs Actual RSSI - BS3")

plt.legend()
plt.grid(True)


plt.show()
