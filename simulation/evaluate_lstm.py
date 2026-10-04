
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
# 3. Load target scaler
# --------------------------------------------------

target_scaler = joblib.load(
    "../data/target_scaler.pkl"
)


# --------------------------------------------------
# 4. Predict
# --------------------------------------------------

print("Generating predictions...")

y_pred_scaled = model.predict(
    X_test,
    verbose=1
)


# --------------------------------------------------
# 5. Convert back to dBm
# --------------------------------------------------

y_test = target_scaler.inverse_transform(
    y_test_scaled
)

y_pred = target_scaler.inverse_transform(
    y_pred_scaled
)


# --------------------------------------------------
# 6. Calculate metrics
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)


print("\nLSTM Test Results")
print("---------------------------")

print(
    "MAE:",
    round(mae, 3),
    "dBm"
)

print(
    "RMSE:",
    round(rmse, 3),
    "dBm"
)


# --------------------------------------------------
# 7. Calculate error for each BS
# --------------------------------------------------

for i, bs in enumerate(
    ["BS1", "BS2", "BS3"]
):

    bs_mae = mean_absolute_error(
        y_test[:, i],
        y_pred[:, i]
    )

    bs_rmse = np.sqrt(
        mean_squared_error(
            y_test[:, i],
            y_pred[:, i]
        )
    )

    print(
        f"\n{bs}:"
    )

    print(
        "MAE:",
        round(bs_mae, 3),
        "dBm"
    )

    print(
        "RMSE:",
        round(bs_rmse, 3),
        "dBm"
    )


# --------------------------------------------------
# 8. Show sample predictions
# --------------------------------------------------

print("\nSample predictions:")
print("---------------------------")

for i in range(10):

    print(
        f"Sample {i + 1}:"
    )

    print(
        "Actual :",
        np.round(y_test[i], 2)
    )

    print(
        "Predicted:",
        np.round(y_pred[i], 2)
    )

    print()

