
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------
# 1. Load scaled data
# --------------------------------------------------

X_train = np.load(
    "../data/X_train_scaled.npy"
)

y_train = np.load(
    "../data/y_train_scaled.npy"
)

X_val = np.load(
    "../data/X_validation_scaled.npy"
)

y_val = np.load(
    "../data/y_validation_scaled.npy"
)


print("Training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation data:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# --------------------------------------------------
# 2. Create LSTM model
# --------------------------------------------------

model = Sequential([

    LSTM(
        64,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(3)
])


# --------------------------------------------------
# 3. Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# --------------------------------------------------
# 4. Display model
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# 5. Early stopping
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=64,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# --------------------------------------------------
# 7. Save trained model
# --------------------------------------------------

model.save(
    "../models/lstm_rssi_model.keras"
)


print("\nModel training completed.")

print(
    "Model saved to:",
    "../models/lstm_rssi_model.keras"
)

