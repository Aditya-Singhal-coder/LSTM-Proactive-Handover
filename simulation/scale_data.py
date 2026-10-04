
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


# --------------------------------------------------
# Load sequences
# --------------------------------------------------

X_train = np.load("../data/X_train.npy")
X_val = np.load("../data/X_validation.npy")
X_test = np.load("../data/X_test.npy")

y_train = np.load("../data/y_train.npy")
y_val = np.load("../data/y_validation.npy")
y_test = np.load("../data/y_test.npy")


print("Original shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# --------------------------------------------------
# Create scaler
# --------------------------------------------------

scaler = StandardScaler()


# --------------------------------------------------
# Fit scaler only on training data
# --------------------------------------------------

# Reshape:
# (samples, timesteps, features)
#        ↓
# (samples * timesteps, features)

train_2d = X_train.reshape(
    -1,
    X_train.shape[-1]
)

scaler.fit(train_2d)


# --------------------------------------------------
# Scale all datasets
# --------------------------------------------------

X_train_scaled = scaler.transform(
    train_2d
).reshape(X_train.shape)

X_val_scaled = scaler.transform(
    X_val.reshape(-1, X_val.shape[-1])
).reshape(X_val.shape)

X_test_scaled = scaler.transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)


# --------------------------------------------------
# Scale target values
# --------------------------------------------------

target_scaler = StandardScaler()

target_scaler.fit(y_train)

y_train_scaled = target_scaler.transform(
    y_train
)

y_val_scaled = target_scaler.transform(
    y_val
)

y_test_scaled = target_scaler.transform(
    y_test
)


# --------------------------------------------------
# Save scaled data
# --------------------------------------------------

np.save(
    "../data/X_train_scaled.npy",
    X_train_scaled
)

np.save(
    "../data/X_validation_scaled.npy",
    X_val_scaled
)

np.save(
    "../data/X_test_scaled.npy",
    X_test_scaled
)

np.save(
    "../data/y_train_scaled.npy",
    y_train_scaled
)

np.save(
    "../data/y_validation_scaled.npy",
    y_val_scaled
)

np.save(
    "../data/y_test_scaled.npy",
    y_test_scaled
)


# --------------------------------------------------
# Save scalers
# --------------------------------------------------

joblib.dump(
    scaler,
    "../data/input_scaler.pkl"
)

joblib.dump(
    target_scaler,
    "../data/target_scaler.pkl"
)


# --------------------------------------------------
# Verify
# --------------------------------------------------

print("\nScaled shapes:")

print(
    "X_train:",
    X_train_scaled.shape
)

print(
    "X_val:",
    X_val_scaled.shape
)

print(
    "X_test:",
    X_test_scaled.shape
)

print(
    "y_train:",
    y_train_scaled.shape
)

print(
    "y_val:",
    y_val_scaled.shape
)

print(
    "y_test:",
    y_test_scaled.shape
)

print("\nScaling completed successfully.")

