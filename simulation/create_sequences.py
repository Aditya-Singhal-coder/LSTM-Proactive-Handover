
import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEQUENCE_LENGTH = 10
PREDICTION_HORIZON = 5

FEATURES = [
    "BS1_power",
    "BS2_power",
    "BS3_power",
    "speed",
    "direction"
]

TARGETS = [
    "BS1_power",
    "BS2_power",
    "BS3_power"
]


# --------------------------------------------------
# Create sequences for one simulation
# --------------------------------------------------

def create_sequences(df):

    x = []
    y = []

    values = df[FEATURES].values
    targets = df[TARGETS].values

    total_rows = len(df)

    for i in range(
        SEQUENCE_LENGTH,
        total_rows - PREDICTION_HORIZON + 1
    ):

        # Previous 10 seconds
        input_sequence = values[
            i - SEQUENCE_LENGTH:i
        ]

        # Future RSSI
        target = targets[
            i + PREDICTION_HORIZON - 1
        ]

        x.append(input_sequence)
        y.append(target)

    return np.array(x), np.array(y)


# --------------------------------------------------
# Process complete dataset
# --------------------------------------------------

def process_dataset(file_path):

    df = pd.read_csv(file_path)

    all_x = []
    all_y = []

    # Process each trajectory separately
    for simulation_id, group in df.groupby(
        "simulation_id"
    ):

        # Sort by time
        group = group.sort_values("time")

        x, y = create_sequences(group)

        if len(x) > 0:

            all_x.append(x)
            all_y.append(y)

    # Combine all simulations
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)

    return x, y


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Creating training sequences...")

    x_train, y_train = process_dataset(
        "../data/train.csv"
    )

    print("Creating validation sequences...")

    x_val, y_val = process_dataset(
        "../data/validation.csv"
    )

    print("Creating testing sequences...")

    x_test, y_test = process_dataset(
        "../data/test.csv"
    )

    # --------------------------------------------------
    # Print shapes
    # --------------------------------------------------

    print("\nSequence shapes:")
    
    print(
        "X train:",
        x_train.shape
    )

    print(
        "Y train:",
        y_train.shape
    )

    print(
        "X validation:",
        x_val.shape
    )

    print(
        "Y validation:",
        y_val.shape
    )

    print(
        "X test:",
        x_test.shape
    )

    print(
        "Y test:",
        y_test.shape
    )

    # --------------------------------------------------
    # Save sequences
    # --------------------------------------------------

    np.save(
        "../data/X_train.npy",
        x_train
    )

    np.save(
        "../data/y_train.npy",
        y_train
    )

    np.save(
        "../data/X_validation.npy",
        x_val
    )

    np.save(
        "../data/y_validation.npy",
        y_val
    )

    np.save(
        "../data/X_test.npy",
        x_test
    )

    np.save(
        "../data/y_test.npy",
        y_test
    )

    print("\nSequences saved successfully.")
