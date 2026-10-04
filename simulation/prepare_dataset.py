
import pandas as pd


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("../data/rssi_dataset.csv")


# --------------------------------------------------
# 2. Basic information
# --------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 3. Number of simulations
# --------------------------------------------------

print("\nNumber of simulations:")
print(df["simulation_id"].nunique())


# --------------------------------------------------
# 4. Samples per simulation
# --------------------------------------------------

samples = df.groupby(
    "simulation_id"
).size()

print("\nSamples per simulation:")
print(samples.describe())


# --------------------------------------------------
# 5. RSSI statistics
# --------------------------------------------------

rssi_columns = [
    "BS1_power",
    "BS2_power",
    "BS3_power"
]

print("\nRSSI statistics:")
print(df[rssi_columns].describe())


# --------------------------------------------------
# 6. Speed statistics
# --------------------------------------------------

print("\nSpeed statistics:")
print(df["speed"].describe())


# --------------------------------------------------
# 7. Direction statistics
# --------------------------------------------------

print("\nDirection statistics:")
print(df["direction"].describe())


# --------------------------------------------------
# 8. Split simulations
# --------------------------------------------------

simulation_ids = df["simulation_id"].unique()

train_ids = simulation_ids[:700]
val_ids = simulation_ids[700:850]
test_ids = simulation_ids[850:1000]


# --------------------------------------------------
# 9. Create datasets
# --------------------------------------------------

train_df = df[
    df["simulation_id"].isin(train_ids)
]

val_df = df[
    df["simulation_id"].isin(val_ids)
]

test_df = df[
    df["simulation_id"].isin(test_ids)
]


# --------------------------------------------------
# 10. Print split information
# --------------------------------------------------

print("\nDataset split:")

print(
    "Training:",
    train_df.shape,
    "| Simulations:",
    train_df["simulation_id"].nunique()
)

print(
    "Validation:",
    val_df.shape,
    "| Simulations:",
    val_df["simulation_id"].nunique()
)

print(
    "Testing:",
    test_df.shape,
    "| Simulations:",
    test_df["simulation_id"].nunique()
)


# --------------------------------------------------
# 11. Save splits
# --------------------------------------------------

train_df.to_csv(
    "../data/train.csv",
    index=False
)

val_df.to_csv(
    "../data/validation.csv",
    index=False
)

test_df.to_csv(
    "../data/test.csv",
    index=False
)

print("\nDatasets saved successfully.")

