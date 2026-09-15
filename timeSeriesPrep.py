import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("Toronto Island Ferry Tickets.csv")
print("\nFirst 20 IDs:")
print(df["_id"].head(20).to_list())

print("\nLast 20 IDs:")
print(df["_id"].tail(20).to_list())

print("Original columns:")
print(df.columns.tolist())

print("\nOriginal shape:")
print(df.shape)


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nCleaned columns:")
print(df.columns.tolist())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. CONVERT TIMESTAMP
# ============================================================

# Let pandas detect the timestamp format
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

print("\nTimestamp conversion:")
print(df["Timestamp"].head(20))

print("\nNumber of valid timestamps:")
print(df["Timestamp"].notna().sum())

print("\nNumber of missing timestamps:")
print(df["Timestamp"].isna().sum())


# ============================================================
# 5. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values("Timestamp").reset_index(drop=True)


# ============================================================
# 6. CHECK DUPLICATE RECORDS
# ============================================================

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 7. CHECK TIMESTAMP VALUES
# ============================================================

print("\nFirst 20 timestamps:")
print(df["Timestamp"].head(20))

print("\nUnique timestamps:")
print(df["Timestamp"].drop_duplicates().sort_values().head(30))

print("\nNumber of unique timestamps:")
print(df["Timestamp"].nunique())

# ============================================================
# CHECK 15-MINUTE INTERVALS
# ============================================================

time_values = (
    df["Timestamp"]
    .dropna()
    .sort_values()
    .drop_duplicates()
)

time_diff = time_values.diff()

print("\nTime intervals:")
print(time_diff.value_counts().head(10))

print("\nUnique time intervals:")
print(time_diff.dropna().unique())

# ============================================================
# 8. CHECK FOR MISSING 15-MINUTE INTERVALS
# ============================================================

# Get valid timestamps
timestamps = (
    df["Timestamp"]
    .dropna()
    .sort_values()
    .drop_duplicates()
)

# Calculate gap between consecutive timestamps
gaps = timestamps.diff()

# Find gaps greater than 15 minutes
missing_intervals = gaps[gaps > pd.Timedelta(minutes=15)]

print("\nNumber of gaps greater than 15 minutes:")
print(len(missing_intervals))

print("\nLargest gaps:")
print(missing_intervals.sort_values(ascending=False).head(10))

# ============================================================
# 8. CREATE TIME-BASED FEATURES
# ============================================================

df["Hour"] = df["Timestamp"].dt.hour

df["Minute"] = df["Timestamp"].dt.minute

df["Time_in_Minutes"] = (
    df["Hour"] * 60 +
    df["Minute"]
)


# ============================================================
# 9. CREATE LAG FEATURES
# ============================================================

df["Lag_1"] = df["Sales Count"].shift(1)

df["Lag_2"] = df["Sales Count"].shift(2)

df["Lag_4"] = df["Sales Count"].shift(4)

df["Lag_8"] = df["Sales Count"].shift(8)


# ============================================================
# 10. CREATE ROLLING FEATURES
# ============================================================

df["Rolling_Mean_4"] = (
    df["Sales Count"]
    .shift(1)
    .rolling(window=4)
    .mean()
)

df["Rolling_Std_4"] = (
    df["Sales Count"]
    .shift(1)
    .rolling(window=4)
    .std()
)

df["Rolling_Max_4"] = (
    df["Sales Count"]
    .shift(1)
    .rolling(window=4)
    .max()
)


# ============================================================
# 11. DISPLAY PREPARED DATA
# ============================================================

print("\nPrepared dataset:")
print(df.head(10))

print("\nPrepared shape:")
print(df.shape)

print("\nPrepared columns:")
print(df.columns.tolist())


# ============================================================
# 12. SAVE PREPARED DATA
# ============================================================

df.to_csv(
    "ferry_time_series_prepared.csv",
    index=False
)

print("\nPrepared dataset saved successfully!")