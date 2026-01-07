# to run this file you need to have the following files in the same directory:
# - test_data/bikes.csv
# install igann with:
# pip install git+https://github.com/MathiasKraus/igann.git@GAM_wrapper
# install i2dgraph with:
# pip install i2dgraph


# %%
# import libs
import igann
import i2dgraph


import json

# import standard libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from pprint import pprint as pp  # pretty print
import os

# function to load example dataset
from sklearn.datasets import fetch_california_housing

# Load the dataset
df = pd.read_csv("test_data/bike_sven.csv")


# remap numeric features
def scale_values(values, new_min, new_max):
    if isinstance(values, (pd.Series, pd.DataFrame)):
        old_min, old_max = values.min(), values.max()
        return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    arr = np.array(values)
    old_min, old_max = arr.min(), arr.max()
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


df["Time of Day"] = df["hr"]
df["Windspeed"] = scale_values(df["windspeed"], 0, 67)
df["Temperature"] = scale_values(df["temp"], -8, 39)
df["Perceived Temperature"] = scale_values(df["atemp"], -16, 50)
df["Humidity"] = scale_values(df["hum"], 0, 100)


# remap cat features for correct categorys
# Rename season values
df["Season"] = df["season"].replace({1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"})
df["Weathersituation"] = df["weathersit"].replace(
    # {1: "Clear", 2: "Mist", 3: "Light Rain", 4: "Heavy Rain"}
    {
        1: "Clear",
        2: "Cloudy",
        3: "Light Rain",
        4: "Heavy Rain",
    }
)


# Create Day Type variable based on Working Day, Weekend, Holiday
df["Type of Day"] = np.where(
    (df["workingday"] == 1) & (df["holiday"] == 0),
    "Working Day",
    np.where((df["workingday"] == 0) & (df["holiday"] == 0), "Weekend", "Holiday"),
)


df.dropna(subset=["cnt"], inplace=True)

print(df.info())

# %%
# set correct nan
df.replace("-", np.nan, inplace=True)

# drop examples with nans
df.dropna(inplace=True)

# set X and y
y = pd.DataFrame(df["cnt"])


# drop old and not needed columns
feature_to_drop = [
    # we also drop the columns with the old names
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "cnt",
    "instant",
    "workingday",
    "casual",
    "registered",
    "weekday",
    # also remove some renamed
    # "Temperature",
    # "Weathersituation",
    "Perceived Temperature",
    "Season",
    # "Time of Day",
    # "Type of Day",
    # "Humidity"
    # "Windspeed"
]

X = df.drop(columns=feature_to_drop, inplace=False)
print(X.shape)


print(X.describe())
print(X.info())
# %%
# very normal preprocessing
from feature_engine.outliers import Winsorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# define feature types for preprocessing (also make sure to use the correct type in the datagframe)
# define cat

# this dataset has not cats
cat_features = [
    "Weathersituation",
    # "Perceived Temperature",
    # "Season",
    "Time of Day",
    "Type of Day",
    # "Humidity"
    # "Windspeed"
]

# define numeric features
num_features = [feature for feature in X.columns if feature not in cat_features]


# create transformer for num features
num_Transformer = Pipeline(
    [
        (
            "num_imputer",
            SimpleImputer(strategy="mean"),
        ),
        # (
        #     "winsorizer",
        #     Winsorizer(
        #         capping_method="gaussian",  # or "quantiles"
        #         tail="both",  # "left", "right", or "both"
        #         fold=4,  # stdevs away if "gaussian", or quantile distance if "quantiles"
        #     ),
        # ),
    ]
)

# create transformer for cat features
cat_Transformer = Pipeline(
    [
        # no one-hot-encoding is use here (igann does this by it self)
        (
            "cat_imputer",
            SimpleImputer(strategy="most_frequent"),
        )
    ]
)

# wrap it in CloumnTransformer
column_Transformer = ColumnTransformer(
    transformers=[
        ("num", num_Transformer, num_features),
        ("cat", cat_Transformer, cat_features),
    ],
    verbose_feature_names_out=False,
).set_output(transform="pandas")

# transform X
X = column_Transformer.fit_transform(X)
X = X.astype(
    {
        # "yr": "object",
        # "mnth": "object",
        "Time of Day": "object",
        "Type of Day": "object",
        "Weathersituation": "object",
    }
)

print(X.info())
X.describe()

y_scaler = StandardScaler()
#
y_unscaled = y.copy()
# y = y_scaler.fit_transform(y)

# %%
# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

_, X_train_reduced, _, y_train_reduced = train_test_split(
    X_train, y_train, test_size=2000, random_state=42
)


# print(X_train_reduced.shape)
# %%
# train 2 models

# first normal igann
from igann import IGANN

igann = IGANN(
    task="regression",
    n_estimators=1000,
    verbose=0,
    scale_y=True,
)  # 1,

igann.fit(X_train, y_train)

# second igann interactive
from igann import IGANN_interactive

igann_i = IGANN_interactive(
    task="regression",
    n_estimators=1000,
    regressor_limit=100,  # set this to n_estimator otherwise wired things can happen
    verbose=1,  # 1,
    GAM_detail=100,  # number of points used to save and represent the shapefunction
    scale_y=True,
)
igann_i.fit(X_train, y_train)

igann.plot_single(show_n=8)

igann_i.interact()

# %%
from sklearn.metrics import (
    roc_auc_score,
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
)


def test_model(model, X, y):
    y_pred = model.predict(X)
    rsme = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return rsme, mae


def test_model_cliped(model, X, y):
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, 1000)
    rsme = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return rsme, mae


y_pred_igann = igann.predict(X_test)
y_pred_igann_i = igann_i.predict(X_test)

y_pred_igann_cliped = np.clip(y_pred_igann, 0, 1000)


plt.plot(y_test, y_pred_igann, "o", label="igann")
plt.plot(y_test, y_pred_igann_i, "o", label="igann_i")
plt.plot(y_test, y_pred_igann_cliped, "o", label="igann_cliped")
plt.plot(y_test, y_test, "-", label="true")

plt.legend()
plt.show()

print(f"igann: rsme, mae = {test_model(igann, X_test, y_test)}")
print(f"igann_interactive: rsme, mae = {test_model(igann_i, X_test, y_test)}")

print(f"igann_cliped: rsme, mae = {test_model_cliped(igann, X_test, y_test)}")
print(
    f"igann_interactive_cliped: rsme, mae = {test_model_cliped(igann_i, X_test, y_test)}"
)

# y_test_unscaled = y_scaler.inverse_transform(y_test)
# y_pred = igann.predict(X_test)
# y_pred_unscaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1))

# print(f"rsme: {root_mean_squared_error(y_test_unscaled, y_pred_unscaled)}")
# %%

igann.get_shape_functions_as_dict()


def reorder_categorical(feature_subdict, new_order):
    """
    Reorders the classes (and associated values) of a single categorical feature.

    Parameters
    ----------
    feature_subdict : dict
        A dictionary containing keys:
          - 'x': list of class labels (strings)
          - 'y': list or array of the same length as 'x'
          - 'hist': dict with:
              * 'classes': list of class labels (same as 'x')
              * 'counts': list of counts for each class
    new_order : list of str
        The new desired order of the classes. Must be a permutation of the old classes.

    Returns
    -------
    dict
        The same dictionary, but with 'x', 'y', and 'hist' data reordered
        according to `new_order`.
    """
    old_classes = feature_subdict["x"]

    # Create a mapping from old index to new index
    # For each class in `new_order`, find its index in the old list
    index_map = [old_classes.index(cls) for cls in new_order]

    # Reorder the 'x' classes
    feature_subdict["x"] = [old_classes[i] for i in index_map]

    # Reorder the 'y' values to match the new class order
    old_y = feature_subdict["y"]
    feature_subdict["y"] = [old_y[i] for i in index_map]

    # Reorder the hist classes
    old_hist_classes = feature_subdict["hist"]["classes"]
    old_hist_counts = feature_subdict["hist"]["counts"]

    # Reorder 'hist'['classes']
    feature_subdict["hist"]["classes"] = [old_hist_classes[i] for i in index_map]
    # Reorder 'hist'['counts']
    feature_subdict["hist"]["counts"] = [old_hist_counts[i] for i in index_map]

    return feature_subdict


cat_feature_map = {
    "Weathersituation": [
        "Clear",
        "Cloudy",
        "Light Rain",
        "Heavy Rain",
    ],
    # "Season": [
    #     "Fall",
    #     "Winter",
    #     "Spring",
    #     "Summer",
    # ],
    "Time of Day": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
    ],
    "Type of Day": ["Working Day", "Weekend", "Holiday"],
}


def reorder_cat_classes(feature_dict, cat_feature_map):
    # helper function for one feature
    def reorder_categorical(feature_subdict, new_order):
        old_classes = feature_subdict["x"]

        # Create a mapping from old index to new index
        # For each class in `new_order`, find its index in the old list
        index_map = [old_classes.index(cls) for cls in new_order]

        # Reorder the 'x' classes
        feature_subdict["x"] = [old_classes[i] for i in index_map]

        # Reorder the 'y' values to match the new class order
        old_y = feature_subdict["y"]
        feature_subdict["y"] = [old_y[i] for i in index_map]

        # Reorder the hist classes
        old_hist_classes = feature_subdict["hist"]["classes"]
        old_hist_counts = feature_subdict["hist"]["counts"]

        # Reorder 'hist'['classes']
        feature_subdict["hist"]["classes"] = [old_hist_classes[i] for i in index_map]
        # Reorder 'hist'['counts']
        feature_subdict["hist"]["counts"] = [old_hist_counts[i] for i in index_map]

        return feature_subdict

    # loop trough all features that needs reorderung
    for feature in cat_feature_map.keys():
        sub_dict = feature_dict[feature]
        sub_dict = reorder_categorical(sub_dict, cat_feature_map[feature])
        feature_dict[feature] = sub_dict


# %%
def plot_features(features_dict, font_size=14):
    """
    Plots each feature in `features_dict`:
      - If 'datatype' == 'numerical', plots a line chart of x vs. y.
      - If 'datatype' == 'categorical', plots a bar chart of x vs. y.

    Parameters
    ----------
    features_dict : dict
        Dictionary of features, where each key is a feature name, and each value
        is another dictionary that must contain at least:
          {
            'name': str,
            'datatype': 'numerical' or 'categorical',
            'x': list or array,
            'y': list or array,
            ...
          }
    """
    # Create one figure per feature (you could also do subplots if you like).
    for feature_name, feature_data in features_dict.items():
        plt.figure(figsize=(6, 4))

        # Extract what we need
        x = feature_data["x"]
        y = feature_data["y"]
        datatype = feature_data["datatype"]

        if datatype == "numerical":
            # Line plot for numerical features
            plt.plot(x, y, linestyle="-", color="black")
            plt.xlabel(feature_name, fontsize=font_size)
            plt.ylabel("Number of Bikes", fontsize=font_size)
            plt.title(f"{feature_name}", fontsize=font_size + 2)

        elif datatype == "categorical":

            # If y == 0, change it to 1, otherwise leave it as is.
            y_mean = np.min(y)
            y = [val if val != 0 else 0.01 * y_mean for val in y]

            # Bar plot for categorical features
            plt.bar(x, y, fill="black", color="black")
            plt.xlabel(feature_name, fontsize=font_size)
            plt.ylabel("Number of Bikes", fontsize=font_size)
            plt.title(f"{feature_name}", fontsize=font_size + 2)
            # If x-axis labels are too long, you may want to rotate them:
            plt.xticks(
                rotation=0,
                # ha="right",
            )

        else:
            print(f"Skipping {feature_name}: Unrecognized datatype '{datatype}'")
            plt.close()
            continue

        # 1. Add a dashed horizontal line at y=0
        plt.axhline(y=0, color="black", linestyle="--")

        # 2. Set the fixed y-axis range from -10 to 10
        plt.ylim(min(-10, min(y) - 10), max(10, max(y) + 10))

        plt.tight_layout()
        plt.savefig(fname=f"./igann_i_plots/{feature_name}.svg", format="svg")
        plt.savefig(fname=f"./igann_i_plots/{feature_name}")

        plt.show()


print(np.mean(y))

# %%
feature_dict = igann.get_shape_functions_as_dict()
reorder_cat_classes(feature_dict, cat_feature_map)
plot_features(feature_dict)
# %%

# 1. Build a df_test that has the same rows as X_test
df_test = df.loc[X_test.index].copy()

# 2. Randomly pick 20 samples from the test set
df_20 = df_test.sample(20, random_state=42)

# 3. Get corresponding X rows so we can make predictions
X_20 = X_test.loc[df_20.index]

# 4. Predict with igann
df_20["model_pred"] = igann.predict(X_20)

# 5. Add the true target
df_20["y_true"] = df_20["cnt"]  # or whatever your final target column is

# 6. Create a ‘number’ column from 1..20
df_20.insert(0, "number", range(1, len(df_20) + 1))

# 7. Rename columns to match desired output
#    "Weather Condition" -> from "Weathersituation"
df_20.rename(
    columns={
        "Weathersituation": "Weather Condition",
    },
    inplace=True,
)

# 8. Reorder columns exactly as requested:
#    [number, Season, Weather Condition, Temperature, Day Type, Humidity, Wind Speed, Time of Day, model_pred, y_true]
ordered_cols = [
    "number",
    "Weather Condition",  # was "Weathersituation"
    "Temperature",
    "Type of Day",
    "Humidity",
    "Windspeed",  # "Wind Speed"
    "Time of Day",
    "model_pred",
    "y_true",
]


df_20 = df_20[ordered_cols]


def format_time_str(hour):
    from datetime import datetime

    # Step 1: Create a 24-hour string (zero-padded)
    hour_24_str = f"{hour:02d}:00"

    # Step 2: Convert to 12-hour format
    #   e.g., hour=22 -> "10 PM"
    #   We'll lowercase the 'AM/PM' part to match your example
    hour_12_str = (
        datetime.strptime(str(hour), "%H")
        .strftime("%I %p")
        .lstrip("0")  # Remove the leading '0' if present
        .upper()
    )
    # Build the final string "HH:00 (HH pm/am)"
    return f"{hour_24_str} ({hour_12_str})"


df_20["Time of Day"] = df_20["Time of Day"].apply(format_time_str)

# df_20["error"] = df_20["model_pred"] - df_20["y_true"]

# df_20["absolute_error"] = abs(df_20["error"])
# print(np.mean(df_20["absolute_error"]))

# 9. Save to CSV without headers, no index
df_20.to_csv("sample_20_sosci.csv", index=False, header=False)

# %%
