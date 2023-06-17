import pickle
import pandas as pd
import os

import argparse

parser = argparse.ArgumentParser(description="This should take in the year and month")
parser.add_argument(
    "-y",
    "--year",
    type=int,
    default=2022,
    choices=[2021, 2022, 2023],
    dest="year",
    help="This should be the year for which we want to use the yellow tripdata",
)
parser.add_argument(
    "-m",
    "--month",
    type=int,
    default=2,
    choices=range(1, 13),
    dest="month",
    help="This should be the year for month we want to use the yellow tripdata",
)

args = parser.parse_args()
year = args.year
month = args.month


with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
output_file = f"outputs/yellow_tripdata_{year:04d}-{month:02d}.parquet"


categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


df = read_data(input_file)
df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")


dicts = df[categorical].to_dict(orient="records")
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(y_pred.mean())

df_result = df[["ride_id"]]
df_result["predicted_duration"] = y_pred


if not os.path.exists("outputs"):
    os.mkdir("outputs")

df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
