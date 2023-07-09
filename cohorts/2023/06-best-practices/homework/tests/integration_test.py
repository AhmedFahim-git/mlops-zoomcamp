from datetime import datetime
import pandas as pd
import os
import pytest


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


categorical = ["PULocationID", "DOLocationID"]
year = 2022
month = 1
if os.getenv("S3_ENDPOINT_URL"):
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
else:
    S3_ENDPOINT_URL = "http://localhost:4566"

options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}


def get_input_path(year, month):
    default_input_pattern = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = "s3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


input_file = get_input_path(year, month)
output_file = get_output_path(year, month)


@pytest.fixture
def make_test_df() -> pd.DataFrame:
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    df_input = pd.DataFrame(data, columns=columns)

    df_input.to_parquet(
        input_file,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=options,
    )

    os.system("python batch.py 2022 1")

    return pd.read_parquet(output_file, storage_options=options)


@pytest.fixture
def make_expected_df() -> pd.DataFrame:
    data = [
        ("2022/01_0", 24.7818),
        ("2022/01_1", 0.6175),
        ("2022/01_2", 6.1081),
    ]

    columns = ["ride_id", "predicted_duration"]
    return pd.DataFrame(data, columns=columns)


def test_integration(make_test_df, make_expected_df):
    pd.testing.assert_frame_equal(
        make_test_df, make_expected_df, check_exact=False, atol=0.0001
    )
