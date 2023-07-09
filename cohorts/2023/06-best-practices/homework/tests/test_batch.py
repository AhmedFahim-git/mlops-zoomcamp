from datetime import datetime
import pandas as pd
import pytest


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


categorical = ["PULocationID", "DOLocationID"]


def prepare_data(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype(int).astype(str)

    return df


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
    df = pd.DataFrame(data, columns=columns)
    return prepare_data(df, categorical)


@pytest.fixture
def make_expected_df() -> pd.DataFrame:
    data = [
        ("-1", "-1", dt(1, 2), dt(1, 10), 8.0),
        ("1", "-1", dt(1, 2), dt(1, 10), 8.0),
        ("1", "2", dt(2, 2), dt(2, 3), 1.0),
    ]
    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "duration",
    ]
    return pd.DataFrame(data, columns=columns)


def test_prepare_data(make_test_df, make_expected_df):
    pd.testing.assert_frame_equal(make_test_df, make_expected_df)
