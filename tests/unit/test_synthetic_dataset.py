"""Tests for the expanded synthetic manufacturing dataset."""

import pandas as pd

from scripts.generate_sample_data import generate_events


def test_generator_creates_600_events() -> None:
    dataframe = generate_events()

    assert len(dataframe) == 600


def test_event_ids_are_unique() -> None:
    dataframe = generate_events()

    assert dataframe["event_id"].is_unique


def test_all_production_lines_are_represented() -> None:
    dataframe = generate_events()

    assert set(dataframe["production_line"]) == {
        "LINE-01",
        "LINE-02",
        "LINE-03",
    }


def test_all_measurement_types_are_represented() -> None:
    dataframe = generate_events()

    assert set(dataframe["event_type"]) == {
        "temperature",
        "production_output",
        "defect_rate",
    }


def test_ground_truth_labels_are_boolean() -> None:
    dataframe = generate_events()

    assert pd.api.types.is_bool_dtype(
        dataframe["known_anomaly"]
    )

    assert dataframe["known_anomaly"].any()
    assert (~dataframe["known_anomaly"]).any()


def test_generator_is_reproducible() -> None:
    first = generate_events()
    second = generate_events()

    pd.testing.assert_frame_equal(first, second)