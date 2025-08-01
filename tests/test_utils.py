"""Tests for utils.py."""

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prpl_utils.utils import (
    _DISABLED_cached_property_until_field_change,
    draw_dag,
    get_signed_angle_distance,
    timer,
    wrap_angle,
)


def test_cached_property_until_field_change() -> None:
    """Tests for cached_property_until_field_change()."""
    # pylint: disable=comparison-with-callable

    num_times_invoked = 0

    @dataclass
    class A:
        """A test class."""

        x: int
        y: int

        @_DISABLED_cached_property_until_field_change
        def xy(self) -> int:
            """Returns x * y."""
            nonlocal num_times_invoked
            num_times_invoked += 1
            return self.x * self.y

    a = A(x=3, y=4)
    assert a.xy == 12
    assert num_times_invoked == 1
    assert a.xy == 12
    assert num_times_invoked == 1
    a.x = 5
    assert a.xy == 20
    assert num_times_invoked == 2
    assert a.xy == 20
    assert num_times_invoked == 2
    a.y = -2
    assert a.xy == -10
    assert num_times_invoked == 3
    assert a.xy == -10
    assert num_times_invoked == 3
    a.x = 3
    a.y = 4
    assert a.xy == 12
    assert num_times_invoked == 4
    assert a.xy == 12
    assert num_times_invoked == 4


def test_wrap_angle():
    """Tests for wrap_angle()."""
    assert np.isclose(wrap_angle(0.0), 0.0)
    assert np.isclose(wrap_angle(np.pi / 2), np.pi / 2)
    assert np.isclose(wrap_angle(-np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(np.pi), np.pi)
    assert np.isclose(wrap_angle(-np.pi), -np.pi)
    assert np.isclose(wrap_angle(2 * np.pi), 0.0)
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(5 * np.pi / 2), np.pi / 2)


def test_get_signed_angle_distance():
    """Tests for get_signed_angle_distance()."""
    assert np.isclose(get_signed_angle_distance(0, 0), 0)
    assert np.isclose(get_signed_angle_distance(np.pi / 2, 0), np.pi / 2)
    assert np.isclose(get_signed_angle_distance(0, np.pi / 2), -np.pi / 2)
    assert np.isclose(get_signed_angle_distance(0, -np.pi / 2), np.pi / 2)
    assert np.isclose(get_signed_angle_distance(-np.pi / 2, 0), -np.pi / 2)


def test_draw_dag():
    """Tests for draw_dag()."""
    filepath = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name)
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]
    draw_dag(edges, filepath)
    assert filepath.exists()
    os.remove(filepath)


def test_timer():
    """Tests for the timer contextmanager."""
    with timer() as result:
        time.sleep(0.1)
    assert result["time"] > 0.1
