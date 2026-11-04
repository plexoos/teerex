"""Shared setup for headless plotting tests."""

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
