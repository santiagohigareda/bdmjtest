"""Bootstrapped Davidson-MacKinnon J-test.

Provides a fast Bootstrapped J-test implementation for comparing non-nested linear and non-linear models
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bdmjtest")
except PackageNotFoundError:
    __version__ = "2.0.0.dev"

from .bdmjtest import jtest_np, jtest_sm, jtest_scipy, JtestResults
__all__ = ["jtest_np",
           "jtest_sm",
           "jtest_scipy",
           "JtestResults"]
