"""Bootstrapped Davidson-MacKinnon J-test.

Provides a fast Bootstrapped J-test implementation for comparing non-nested linear and non-linear models
"""

from .bdmjtest import jtest_np, jtest_sm, jtest_scipy, JtestResults
__all__ = ["jtest_np",
           "jtest_sm",
           "jtest_scipy",
           "JtestResults"]
