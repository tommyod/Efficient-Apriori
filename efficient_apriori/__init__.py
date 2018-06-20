#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Apriori algorithm.
"""

__version__ = '0.4.4'

from efficient_apriori.apriori import apriori

def run_tests():
    """
    Run all tests.
    """
    import pytest
    import os
    base, _ = os.path.split(__file__)
    pytest.main(args=[base, '--doctest-modules'])
