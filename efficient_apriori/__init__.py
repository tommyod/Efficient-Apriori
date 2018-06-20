#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Apriori algorithm.
"""

__version__ = '0.4.2'

from efficient_apriori.apriori import apriori

def run_tests():
    """
    Run all tests.
    """
    import pytest
    pytest.main(args=[__file__, '--doctest-modules', '-v'])
