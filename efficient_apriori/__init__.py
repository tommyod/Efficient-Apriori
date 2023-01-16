#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Apriori algorithm.
"""

# We use semantic versioning
# See https://semver.org/
__version__ = "2.0.2"

import sys
from efficient_apriori.apriori import apriori
from efficient_apriori.itemsets import itemsets_from_transactions
from efficient_apriori.rules import Rule, generate_rules_apriori

__all__ = ["apriori", "itemsets_from_transactions", "Rule", "generate_rules_apriori"]


def run_tests():
    """
    Run all tests.
    """
    import pytest
    import os

    base, _ = os.path.split(__file__)
    pytest.main(args=[base, "--doctest-modules"])


if (sys.version_info[0] < 3) or (sys.version_info[1] < 7):
    msg = "The `efficient_apriori` package only works for Python 3.7+."
    raise Exception(msg)
