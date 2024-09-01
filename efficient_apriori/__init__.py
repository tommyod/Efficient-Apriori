#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Apriori algorithm.
"""

import importlib.metadata
from efficient_apriori.apriori import apriori
from efficient_apriori.itemsets import itemsets_from_transactions
from efficient_apriori.rules import Rule, generate_rules_apriori

# We use semantic versioning
# See https://semver.org/
__version__ = importlib.metadata.version("efficient_apriori")

__all__ = ["apriori", "itemsets_from_transactions", "Rule", "generate_rules_apriori"]


def run_tests():
    """
    Run all tests.
    """
    import pytest
    import os

    base, _ = os.path.split(__file__)
    pytest.main(args=[base, "--doctest-modules"])
