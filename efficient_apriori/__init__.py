#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Apriori algorithm.
"""

__version__ = "0.4.5"

import sys
from efficient_apriori.apriori import apriori
from efficient_apriori.itemsets import itemsets_from_transactions
from efficient_apriori.rules import Rule, generate_rules_apriori


apriori = apriori
itemsets_from_transactions = itemsets_from_transactions
Rule = Rule
generate_rules_apriori = generate_rules_apriori


def run_tests():
    """
    Run all tests.
    """
    import pytest
    import os

    base, _ = os.path.split(__file__)
    pytest.main(args=[base, "--doctest-modules"])


if (sys.version_info[0] < 3) or (sys.version_info[1] < 6):
    raise Exception("The `efficient_apriori` package only works for Python 3.6+.")
