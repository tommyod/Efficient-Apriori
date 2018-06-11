#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 20:03:20 2018

@author: tommy
"""

import pytest
import collections
import itertools

def itemsets_from_transactions_naive(transactions, min_support):
    """
    Naive algorithm for testing.
    """
    unique_items = set(k for t in transactions for k in t)

    L = collections.defaultdict(list)
    for k in range(1, len(unique_items) + 1):
        for combination in itertools.combinations(unique_items, k):
            counts = 0
            for transaction in transactions:
                if set.issubset(set(combination), set(transaction)):
                    counts += 1     
            if counts >= min_support:
                L[k].append(tuple(sorted(list(combination)))) 
    return L

def test_stochastic():
    pass


if __name__ == '__main__':
    pytest.main(args=[__file__, '--doctest-modules', '-v'])