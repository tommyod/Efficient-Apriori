#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-level implementations of the apriori algorithm.
"""

from efficient_apriori.itemsets import itemsets_from_transactions
from efficient_apriori.rules import generate_rules_simple

def apriori(transactions:list, min_support:float=0.5, min_confidence:float=0.5):
    """
    The classic apriori algorithm.
    
    Examples
    --------
    >>> transactions = [('a', 'b', 'c'), ('a', 'b', 'd'), ('f', 'b', 'g')]
    >>> itemsets, rules = apriori(transactions, min_confidence=1)
    >>> rules
    [{a} -> {b}]
    """
    itemsets = itemsets_from_transactions(transactions, min_support)
    rules = list(generate_rules_simple(itemsets, min_confidence))
    return itemsets, rules

          
if __name__ == '__main__':
    import pytest
    pytest.main(args=['.', '--doctest-modules', '-v'])