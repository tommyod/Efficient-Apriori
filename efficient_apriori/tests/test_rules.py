#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

import pytest
import itertools
import random

from efficient_apriori.itemsets import itemsets_from_transactions
from efficient_apriori.rules import (
    Rule,
    generate_rules_simple,
    generate_rules_apriori,
)
from efficient_apriori.tests.test_itemsets import generate_transactions


def generate_rules_naively(itemsets, min_confidence, num_transactions):
    """
    Generate association rules naively, for testing purposes.
    """

    def proper_subsets(itemset: set):
        """
        Yield every proper subset of a set.
        """
        size = range(1, len(itemset))
        arg = [itertools.combinations(itemset, i) for i in size]
        yield from itertools.chain(*arg)

    def count(itemset):
        """
        Helper function to find the count of an itemset in the transactions.
        """
        return itemsets[len(itemset)][itemset]

    # For every itemset size greater than 1, yield every itemset of that size
    itemsets_gen = (iset for size in filter(lambda x: x > 1, itemsets.keys()) for iset in itemsets[size].keys())

    for itemset in itemsets_gen:
        count_full = count(itemset)

        # For every subset, get the difference, create a rule and check
        for lhs in proper_subsets(itemset):
            rhs = set(itemset).difference(set(lhs))
            rhs = tuple(sorted(list(rhs)))
            rule = Rule(lhs, rhs, count_full, count(lhs), count(rhs), num_transactions)

            # If the confidence of the rule is high enough, yield it
            if rule.confidence >= min_confidence:
                yield rule


def test_generate_rules_apriori_large():
    """
    Test with lots of data.
    This test will fail if the second argument to `_ap_genrules` is not
    validated as non-empty before the recursive function call. We must have
    if H_m_copy:
        yield from _ap_genrules
    for this test to pass.
    """

    transactions = generate_transactions(num_transactions=100, unique_items=30, items_row=(1, 20), seed=123)

    itemsets, num_transactions = itemsets_from_transactions(transactions, 0.1)

    min_conf = 0.3
    rules_apri = generate_rules_apriori(itemsets, min_conf, num_transactions)
    rules_naive = generate_rules_naively(itemsets, min_conf, num_transactions)
    rules_apri = list(rules_apri)
    rules_naive = list(rules_naive)

    # Test equal length, since no duplicates should be returned by apriori
    assert len(rules_apri) == len(rules_naive)

    # Test equal results
    assert set(rules_apri) == set(rules_naive)


input_data = [
    list(
        generate_transactions(
            num_transactions=random.randint(15, 25),
            unique_items=random.randint(1, 8),
            items_row=(1, random.randint(2, 6)),
        )
    )
    for i in range(10)
]


@pytest.mark.parametrize("transactions", input_data)
def test_generate_rules_simple_vs_naive(transactions):
    """
    Test the naive rule finder vs. the simple one from the paper.
    """

    itemsets, num_transactions = itemsets_from_transactions(transactions, 0.25)

    min_conf = 0.1
    rules_naive = generate_rules_naively(itemsets, min_conf, num_transactions)
    rules_simple = generate_rules_simple(itemsets, min_conf, num_transactions)
    assert set(rules_naive) == set(rules_simple)


@pytest.mark.parametrize("transactions", input_data)
def test_generate_rules_simple_vs_apriori(transactions):
    """
    Test the naive rule finder vs. the simple one from the paper.
    """

    itemsets, num_transactions = itemsets_from_transactions(transactions, 0.1)

    min_conf = 0.1
    rules_apri = generate_rules_apriori(itemsets, min_conf, num_transactions)
    rules_simple = generate_rules_simple(itemsets, min_conf, num_transactions)
    assert set(rules_apri) == set(rules_simple)


@pytest.mark.parametrize("transactions", input_data)
def test_generate_rules_naive_vs_apriori(transactions):
    """
    Test the naive rule finder vs. the simple one from the paper.
    """

    itemsets, num_transactions = itemsets_from_transactions(transactions, 0.15)

    min_conf = 0.3
    rules_apri = generate_rules_apriori(itemsets, min_conf, num_transactions)
    rules_naive = generate_rules_naively(itemsets, min_conf, num_transactions)
    rules_apri = list(rules_apri)
    rules_naive = list(rules_naive)

    # Test equal length, since no duplicates should be returned by apriori
    assert len(rules_apri) == len(rules_naive)

    # Test equal results
    assert set(rules_apri) == set(rules_naive)


def speeds():
    """
    Test the naive rule finder vs. the simple one from the paper.
    """
    import random

    random.seed(123456)
    transactions = generate_transactions(
        num_transactions=random.randint(250, 500),
        unique_items=random.randint(8, 9),
        items_row=(10, 50),
    )

    itemsets, num_transactions = itemsets_from_transactions(transactions, 0.1)
    import time

    min_conf = 0.5

    print(itemsets)

    st = time.perf_counter()
    rules_apri = generate_rules_apriori(itemsets, min_conf, num_transactions)
    rules_apri = list(rules_apri)
    time_formatted = round(time.perf_counter() - st, 40)
    print("Fast apriori ran in {} s".format(time_formatted))

    st = time.perf_counter()
    rules_simple = generate_rules_simple(itemsets, min_conf, num_transactions)
    rules_simple = list(rules_simple)
    time_formatted = round(time.perf_counter() - st, 40)
    print("Simple apriori ran in {} s".format(time_formatted))

    st = time.perf_counter()
    rules_naive = generate_rules_naively(itemsets, min_conf, num_transactions)
    rules_naive = list(rules_naive)
    time_formatted = round(time.perf_counter() - st, 40)
    print("Naive apriori ran in {} s".format(time_formatted))


if __name__ == "__main__":
    pytest.main(args=[".", "--doctest-modules", "-v"])
