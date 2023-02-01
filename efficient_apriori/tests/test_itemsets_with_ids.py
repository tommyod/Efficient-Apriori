#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

import pytest
import itertools
import random

from efficient_apriori.itemsets import itemsets_from_transactions, ItemsetCount


def generate_transactions(num_transactions, unique_items, items_row=(1, 100), seed=None):
    """
    Generate synthetic transactions.
    """
    if seed:
        random.seed(seed)
    else:
        random.seed()

    items = list(range(unique_items))

    for _ in range(num_transactions):
        items_this_row = random.randint(*items_row)
        yield random.sample(items, k=min(unique_items, items_this_row))


def itemsets_from_transactions_naive(transactions, min_support):
    """
    Naive algorithm used for testing only.
    """

    # Get the unique items from every transaction
    unique_items = {k for ts in transactions for k in ts}
    num_transactions = len(transactions)

    # Create an output dictionary
    L = dict()

    # For every possible combination length
    for k in range(1, len(unique_items) + 1):
        # For every possible combination
        for combination in itertools.combinations(unique_items, k):
            # Naively count how many transactions contain the combination
            counts = ItemsetCount()
            for i, t in enumerate(transactions):
                if set.issubset(set(combination), set(t)):
                    counts.itemset_count += 1
                    counts.members.add(i)

            # If the count exceeds the minimum support, add it
            if (counts.itemset_count / num_transactions) >= min_support:
                try:
                    L[k][tuple(sorted(list(combination)))] = counts
                except KeyError:
                    L[k] = dict()
                    L[k][tuple(sorted(list(combination)))] = counts

        try:
            L[k] = {k: v for (k, v) in sorted(L[k].items())}
            if L[k] == {}:
                del L[k]
                return L, num_transactions
        except KeyError:
            return L, num_transactions

    return L, num_transactions


input_data = [
    (
        list(
            generate_transactions(
                random.randint(5, 25),
                random.randint(1, 8),
                (1, random.randint(2, 8)),
            )
        ),
        random.randint(1, 4) / 10,
    )
    for i in range(500)
]


@pytest.mark.parametrize("transactions, min_support", input_data)
def test_itemsets_from_transactions_stochastic(transactions, min_support):
    """
    Test random inputs.
    """
    result, _ = itemsets_from_transactions(list(transactions), min_support, output_transaction_ids=True)
    naive_result, _ = itemsets_from_transactions_naive(list(transactions), min_support)

    for key in set.union(set(result.keys()), set(naive_result.keys())):
        assert result[key] == naive_result[key]


@pytest.mark.parametrize("transactions, min_support", input_data)
def test_itemsets_max_length(transactions, min_support):
    """
    The that nothing larger than max length is returned.
    """
    max_len = random.randint(1, 5)
    result, _ = itemsets_from_transactions(
        list(transactions),
        min_support,
        max_length=max_len,
        output_transaction_ids=True,
    )

    assert all(list(k <= max_len for k in result.keys()))
    for length, itemsets in result.items():
        for itemset_count in itemsets.values():
            assert all(isinstance(i, int) for i in itemset_count.members)


if __name__ == "__main__":
    pytest.main(args=[".", "--doctest-modules", "-v"])
