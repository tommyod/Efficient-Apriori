#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-level implementations of the apriori algorithm.
"""

import typing
from efficient_apriori.itemsets import itemsets_from_transactions, ItemsetCount
from efficient_apriori.rules import generate_rules_apriori


def apriori(
    transactions: typing.Union[typing.List[tuple], typing.Callable],
    min_support: float = 0.5,
    min_confidence: float = 0.5,
    max_length: int = 8,
    verbosity: int = 0,
    output_transaction_ids: bool = False,
):
    """
    The classic apriori algorithm as described in 1994 by Agrawal et al.

    The Apriori algorithm works in two phases. Phase 1 iterates over the
    transactions several times to build up itemsets of the desired support
    level. Phase 2 builds association rules of the desired confidence given the
    itemsets found in Phase 1. Both of these phases may be correctly
    implemented by exhausting the search space, i.e. generating every possible
    itemset and checking it's support. The Apriori prunes the search space
    efficiently by deciding apriori if an itemset possibly has the desired
    support, before iterating over the entire dataset and checking.

    Parameters
    ----------
    transactions : list of tuples, list of itemsets.TransactionWithId,
        or a callable returning a generator. Use TransactionWithId's when
        the transactions have ids which should appear in the outputs.
        The transactions may be either a list of tuples, where the tuples must
        contain hashable items. Alternatively, a callable returning a generator
        may be passed. A generator is not sufficient, since the algorithm will
        exhaust it, and it needs to iterate over it several times. Therefore,
        a callable returning a generator must be passed.
    min_support : float
        The minimum support of the rules returned. The support is frequency of
        which the items in the rule appear together in the data set.
    min_confidence : float
        The minimum confidence of the rules returned. Given a rule X -> Y, the
        confidence is the probability of Y, given X, i.e. P(Y|X) = conf(X -> Y)
    max_length : int
        The maximum length of the itemsets and the rules.
    verbosity : int
        The level of detail printing when the algorithm runs. Either 0, 1 or 2.
    output_transaction_ids : bool
        If set to true, the output contains the ids of transactions that
        contain a frequent itemset. The ids are the enumeration of the
        transactions in the sequence they appear.
    Examples
    --------
    >>> transactions = [('a', 'b', 'c'), ('a', 'b', 'd'), ('f', 'b', 'g')]
    >>> itemsets, rules = apriori(transactions, min_confidence=1)
    >>> rules
    [{a} -> {b}]
    """

    itemsets, num_trans = itemsets_from_transactions(
        transactions,
        min_support,
        max_length,
        verbosity,
        output_transaction_ids,
    )

    if itemsets and isinstance(next(iter(itemsets[1].values())), ItemsetCount):
        itemsets_for_rules = _convert_to_counts(itemsets)
    else:
        itemsets_for_rules = itemsets

    rules = generate_rules_apriori(itemsets_for_rules, min_confidence, num_trans, verbosity)
    return itemsets, list(rules)


def _convert_to_counts(itemsets):
    itemsets_counts = {}
    for size, sets in itemsets.items():
        itemsets_counts[size] = {i: c.itemset_count for i, c in sets.items()}
    return itemsets_counts


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "-v"])
