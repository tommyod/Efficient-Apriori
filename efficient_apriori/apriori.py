#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-level implementations of the apriori algorithm.
"""

import typing
from efficient_apriori.itemsets import itemsets_from_transactions
from efficient_apriori.rules import generate_rules_apriori


def apriori(
    transactions: typing.Iterable[typing.Union[set, tuple, list]],
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
    transactions : list of transactions (sets/tuples/lists). Each element in
        the transactions must be hashable.
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
        output_transaction_ids=True,
    )

    itemsets_raw = {
        length: {item: counter.itemset_count for (item, counter) in itemsets.items()}
        for (length, itemsets) in itemsets.items()
    }
    rules = generate_rules_apriori(itemsets_raw, min_confidence, num_trans, verbosity)

    if output_transaction_ids:
        return itemsets, list(rules)

    return itemsets_raw, list(rules)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "-v"])
