#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to itemsets.
"""

import itertools
import collections
import numbers
import typing


def join_step(itemsets: typing.List[tuple]):
    """
    Join k length itemsets into k + 1 length itemsets.

    This algorithm assumes that the list of itemsets are sorted, and that the
    itemsets themselves are sorted tuples. Instead of always enumerating all
    n^2 combinations, the algorithm only has n^2 runtime for each block of
    itemsets with the first k - 1 items equal.

    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k, to be joined to k + 1 length
        itemsets.

    Examples
    --------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]
    >>> list(join_step(itemsets))
    [(1, 2, 3, 4), (1, 3, 4, 5)]
    """
    i = 0
    # Iterate over every itemset in the itemsets
    while i < len(itemsets):

        # The number of rows to skip in the while-loop, intially set to 1
        skip = 1

        # Get all but the last item in the itemset, and the last item
        *itemset_first, itemset_last = itemsets[i]

        # We now iterate over every itemset following this one, stopping
        # if the first k - 1 items are not equal. If we're at (1, 2, 3),
        # we'll consider (1, 2, 4) and (1, 2, 7), but not (1, 3, 1)

        # Keep a list of all last elements, i.e. tail elements, to perform
        # 2-combinations on later on
        tail_items = [itemset_last]
        tail_items_append = tail_items.append  # Micro-optimization

        # Iterate over ever itemset following this itemset
        for j in range(i + 1, len(itemsets)):

            # Get all but the last item in the itemset, and the last item
            *itemset_n_first, itemset_n_last = itemsets[j]

            # If it's the same, append and skip this itemset in while-loop
            if itemset_first == itemset_n_first:

                # Micro-optimization
                tail_items_append(itemset_n_last)
                skip += 1

            # If it's not the same, break out of the for-loop
            else:
                break

        # For every 2-combination in the tail items, yield a new candidate
        # itemset, which is sorted.
        itemset_first_tuple = tuple(itemset_first)
        for a, b in sorted(itertools.combinations(tail_items, 2)):
            yield itemset_first_tuple + (a,) + (b,)

        # Increment the while-loop counter
        i += skip


def prune_step(
    itemsets: typing.Iterable[tuple], possible_itemsets: typing.List[tuple]
):
    """
    Prune possible itemsets whose subsets are not in the list of itemsets.

    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k.
    possible_itemsets : list of itemsets
        A list of possible itemsets of length k + 1 to be pruned.

    Examples
    -------
    >>> itemsets = [('a', 'b', 'c'), ('a', 'b', 'd'),
    ...             ('b', 'c', 'd'), ('a', 'c', 'd')]
    >>> possible_itemsets = list(join_step(itemsets))
    >>> list(prune_step(itemsets, possible_itemsets))
    [('a', 'b', 'c', 'd')]
    """

    # For faster lookups
    itemsets = set(itemsets)

    # Go through every possible itemset
    for possible_itemset in possible_itemsets:

        # Remove 1 from the combination, same as k-1 combinations
        # The itemsets created by removing the last two items in the possible
        # itemsets must be part of the itemsets by definition,
        # due to the way the `join_step` function merges the sorted itemsets

        for i in range(len(possible_itemset) - 2):
            removed = possible_itemset[:i] + possible_itemset[i + 1 :]

            # If every k combination exists in the set of itemsets,
            # yield the possible itemset. If it does not exist, then it's
            # support cannot be large enough, since supp(A) >= supp(AB) for
            # all B, and if supp(S) is large enough, then supp(s) must be large
            # enough for every s which is a subset of S.
            # This is the downward-closure property of the support function.
            if removed not in itemsets:
                break

        # If we have not breaked yet
        else:
            yield possible_itemset


def apriori_gen(itemsets: typing.List[tuple]):
    """
    Compute all possible k + 1 length supersets from k length itemsets.

    This is done efficienly by using the downward-closure property of the
    support function, which states that if support(S) > k, then support(s) > k
    for every subset s of S.

    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k.

    Examples
    -------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]
    >>> possible_itemsets = list(join_step(itemsets))
    >>> list(prune_step(itemsets, possible_itemsets))
    [(1, 2, 3, 4)]
    """
    possible_extensions = join_step(itemsets)
    yield from prune_step(itemsets, possible_extensions)


def itemsets_from_transactions(
    transactions: typing.Union[typing.List[tuple], typing.Callable],
    min_support: float,
    max_length: int = 8,
    verbosity: int = 0,
):
    """
    Compute itemsets from transactions by building the itemsets bottom up and
    iterating over the transactions to compute the support repedately. This is
    the heart of the Apriori algorithm by Agrawal et al. in the 1994 paper.

    Parameters
    ----------
    transactions : either a list of itemsets (tuples with hashable entries),
                   or a function returning a generator
        A list of transactions. They can be of varying size. To pass through
        data without reading everything into memory at once, a callable
        returning a generator may also be passed.
    min_support : float
        The minimum support of the itemsets, i.e. the minimum frequency as a
        percentage.
    max_length : int
        The maximum length of the itemsets.
    verbosity : int
        The level of detail printing when the algorithm runs. Either 0, 1 or 2.

    Examples
    --------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> transactions = [(1, 3, 4), (2, 3, 5), (1, 2, 3, 5), (2, 5)]
    >>> itemsets, _ = itemsets_from_transactions(transactions, min_support=2/5)
    >>> itemsets[1] == {(1,): 2, (2,): 3, (3,): 3, (5,): 3}
    True
    >>> itemsets[2] == {(1, 3): 2, (2, 3): 2, (2, 5): 3, (3, 5): 2}
    True
    >>> itemsets[3] == {(2, 3, 5): 2}
    True
    """
    # STEP 0 - Sanitize user inputs
    # -----------------------------

    # If the transactions are iterable, convert it to sets for faster lookups
    if isinstance(transactions, collections.abc.Iterable):
        transaction_sets = [set(t) for t in transactions if len(t) > 0]

        def transactions():
            return transaction_sets

    # Assume the transactions is a callable, returning a generator
    elif callable(transactions):
        ret_value = transactions()
        if not isinstance(ret_value, collections.abc.Generator):
            msg = (
                "`transactions` must be an iterable or a callable "
                + "returning an iterable."
            )
            raise TypeError(msg)
    else:
        msg = (
            "`transactions` must be an iterable or a callable "
            + "returning an iterable."
        )
        raise TypeError(msg)

    if not (
        isinstance(min_support, numbers.Number) and (0 <= min_support <= 1)
    ):
        raise ValueError("`min_support` must be a number between 0 and 1.")

    # Keep a dictionary stating whether to consider the row, this will allow
    # row-pruning later on if no information was retrieved earlier from it
    use_transaction = collections.defaultdict(lambda: True)

    # STEP 1 - Generate all large itemsets of size 1
    # ----------------------------------------------
    if verbosity > 0:
        print("Generating itemsets.")
        print(" Counting itemsets of length 1.")

    counts = collections.defaultdict(int)
    num_transactions = 0
    for transaction in transactions():
        num_transactions += 1  # Increment counter for transactions
        for item in transaction:
            counts[item] += 1  # Increment counter for single-item itemsets

    large_itemsets = [
        (i, c)
        for (i, c) in counts.items()
        if (c / num_transactions) >= min_support
    ]

    if verbosity > 0:
        num_cand, num_itemsets = len(counts.items()), len(large_itemsets)
        print("  Found {} candidate itemsets of length 1.".format(num_cand))
        print("  Found {} large itemsets of length 1.".format(num_itemsets))
    if verbosity > 1:
        print("    {}".format(list((i,) for (i, c) in large_itemsets)))

    # If large itemsets were found, convert to dictionary
    if large_itemsets:
        large_itemsets = {1: {(i,): c for (i, c) in sorted(large_itemsets)}}

    # No large itemsets were found, return immediately
    else:
        return dict(), num_transactions

    # STEP 2 - Build up the size of the itemsets
    # ------------------------------------------

    # While there are itemsets of the previous size
    issubset = set.issubset  # Micro-optimization
    k = 2
    while large_itemsets[k - 1] and (max_length != 1):
        if verbosity > 0:
            print(" Counting itemsets of length {}.".format(k))

        # STEP 2a) - Build up candidate of larger itemsets

        # Retrieve the itemsets of the previous size, i.e. of size k - 1
        itemsets_list = list(large_itemsets[k - 1].keys())

        # Generate candidates of length k + 1 by joning, prune, and copy as set
        C_k = list(apriori_gen(itemsets_list))
        C_k_sets = [set(itemset) for itemset in C_k]

        if verbosity > 0:
            print(
                "  Found {} candidate itemsets of length {}.".format(
                    len(C_k), k
                )
            )
        if verbosity > 1:
            print("   {}".format(C_k))

        # If no candidate itemsets were found, break out of the loop
        if not C_k:
            break

        # Prepare counts of candidate itemsets (from the pruen step)
        candidate_itemset_counts = collections.defaultdict(int)
        if verbosity > 1:
            print("    Iterating over transactions.")
        for row, transaction in enumerate(transactions()):

            # If we've excluded this transaction earlier, do not consider it
            if not use_transaction[row]:
                continue

            # Assert that no items were found in this row
            found_any = False
            for candidate, candidate_set in zip(C_k, C_k_sets):

                # This is where most of the time is spent in the algorithm
                # If the candidate set is a subset, add count and mark the row
                if issubset(candidate_set, transaction):
                    candidate_itemset_counts[candidate] += 1
                    found_any = True

            # If no candidate sets were found in this row, skip this row of
            # transactions in the future
            if not found_any:
                use_transaction[row] = False

        # Only keep the candidates whose support is over the threshold
        C_k = [
            (i, c)
            for (i, c) in candidate_itemset_counts.items()
            if (c / num_transactions) >= min_support
        ]

        # If no itemsets were found, break out of the loop
        if not C_k:
            break

        # Candidate itemsets were found, add them and progress the while-loop
        # They must be sorted to maintain the invariant when joining/pruning
        large_itemsets[k] = {i: c for (i, c) in sorted(C_k)}

        if verbosity > 0:
            num_found = len(large_itemsets[k])
            pp = "  Found {} large itemsets of length {}.".format(num_found, k)
            print(pp)
        if verbosity > 1:
            print("   {}".format(list(large_itemsets[k].keys())))
        k += 1

        # Break out if we are about to consider larger itemsets than the max
        if k > max_length:
            break

    if verbosity > 0:
        print("Itemset generation terminated.\n")

    return large_itemsets, num_transactions


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "-v"])
