#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to itemsets.
"""

import itertools
import numbers
import typing
import collections
from dataclasses import field, dataclass
import collections.abc


@dataclass
class ItemsetCount:
    itemset_count: int = 0
    members: set = field(default_factory=set)


class TransactionManager:
    # The brilliant transaction manager idea is due to:
    # https://github.com/ymoch/apyori/blob/master/apyori.py

    def __init__(self, transactions: typing.Iterable[typing.Iterable[typing.Hashable]]):

        # A lookup that returns indices of transactions for each item
        self._indices_by_item = collections.defaultdict(set)

        # Populate
        i = -1
        for i, transaction in enumerate(transactions):
            for item in transaction:
                self._indices_by_item[item].add(i)

        # Total number of transactions
        self._transactions = i + 1

    @property
    def items(self):
        return set(self._indices_by_item.keys())

    def __len__(self):
        return self._transactions

    def transaction_indices(self, transaction: typing.Iterable[typing.Hashable]):
        """Return the indices of the transaction."""

        transaction = set(transaction)  # Copy
        item = transaction.pop()
        indices = self._indices_by_item[item]
        while transaction:
            item = transaction.pop()
            indices = indices.intersection(self._indices_by_item[item])
        return indices

    def transaction_indices_sc(self, transaction: typing.Iterable[typing.Hashable], min_support: float = 0):
        """Return the indices of the transaction, with short-circuiting.

        Returns (over_or_equal_to_min_support, set_of_indices)
        """

        # Sort items by number of transaction rows the item appears in,
        # starting with the item beloning to the most transactions
        transaction = sorted(transaction, key=lambda item: len(self._indices_by_item[item]), reverse=True)

        # Pop item appearing in the fewest
        item = transaction.pop()
        indices = self._indices_by_item[item]
        support = len(indices) / len(self)
        if support < min_support:
            return False, None

        # The support is a non-increasing function
        # Sorting by number of transactions the items appear in is a heuristic
        # to make the support drop as quickly as possible
        while transaction:
            item = transaction.pop()
            indices = indices.intersection(self._indices_by_item[item])
            support = len(indices) / len(self)
            if support < min_support:
                return False, None

        # No short circuit happened
        return True, indices


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

        # The number of rows to skip in the while-loop, initially set to 1
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


def prune_step(itemsets: typing.Iterable[tuple], possible_itemsets: typing.List[tuple]):
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

    This is done efficiently by using the downward-closure property of the
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
    transactions: typing.Iterable[typing.Union[set, tuple, list]],
    min_support: float,
    max_length: int = 8,
    verbosity: int = 0,
    output_transaction_ids: bool = False,
):
    """
    Compute itemsets from transactions by building the itemsets bottom up and
    iterating over the transactions to compute the support repedately. This is
    the heart of the Apriori algorithm by Agrawal et al. in the 1994 paper.

    Parameters
    ----------
    transactions : a list of itemsets (tuples/sets/lists with hashable entries)
    min_support : float
        The minimum support of the itemsets, i.e. the minimum frequency as a
        percentage.
    max_length : int
        The maximum length of the itemsets.
    verbosity : int
        The level of detail printing when the algorithm runs. Either 0, 1 or 2.
    output_transaction_ids : bool
        If set to true, the output contains the ids of transactions that
        contain a frequent itemset. The ids are the enumeration of the
        transactions in the sequence they appear.

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
    if not (isinstance(min_support, numbers.Number) and (0 <= min_support <= 1)):
        raise ValueError("`min_support` must be a number between 0 and 1.")

    # Store in transaction manager
    manager = TransactionManager(transactions)

    # If no transactions are present
    transaction_count = len(manager)
    if transaction_count == 0:
        return dict(), 0  # large_itemsets, num_transactions

    # STEP 1 - Generate all large itemsets of size 1
    # ----------------------------------------------
    if verbosity > 0:
        print("Generating itemsets.")
        print(" Counting itemsets of length 1.")

    candidates: typing.Dict[tuple, int] = {(item,): len(indices) for item, indices in manager._indices_by_item.items()}
    large_itemsets: typing.Dict[int, typing.Dict[tuple, int]] = {
        1: {item: count for (item, count) in candidates.items() if (count / len(manager)) >= min_support}
    }

    if verbosity > 0:
        print("  Found {} candidate itemsets of length 1.".format(len(manager.items)))
        print("  Found {} large itemsets of length 1.".format(len(large_itemsets.get(1, dict()))))
    if verbosity > 1:
        print("    {}".format(list(item for item in large_itemsets.get(1, dict()).keys())))

    # If large itemsets were found, convert to dictionary
    if not large_itemsets.get(1, dict()):
        return dict(), 0  # large_itemsets, num_transactions

    # STEP 2 - Build up the size of the itemsets
    # ------------------------------------------

    # While there are itemsets of the previous size
    k = 2
    while large_itemsets[k - 1] and (max_length != 1):
        if verbosity > 0:
            print(" Counting itemsets of length {}.".format(k))

        # STEP 2a) - Build up candidate of larger itemsets

        # Retrieve the itemsets of the previous size, i.e. of size k - 1
        # They must be sorted to maintain the invariant when joining/pruning
        itemsets_list = sorted(item for item in large_itemsets[k - 1].keys())

        # Gen candidates of length k + 1 by joining, prune, and copy as set
        # This algorithm assumes that the list of itemsets are sorted,
        # and that the itemsets themselves are sorted tuples
        C_k: typing.List[tuple] = list(apriori_gen(itemsets_list))

        if verbosity > 0:
            print("  Found {} candidate itemsets of length {}.".format(len(C_k), k))
        if verbosity > 1:
            print("   {}".format(C_k))

        # If no candidate itemsets were found, break out of the loop
        if not C_k:
            break

        # Prepare counts of candidate itemsets (from the prune step)
        if verbosity > 1:
            print("    Iterating over transactions.")

        # Keep only large transactions
        found_itemsets: typing.Dict[tuple, int] = dict()
        for candidate in C_k:
            over_min_support, indices = manager.transaction_indices_sc(candidate, min_support=min_support)
            if over_min_support:
                found_itemsets[candidate] = len(indices)

        # If no itemsets were found, break out of the loop
        if not found_itemsets:
            break

        # Candidate itemsets were found, add them
        large_itemsets[k] = {i: counts for (i, counts) in found_itemsets.items()}

        if verbosity > 0:
            num_found = len(large_itemsets[k])
            print("  Found {} large itemsets of length {}.".format(num_found, k))
        if verbosity > 1:
            print("   {}".format(list(large_itemsets[k].keys())))
        k += 1

        # Break out if we are about to consider larger itemsets than the max
        if k > max_length:
            break

    if verbosity > 0:
        print("Itemset generation terminated.\n")

    if output_transaction_ids:
        itemsets_out = {
            length: {
                item: ItemsetCount(itemset_count=count, members=manager.transaction_indices(set(item)))
                for (item, count) in itemsets.items()
            }
            for (length, itemsets) in large_itemsets.items()
        }
        return itemsets_out, len(manager)

    return large_itemsets, len(manager)


if __name__ == "__main__":

    import pytest

    pytest.main(args=[".", "--doctest-modules", "-v"])
