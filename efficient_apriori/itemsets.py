#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to itemsets.
"""

import itertools
import collections
import numbers
import typing
from abc import ABC, abstractmethod

from collections import defaultdict
from dataclasses import field, dataclass


@dataclass
class ItemsetCount:
    itemset_count: int = 0
    members: set = field(default_factory=set)


class TransactionWithId:
    def __init__(self, transaction: tuple, id_: typing.Optional):
        self.transaction = transaction
        self.id = id_

    def __repr__(self):
        t = ', '.join(self.transaction)
        return "TransactionWithId{transaction=%s, id=%s}" % (t, self.id)


class TransactionBroker(ABC):
    def __init__(self, transactions):
        self.transactions = transactions

    @abstractmethod
    def rows(self):
        pass


class Transactions(TransactionBroker):
    def rows(self):
        transaction_sets = [set(t) for t in self.transactions if len(t) > 0]
        return enumerate(transaction_sets)


class TransactionsWithIDs(TransactionBroker):
    def rows(self):
        return [(t.id, set(t.transaction),) for t in self.transactions]


class GeneratorTransactions(Transactions):
    def rows(self):
        count = 0
        for t in self.transactions():
            yield count, set(t)
            count += 1


class GeneratorTransactionsWithIds(TransactionsWithIDs):
    def rows(self):
        for t in self.transactions():
            yield t.id, set(t.transaction)


class CountBroker(ABC):
    @abstractmethod
    def get_counter(self):
        pass

    @abstractmethod
    def increment_count(self, transaction_id: str, count):
        pass

    @abstractmethod
    def get_count(self, count):
        pass


class Count(CountBroker):
    def get_counter(self):
        return 0

    def increment_count(self, _, count):
        return count + 1

    def get_count(self, count):
        return count


class CountWithIds(CountBroker):
    def get_counter(self):
        return ItemsetCount()

    def increment_count(self, transaction_id: str, count: ItemsetCount):
        count.itemset_count += 1
        count.members.add(transaction_id)
        return count

    def get_count(self, count):
        return count.itemset_count


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
            removed = possible_itemset[:i] + possible_itemset[i + 1:]

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
        transactions: typing.Union[typing.List[tuple],
                                   typing.Callable,
                                   typing.List[TransactionWithId]],
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
    transactions : a list of itemsets (tuples with hashable entries),
                   a list of itemsets.TransactionWithId
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

    def empty_result(transaction_count):
        return dict(), transaction_count

    # STEP 0 - Sanitize user inputs
    # -----------------------------
    if not (
            isinstance(min_support, numbers.Number) and (0 <= min_support <= 1)
    ):
        raise ValueError("`min_support` must be a number between 0 and 1.")

    if not transactions:
        return empty_result(0)
    elif isinstance(transactions, collections.Iterable):
        first = next(iter(transactions))
        if transactions and isinstance(first, TransactionWithId):
            transaction_broker = TransactionsWithIDs(transactions)
            count_broker = CountWithIds()
        else:
            transaction_broker = Transactions(transactions)
            count_broker = Count()
    # Assume the transactions is a callable, returning a generator
    elif callable(transactions):
        generator = transactions()
        if not isinstance(generator, collections.abc.Generator):
            msg = (
                    "`transactions` must be an iterable or a callable "
                    + "returning an iterable."
            )
            raise TypeError(msg)
        first = next(generator)
        if isinstance(first, TransactionWithId):
            transaction_broker = GeneratorTransactionsWithIds(transactions)
            count_broker = CountWithIds()
        else:
            transaction_broker = GeneratorTransactions(transactions)
            count_broker = Count()
    else:
        msg = (
                "`transactions` must be an iterable or a callable "
                + "returning an iterable."
        )
        raise TypeError(msg)

    # Keep a dictionary stating whether to consider the row, this will allow
    # row-pruning later on if no information was retrieved earlier from it
    use_transaction = defaultdict(lambda: True)

    # STEP 1 - Generate all large itemsets of size 1
    # ----------------------------------------------
    if verbosity > 0:
        print("Generating itemsets.")
        print(" Counting itemsets of length 1.")

    c = defaultdict(count_broker.get_counter)
    num_transactions = 0
    for row, transaction in transaction_broker.rows():
        num_transactions += 1  # Increment counter for transactions
        for item in transaction:
            item_count = c[item]
            # Increment counter for single-item itemsets
            c[item] = count_broker.increment_count(row, item_count)

    large_itemsets = [
        (i, c)
        for (i, c) in c.items()
        if (count_broker.get_count(c) / num_transactions) >= min_support
    ]

    if verbosity > 0:
        num_cand, num_itemsets = len(c.items()), len(large_itemsets)
        print("  Found {} candidate itemsets of length 1.".format(num_cand))
        print("  Found {} large itemsets of length 1.".format(num_itemsets))
    if verbosity > 1:
        print("    {}".format(list((i,) for (i, c) in large_itemsets)))

    # If large itemsets were found, convert to dictionary
    if large_itemsets:
        large_itemsets = {1: {(i,): c for (i, c) in sorted(large_itemsets)}}

    # No large itemsets were found, return immediately
    else:
        return empty_result(num_transactions)

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

        # Gen candidates of length k + 1 by joining, prune, and copy as set
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

        # Prepare counts of candidate itemsets (from the prune step)
        counts = defaultdict(count_broker.get_counter)
        if verbosity > 1:
            print("    Iterating over transactions.")
        for row, transaction in transaction_broker.rows():

            # If we've excluded this transaction earlier, do not consider it
            if not use_transaction[row]:
                continue

            # Assert that no items were found in this row
            found_any = False
            for candidate, candidate_set in zip(C_k, C_k_sets):

                # This is where most of the time is spent in the algorithm
                # If the candidate set is a subset, add count and mark the row
                if issubset(candidate_set, transaction):
                    c = counts[candidate]
                    counts[candidate] = count_broker.increment_count(row, c)
                    found_any = True

            # If no candidate sets were found in this row, skip this row of
            # transactions in the future
            if not found_any:
                use_transaction[row] = False

        # Only keep the candidates whose support is over the threshold
        C_k = [
            (i, c)
            for (i, c) in counts.items()
            if (count_broker.get_count(c) / num_transactions) >= min_support
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
