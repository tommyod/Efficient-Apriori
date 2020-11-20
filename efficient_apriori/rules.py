#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to association rules.
"""

import typing
import numbers
import itertools
from efficient_apriori.itemsets import apriori_gen


class Rule(object):
    """
    A class for a rule.
    """

    # Number of decimals used for printing
    _decimals = 3

    def __init__(
        self,
        lhs: tuple,
        rhs: tuple,
        count_full: int = 0,
        count_lhs: int = 0,
        count_rhs: int = 0,
        num_transactions: int = 0,
    ):
        """
        Initialize a new rule. This call is a thin wrapper around some data.

        Parameters
        ----------
        lhs : tuple
            The left hand side (antecedent) of the rule. Each item in the tuple
            must be hashable, e.g. a string or an integer.
        rhs : tuple
            The right hand side (consequent) of the rule.
        count_full : int
            The count of the union of the lhs and rhs in the dataset.
        count_lhs : int
            The count of the lhs in the dataset.
        count_rhs : int
            The count of the rhs in the dataset.
        num_transactions : int
            The number of transactions in the dataset.

        Examples
        --------
        >>> r = Rule(('a', 'b'), ('c',), 50, 100, 150, 200)
        >>> r.confidence  # Probability of 'c', given 'a' and 'b'
        0.5
        >>> r.support  # Probability of ('a', 'b', 'c') in the data
        0.25
        >>> # Ratio of observed over expected support if lhs, rhs = independent
        >>> r.lift == 2 / 3
        True
        >>> print(r)
        {a, b} -> {c} (conf: 0.500, supp: 0.250, lift: 0.667, conv: 0.500)
        >>> r
        {a, b} -> {c}
        """
        self.lhs = lhs  # antecedent
        self.rhs = rhs  # consequent
        self.count_full = count_full
        self.count_lhs = count_lhs
        self.count_rhs = count_rhs
        self.num_transactions = num_transactions

    @property
    def confidence(self):
        """
        The confidence of a rule is the probability of the rhs given the lhs.
        If X -> Y, then the confidence is P(Y|X).
        """
        try:
            return self.count_full / self.count_lhs
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def support(self):
        """
        The support of a rule is the frequency of which the lhs and rhs appear
        together in the dataset. If X -> Y, then the support is P(Y and X).
        """
        try:
            return self.count_full / self.num_transactions
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def lift(self):
        """
        The lift of a rule is the ratio of the observed support to the expected
        support if the lhs and rhs were independent.If X -> Y, then the lift is
        given by the fraction P(X and Y) / (P(X) * P(Y)).
        """
        try:
            observed_support = self.count_full / self.num_transactions
            prod_counts = self.count_lhs * self.count_rhs
            expected_support = prod_counts / self.num_transactions ** 2
            return observed_support / expected_support
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def conviction(self):
        """
        The conviction of a rule X -> Y is the ratio P(not Y) / P(not Y | X).
        It's the proportion of how often Y does not appear in the data to how
        often Y does not appear in the data, given X. If the ratio is large,
        then the confidence is large and Y appears often.
        """
        try:
            eps = 10e-10  # Avoid zero division
            prob_not_rhs = 1 - self.count_rhs / self.num_transactions
            prob_not_rhs_given_lhs = 1 - self.confidence
            return prob_not_rhs / (prob_not_rhs_given_lhs + eps)
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def rpf(self):
        """
        The RPF (Rule Power Factor) is the confidence times the support.
        """
        try:
            return self.confidence * self.support
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @staticmethod
    def _pf(s):
        """
        Pretty formatting of an iterable.
        """
        return "{" + ", ".join(str(k) for k in s) + "}"

    def __repr__(self):
        """
        Representation of a rule.
        """
        return "{} -> {}".format(self._pf(self.lhs), self._pf(self.rhs))

    def __str__(self):
        """
        Printing of a rule.
        """
        conf = "conf: {0:.3f}".format(self.confidence)
        supp = "supp: {0:.3f}".format(self.support)
        lift = "lift: {0:.3f}".format(self.lift)
        conv = "conv: {0:.3f}".format(self.conviction)

        return "{} -> {} ({}, {}, {}, {})".format(self._pf(self.lhs), self._pf(self.rhs), conf, supp, lift, conv)

    def __eq__(self, other):
        """
        Equality of two rules.
        """
        return (set(self.lhs) == set(other.lhs)) and (set(self.rhs) == set(other.rhs))

    def __hash__(self):
        """
        Hashing a rule for efficient set and dict representation.
        """
        return hash(frozenset(self.lhs + self.rhs))

    def __len__(self):
        """
        The length of a rule, defined as the number of items in the rule.
        """
        return len(self.lhs + self.rhs)


def generate_rules_simple(
    itemsets: typing.Dict[int, typing.Dict],
    min_confidence: float,
    num_transactions: int,
):
    """
    DO NOT USE. This is a simple top-down algorithm for generating association
    rules. It is included here for testing purposes, and because it is
    mentioned in the 1994 paper by Agrawal et al. It is slow because it does
    not enumerate the search space efficiently: it produces duplicates, and it
    does not prune the search space efficiently.

    Simple algorithm for generating association rules from itemsets.
    """

    # Iterate over every size
    for size in itemsets.keys():

        # Do not consider itemsets of size 1
        if size < 2:
            continue

        # This algorithm returns duplicates, so we keep track of items yielded
        # in a set to avoid yielding duplicates
        yielded: set = set()
        yielded_add = yielded.add

        # Iterate over every itemset of the prescribed size
        for itemset in itemsets[size].keys():

            # Generate rules
            for result in _genrules(itemset, itemset, itemsets, min_confidence, num_transactions):

                # If the rule has been yieded, keep going, else add and yield
                if result in yielded:
                    continue
                else:
                    yielded_add(result)
                    yield result


def _genrules(l_k, a_m, itemsets, min_conf, num_transactions):
    """
    DO NOT USE. This is the gen-rules algorithm from the 1994 paper by Agrawal
    et al. It's a subroutine called by `generate_rules_simple`. However, the
    algorithm `generate_rules_simple` should not be used.
    The naive algorithm from the original paper.

    Parameters
    ----------
    l_k : tuple
        The itemset containing all elements to be considered for a rule.
    a_m : tuple
        The itemset to take m-length combinations of, an move to the left of
        l_k. The itemset a_m is a subset of l_k.
    """

    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    # Iterate over every k - 1 combination of a_m to produce
    # rules of the form a -> (l - a)
    for a_m in itertools.combinations(a_m, len(a_m) - 1):

        # Compute the count of this rule, which is a_m -> (l_k - a_m)
        confidence = count(l_k) / count(a_m)

        # Keep going if the confidence level is too low
        if confidence < min_conf:
            continue

        # Create the right hand set: rhs = (l_k - a_m) , and keep it sorted
        rhs = set(l_k).difference(set(a_m))
        rhs = tuple(sorted(rhs))

        # Create new rule object and yield it
        yield Rule(a_m, rhs, count(l_k), count(a_m), count(rhs), num_transactions)

        # If the left hand side has one item only, do not recurse the function
        if len(a_m) <= 1:
            continue
        yield from _genrules(l_k, a_m, itemsets, min_conf, num_transactions)


def generate_rules_apriori(
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_confidence: float,
    num_transactions: int,
    verbosity: int = 0,
):
    """
    Bottom up algorithm for generating association rules from itemsets, very
    similar to the fast algorithm proposed in the original 1994 paper by
    Agrawal et al.

    The algorithm is based on the observation that for {a, b} -> {c, d} to
    hold, both {a, b, c} -> {d} and {a, b, d} -> {c} must hold, since in
    general conf( {a, b, c} -> {d} ) >= conf( {a, b} -> {c, d} ).
    In other words, if either of the two one-consequent rules do not hold, then
    there is no need to ever consider the two-consequent rule.

    Parameters
    ----------
    itemsets : dict of dicts
        The first level of the dictionary is of the form (length, dict of item
        sets). The second level is of the form (itemset, count_in_dataset)).
    min_confidence :  float
        The minimum confidence required for the rule to be yielded.
    num_transactions : int
        The number of transactions in the data set.
    verbosity : int
        The level of detail printing when the algorithm runs. Either 0, 1 or 2.

    Examples
    --------
    >>> itemsets = {1: {('a',): 3, ('b',): 2, ('c',): 1},
    ...             2: {('a', 'b'): 2, ('a', 'c'): 1}}
    >>> list(generate_rules_apriori(itemsets, 1.0, 3))
    [{b} -> {a}, {c} -> {a}]
    """
    # Validate user inputs
    if not ((0 <= min_confidence <= 1) and isinstance(min_confidence, numbers.Number)):
        raise ValueError("`min_confidence` must be a number between 0 and 1.")

    if not ((num_transactions >= 0) and isinstance(num_transactions, numbers.Number)):
        raise ValueError("`num_transactions` must be a number greater than 0.")

    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    if verbosity > 0:
        print("Generating rules from itemsets.")

    # For every itemset of a perscribed size
    for size in itemsets.keys():

        # Do not consider itemsets of size 1
        if size < 2:
            continue

        if verbosity > 0:
            print(" Generating rules of size {}.".format(size))

        # For every itemset of this size
        for itemset in itemsets[size].keys():

            # Special case to capture rules such as {others} -> {1 item}
            for removed in itertools.combinations(itemset, 1):

                # Compute the left hand side
                remaining = set(itemset).difference(set(removed))
                lhs = tuple(sorted(remaining))

                # If the confidence is high enough, yield the rule
                conf = count(itemset) / count(lhs)
                if conf >= min_confidence:
                    yield Rule(
                        lhs,
                        removed,
                        count(itemset),
                        count(lhs),
                        count(removed),
                        num_transactions,
                    )

            # Generate combinations to start off of. These 1-combinations will
            # be merged to 2-combinations in the function `_ap_genrules`
            H_1 = list(itertools.combinations(itemset, 1))
            yield from _ap_genrules(itemset, H_1, itemsets, min_confidence, num_transactions)

    if verbosity > 0:
        print("Rule generation terminated.\n")


def _ap_genrules(
    itemset: tuple,
    H_m: typing.List[tuple],
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_conf: float,
    num_transactions: int,
):
    """
    Recursively build up rules by adding more items to the right hand side.

    This algorithm is called `ap-genrules` in the original paper. It is
    called by the `generate_rules_apriori` generator above. See it's docs.

    Parameters
    ----------
    itemset : tuple
        The itemset under consideration.
    H_m : tuple
        Subsets of the itemset of length m, to be considered for rhs of a rule.
    itemsets : dict of dicts
        All itemsets and counts for in the data set.
    min_conf : float
        The minimum confidence for a rule to be returned.
    num_transactions : int
        The number of transactions in the data set.
    """

    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    # If H_1 is so large that calling `apriori_gen` will produce right-hand
    # sides as large as `itemset`, there will be no right hand side
    # This cannot happen, so abort if it will
    if len(itemset) <= (len(H_m[0]) + 1):
        return

    # Generate left-hand itemsets of length k + 1 if H is of length k
    H_m = list(apriori_gen(H_m))
    H_m_copy = H_m.copy()

    # For every possible right hand side
    for h_m in H_m:
        # Compute the right hand side of the rule
        lhs = tuple(sorted(set(itemset).difference(set(h_m))))

        # If the confidence is high enough, yield the rule, else remove from
        # the upcoming recursive generator call
        if (count(itemset) / count(lhs)) >= min_conf:
            yield Rule(
                lhs,
                h_m,
                count(itemset),
                count(lhs),
                count(h_m),
                num_transactions,
            )
        else:
            H_m_copy.remove(h_m)

    # Unless the list of right-hand sides is empty, recurse the generator call
    if H_m_copy:
        yield from _ap_genrules(itemset, H_m_copy, itemsets, min_conf, num_transactions)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "-v"])
