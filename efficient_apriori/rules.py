#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to association rules.
"""

import typing
import itertools
from efficient_apriori.itemsets import apriori_gen


class Rule(object):
    """
    A class for a rule.
    """
    
    # Number of decimals used for printing
    _decimals = 3
    
    def __init__(self, lhs: tuple, rhs: tuple, count_full: int=0, 
                 count_lhs: int=0, count_rhs: int=0, num_transactions: int=0):
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
        >>> r.support  # Probability of ('a', 'c', 'c') in the data
        0.25
        >>> # Ratio of observed over expected support if lhs, rhs = independent
        >>> r.lift == 2 / 3
        True
        >>> print(r)
        {a, b} -> {c} (conf: 0.500, supp: 0.250, lift: 0.667)
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
        except (ZeroDivisionError, AttributeError) as error:
            return None
    
    @property
    def support(self):
        """
        The support of a rule is the frequency of which the lhs and rhs appear
        together in the dataset. If X -> Y, then the support is P(Y and X).
        """
        try:
            return self.count_full / self.num_transactions
        except (ZeroDivisionError, AttributeError) as error:
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
            expected_support = (prod_counts) / self.num_transactions ** 2
            return observed_support / expected_support
        except (ZeroDivisionError, AttributeError) as error:
            return None
        
    @staticmethod
    def _pf(s):
        """
        Pretty formatting of an iterable.
        """
        return '{' + ', '.join(str(k) for k in s) + '}'
        
    def __repr__(self):
        """
        Representation of a rule.
        """
        return '{} -> {}'.format(self._pf(self.lhs), 
                                 self._pf(self.rhs))
    
    def __str__(self):
        """
        Printing of a rule.
        """
        conf = f'conf: {self.confidence:.3f}'
        supp = f'supp: {self.support:.3f}'
        lift = f'lift: {self.lift:.3f}'
        return '{} -> {} ({}, {}, {})'.format(self._pf(self.lhs), 
                                              self._pf(self.rhs), 
                                              conf, supp, lift)
    
    def __eq__(self, other):
        """
        Equality of two rules.
        """
        return (self.lhs == other.lhs) and (self.rhs == other.rhs)
    
    def __hash__(self):
        """
        Hashing a rule for efficient set and dict representation.
        """
        return hash(self.lhs + self.rhs)
    

def generate_rules_simple(itemsets: typing.List[tuple], min_confidence: float, 
                          num_transactions: int):
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
        yielded = set()
        yielded_add = yielded.add
        
        # Iterate over every itemset of the prescribed size
        for itemset in itemsets[size].keys():
            
            # Generate rules
            for result in _genrules(itemset, itemset, itemsets, min_confidence, 
                                    num_transactions):
                
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
        rhs = tuple(sorted(list(rhs)))
        
        # Create new rule object and yield it
        yield Rule(a_m, rhs, count(l_k), count(a_m), count(rhs), 
                   num_transactions)
        
        # If the left hand side has one item only, do not recurse the function
        if len(a_m) <= 1:
            continue
        yield from _genrules(l_k, a_m, itemsets, min_conf, num_transactions)

 
def generate_rules_apriori(itemsets: typing.List[tuple], min_confidence: float, 
                           num_transactions: int):
    """
    Bottom up algorithm for generating association rules from itemsets, very
    similar to the fast algorithm proposed in the original 1994 paper by 
    Agrawal et al.
    
    Parameters
    ----------
    l_k : tuple
        The itemset containing
        
    Examples
    --------
    >>> itemsets = {1: {('a',): 3, ('b',): 2, ('c',): 1}, 
    ...             2: {('a', 'b'): 2, ('a', 'c'): 1}}
    >>> list(generate_rules_apriori(itemsets, 1, 3))
    [{b} -> {a}, {c} -> {a}]
    """
    
    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    # For every itemset of a perscribed size
    for size in itemsets.keys():
        
        # Do not consider itemsets of size 1
        if size < 2:
            continue
        
        # For every itemset of this size
        for itemset in itemsets[size].keys():
            
            # Special case to capture rules such as {1_item} -> {others}
            for removed in itertools.combinations(itemset, 1):
                
                # Compute the right hand side
                rhs = set(itemset).difference(set(removed))
                rhs = tuple(sorted(list(rhs)))
                
                # If the confidence is high enough, yield the rule
                conf = count(itemset) / count(removed)
                if conf >= min_confidence:
                    yield Rule(removed, rhs, count(itemset), 
                               count(removed), count(rhs), 
                               num_transactions)
                    
            # Generate combinations to start off of. These 1-combinations will
            # be merged to 2-combinations in the function `_ap_genrules`
            H_1 = list(itertools.combinations(itemset, 1))
            yield from _ap_genrules(itemset, H_1, itemsets, min_confidence, 
                                    num_transactions)
    
    
def _ap_genrules(itemset, H_1, itemsets, min_conf, num_transactions):
    """
    Recursive algorithm to build up rules from a bottom-up approach.
    """
    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    # If H_1 is so large that calling `apriori_gen` will produce right-hand
    # sides as large as `itemset`, there will be no right hand side
    # This cannot happen, so abort if it will
    if len(itemset) <= (len(H_1[0]) + 1):
        return
    
    # Generate left-hand itemsets of length k + 1 if H is of length k
    H_m = list(apriori_gen(H_1))
    
    H_m_copy = H_m.copy()
    for h_m in H_m:
        # Compute the right ahdn side of the rule
        rhs = tuple(sorted(list(set(itemset).difference(set(h_m)))))

        conf = count(itemset) / count(h_m)
        
        # TODO: Should this rule be the other way around?
        if conf >= min_conf:
            yield Rule(h_m, rhs, count(itemset), count(h_m), 
                       count(rhs), num_transactions)
        else:
            H_m_copy.remove(h_m)
        
    yield from _ap_genrules(itemset, H_m_copy, itemsets, min_conf, 
                            num_transactions)


if __name__ == '__main__':
    import pytest
    pytest.main(args=['.', '--doctest-modules', '-v'])