#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to association rules.
"""

import itertools
from efficient_apriori.itemsets import apriori_gen


class Rule(object):
    """
    A class for a rule.
    """
    def __init__(self, lhs, rhs, count_full=0, count_lhs=0, count_rhs=0, 
                 num_transactions=0):
        """
        Initalize a new rule.
        """
        self.lhs = lhs  # antecedent
        self.rhs = rhs  # consequent
        self.count_full = count_full
        self.count_lhs = count_lhs
        self.count_rhs = count_rhs
        self.num_transactions = num_transactions
        
    @property
    def confidence(self):
        return self.count_full / self.count_lhs
    
    @property
    def support(self):
        return self.count_full / self.num_transactions
        
    def __repr__(self):
        lhs_formatted = '{' + ', '.join(str(k) for k in self.lhs) + '}'
        rhs_formatted = '{' + ', '.join(str(k) for k in self.rhs) + '}'
        repr_str = '{} -> {}'.format(lhs_formatted, rhs_formatted)
        return repr_str
    
    def pprint(self):
        lhs_formatted = '{' + ', '.join(str(k) for k in self.lhs) + '}'
        rhs_formatted = '{' + ', '.join(str(k) for k in self.rhs) + '}'
        conf = f'confidence: {self.confidence:.3f}'
        repr_str = '{} -> {} [{}]'.format(lhs_formatted, 
                                          rhs_formatted, conf)
        return repr_str
    
    def __eq__(self, other):
        return (self.lhs == other.lhs) and (self.rhs == other.rhs)
    
    def __hash__(self):
        return hash(self.lhs + self.rhs)
    

def generate_rules_simple(itemsets, min_confidence, num_transactions):
    """
    Simple top-down algorithm for generating association rules.
    
    This algorithm is presented in section 3 in the original 1994 paper by
    Agrawal. It works by building the rules top-down, calling the function
    `_genrules` to do most of the legwork.
    
    """
    min_conf = min_confidence
    for size in itemsets.keys():
        
        # Do not consider itemsets of size 1
        if size < 2:
            continue
        
        # TODO : Verify if this function MUST return duplicates,
        # or if the implementation is slightly wrong
        yielded = set()
        for itemset in itemsets[size].keys():
            for result in _genrules(itemset, itemset, itemsets, min_conf, 
                                    num_transactions):
                if result in yielded:
                    continue
                else:
                    yielded.add(result)
                    yield result
                    

def _genrules(l_k, a_m, itemsets, min_conf, num_transactions, recurse=True):
    """
    The naive algorithm from the original paper.
    """
    def support(itemset):
        return itemsets[len(itemset)][itemset]
    
    for a_m in itertools.combinations(a_m, len(a_m) - 1):
        conf = support(l_k) / support(a_m)
        if conf >= min_conf:
            rhs = set(l_k).difference(set(a_m))
            rhs = tuple(sorted(list(rhs)))
            yield Rule(a_m, rhs, support(l_k), support(a_m), support(rhs), 
                       num_transactions)
            
            if len(a_m) > 1 and recurse:
                yield from _genrules(l_k, a_m, itemsets, min_conf, 
                                     num_transactions, recurse=True)

 
def generate_rules_apriori(itemsets, min_confidence, num_transactions):
    """
    The faster algorithm from the original paper.
    """
    
    def support(itemset):
        return itemsets[len(itemset)][itemset]

    min_conf = min_confidence
    for size in itemsets.keys():
        
        # Do not consider itemsets of size 1
        if size < 2:
            continue
        
        for itemset in itemsets[size].keys():
            H_1 = list(itertools.combinations(itemset, 1))
            
            
            for removed in itertools.combinations(itemset, 1):
                rhs = set(itemset).difference(set(removed))
                rhs = tuple(sorted(list(rhs)))
                conf = support(itemset) / support(removed)
                if conf >= min_conf:
                    yield Rule(removed, rhs, support(itemset), support(removed), 
                           support(rhs), num_transactions)
                
            yield from _ap_genrules(itemset, H_1, itemsets, min_conf, 
                                num_transactions)
    
    
    
def _ap_genrules(itemset, H_1, itemsets, min_conf, num_transactions):
    """
    The faster algorithm from the original paper.
    """
    print(f'_ap_genrules(itemset={itemset}, H_1={H_1})')
    
    def support(itemset):
        return itemsets[len(itemset)][itemset]

    
    #print(f' Comparing {len(itemset)} > {len(H_1[0]) + 1}')
    if len(itemset) > (len(H_1[0]) + 1):

        H_m = list(apriori_gen(H_1))
        print(f'  {H_m}')
        H_m_copy = H_m.copy()
        for h_m in H_m:
            rhs = set(itemset).difference(set(h_m))
            rhs = tuple(sorted(list(rhs)))
            
            
            
            conf = support(itemset) / support(h_m)
            
            if conf >= min_conf:
                yield Rule(h_m, rhs, support(itemset), support(h_m), 
                           support(rhs), num_transactions)
            else:
                #print(f' Removed: {h_m}')
                H_m_copy.remove(h_m)
            
            
        yield from _ap_genrules(itemset, H_m_copy, itemsets, min_conf, num_transactions)
            
            
    
    


if __name__ == '__main__':
    #import pytest
    #pytest.main(args=['.', '--doctest-modules', '-v'])
    pass

    from efficient_apriori.itemsets import itemsets_from_transactions
    transactions = [('a', 'b', 'c'), ('a', 'b', 'c'), ('a', 'b', 'd')]
    itemsets, num_transactions = itemsets_from_transactions(transactions, min_support=1/2)
    
    print(itemsets)
    
    for rule in generate_rules_simple(itemsets, 0.1, num_transactions):
        print(rule, rule.confidence, rule.support)
        
    print('---------')
    
    for rule in generate_rules_apriori(itemsets, 0.1, num_transactions):
        print(rule, rule.confidence, rule.support)
        
        
    assert set(generate_rules_simple(itemsets, 0.1, num_transactions)) == set(generate_rules_apriori(itemsets, 0.1, num_transactions))
    
    
    
    transactions = [('a', 'b', 'c'), ('a', 'b', 'c'), ('a', 'b', 'd'), ('d', 'a', 'f'), ('b', 'c', 'd')]
    itemsets, num_transactions = itemsets_from_transactions(transactions, min_support=1/20)
    
    print(itemsets)
    
    for rule in generate_rules_simple(itemsets, 0.1, num_transactions):
        print(rule, rule.confidence, rule.support)
        
    print('---------')
    
    for rule in generate_rules_apriori(itemsets, 0.1, num_transactions):
        print(rule, rule.confidence, rule.support)
        
        
    assert set(generate_rules_simple(itemsets, 0.1, num_transactions)) == set(generate_rules_apriori(itemsets, 0.1, num_transactions))
    
    
    

    
    
    