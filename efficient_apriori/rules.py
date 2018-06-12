#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to association rules.
"""

import itertools
from efficient_apriori.itemsets import itemsets_from_transactions

class Rule(object):
    """
    A class for a rule.
    """
    def __init__(self, lhs, rhs, count_full=0, count_lhs=0, count_rhs=0):
        """
        Initalize a new rule.
        """
        self.lhs = lhs # antecedent
        self.rhs = rhs # consequent
        self.count_full = count_full
        self.count_lhs = count_lhs
        self.count_rhs = count_rhs
        
    @property
    def confidence(self):
        return self.count_full / self.count_lhs
        
    def __repr__(self):
        lhs_formatted = '{' + ', '.join(k for k in self.lhs) + '}'
        rhs_formatted = '{' + ', '.join(k for k in self.rhs) + '}'
        repr_str = '{} -> {}'.format(lhs_formatted, rhs_formatted)
        return repr_str
    
    def pprint(self):
        lhs_formatted = '{' + ', '.join(k for k in self.lhs) + '}'
        rhs_formatted = '{' + ', '.join(k for k in self.rhs) + '}'
        conf = f'confidence: {self.confidence:.3f}'
        repr_str = '{} -> {} [{}]'.format(lhs_formatted, 
                    rhs_formatted, conf)
        return repr_str
    def __eq__(self, other):
        return (self.lhs == other.lhs) and (self.rhs == other.rhs)
    
    def __hash__(self):
        return hash(self.lhs + self.rhs)
    

def generate_rules_simple(itemsets, min_confidence):
    """
    The naive algorithm from the original paper.
    """
    min_conf = min_confidence
    for size in itemsets.keys():
        
        # Do not consider itemsets of size 1
        if size < 2:
            continue
        
        yielded = set()
        for itemset in itemsets[size].keys():
            for result in _genrules(itemset, itemset, itemsets, min_conf):
                if result in yielded:
                    continue
                else:
                    yielded.add(result)
                    yield result
                    

def _genrules(l_k, a_m, itemsets, min_conf):
    """
    The naive algorithm from the original paper.
    """
    #print(f'genrules(l_k={l_k}, a_m={a_m}, itemsets, min_conf)')
    
    def support(itemset):
        return itemsets[len(itemset)][itemset]
    
    #print(list(itertools.combinations(a_m, len(a_m) - 1)))
    for a_m in itertools.combinations(a_m, len(a_m) - 1):
        #print(l_k, a_m)
        conf = support(l_k) / support(a_m)
        if conf >= min_conf:
            rhs = set(l_k).difference(set(a_m))
            rhs = tuple(sorted(list(rhs)))
            #print(' ', Rule(a_m, rhs, support(l_k), support(a_m)))
            #print('   ', f'genrules(l_k={l_k}, a_m={a_m})')
            yield Rule(a_m, rhs, support(l_k), support(a_m))
            
            if (len(a_m)) > 1:
                yield from _genrules(l_k, a_m, itemsets, min_conf)
    
def ap_genrules():
    """
    The faster algorithm from the original paper.
    """
    pass


if __name__ == '__main__':
    
    transactions = ['cbd',
                    'abc',
                    'abd',
                    'cba',
                    'abf',
                    'acb',
                    'cbf',
                    'cfe',
                    'afe']
    
    large_itemsets = itemsets_from_transactions(transactions, min_support=1/3)
    rules = generate_rules_naively(large_itemsets, 0.1)
    for rule in rules:
        print(rule)
        
    print()
    rules = list(generate_rules_simple(large_itemsets, 0.1))
    rules.sort(key=lambda rule: rule.confidence)
    print()
    for rule in rules:
        print(rule.pprint())