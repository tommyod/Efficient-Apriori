#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:51:09 2018

@author: tommy
"""

import itertools
import collections
import pytest
import random

def join_step(itemsets):
    """
    Join k length itemsets into k + 1 length itemsets.
    Assumes that the list of itemsets, and the itemsets, are sorted.
    
    Examples
    --------
    >>> 2 + 2
    4
    """

    # Iterate over every itemset in the itemsets
    i = 0
    while i < len(itemsets):
        
        # The number of rows to skip
        skip = 1
        
        # Get all but the last item in the itemset, and the last one
        *itemset_first, itemset_last = itemsets[i]
        
        # To obtain every tail item, go through the next itemsets
        # which start with the same items
        tail_items = [itemset_last]
        tail_items_append = tail_items.append  # Micro-optimization
        for j in range(i + 1, len(itemsets)):
            
            # Get first and last parts
            *itemset_n_first, itemset_n_last = itemsets[j]

            # If it's the same, append and skip this itemset
            if itemset_first == itemset_n_first:
                
                # Micro-optimization
                tail_items_append(itemset_n_last)
                skip += 1
                
            # If it's not the same, break out
            else:
                break
            
        # For every 2-combination in the tail items, yield a new candidate
        # itemset, which is sorted
        itemset_first = tuple(itemset_first)
        for (a, b) in sorted(itertools.combinations(tail_items, 2)):
            yield itemset_first + (a,) + (b,)            
        i += skip

def prune_step(itemsets, possible_itemsets):
    """
    Prune possible itemsets whose subsets are not in the itemsets.
    
    Examples
    -------
    >>> 1 + 1
    2
    """
    
    # For faster lookups
    itemsets = set(itemsets)
    
    # Go through every possible itemset
    for possible_itemset in possible_itemsets:
        
        # Remove 1 from the combination, same as k-1 combinations
        for i in range(len(possible_itemset)):
            removed = possible_itemset[:i] + possible_itemset[i+1:]
            
            # If the k + 1 itemset with one removed is not in
            # the length k itemsets, break -> it's not a candidate
            if removed not in itemsets:
                break
            
        # If we have not breaked yet
        else:
            yield possible_itemset
        

def apriori_gen(itemsets):
    """
    Compute all possible k + 1 length supersets from k length itemsets.
    """
    possible_extensions = join_step(itemsets)
    yield from prune_step(itemsets, possible_extensions)
    
    
def itemsets_from_transactions_naive(transactions, min_support):
    """
    Naive algorithm for testing.
    """
    unique_items = set(k for t in transactions for k in t)

    L = collections.defaultdict(list)
    for k in range(1, len(unique_items) + 1):
        for combination in itertools.combinations(unique_items, k):
            counts = 0
            for transaction in transactions:
                if set.issubset(set(combination), set(transaction)):
                    counts += 1     
            if counts >= min_support:
                L[k].append(tuple(sorted(list(combination)))) 
    return L
                

def itemsets_from_transactions(transactions, min_support):
    """
    Stage 1 of apriori algotihm.
    """
    # Large 1-itemsets
    
    L = collections.Counter(i for transaction in transactions for i in transaction)
    L = [(key,) for (key, value) in L.items() if value >= min_support]
    #print(L)
    L = {1: sorted(L)}
    import time
    
    k = 2
    while L[k - 1]:
        #print(f'k = {k}')
        #time.sleep(0.01)
        
        # Generate new candidates
        possible_extensions = list(join_step(L[k - 1]))
        C_k = list(prune_step(L[k - 1], possible_extensions))
        #print(C_k)
        
        counts = collections.defaultdict(int)
        # Check if they are in the transactions
        for transaction in transactions:
            for candidate in C_k:
                if set.issubset(set(candidate), set(transaction)):
                    counts[candidate] += 1
            
        
        C_k = [k for (k, v) in counts.items() if v >= min_support]
        if C_k:
            L[k] = sorted(C_k)
        else:
            break
        
        k += 1
        
    return L
        


def apriori_classic(transactions, min_support, min_confidence):
    """
    The classic apriori algotihm.
    Assumes sorted transactions, and sorted items in the transactions.
    """
    import time
    start_time= time.perf_counter()
    sets1 = itemsets_from_transactions(transactions, min_support)
    #print(sets1)
    print(f'Itemset algo ran in \t: {time.perf_counter() - start_time} s')
    
    start_time= time.perf_counter()
    sets2 = itemsets_from_transactions_naive(transactions, min_support)
    #print(sets2)
    print(f'Naive algo ran in \t: {time.perf_counter() - start_time} s')
    
    #assert dict(sets1) == dict(sets2)
    
    for k in sets1.keys():
        if not sets1[k] == sets2[k]:
            print('ERROR', sets1[k], sets2[k])
            assert False
        
    for k in sets2.keys():
        if not sets1[k] == sets2[k]:
            print('ERROR', sets1[k], sets2[k])
            assert False

def sort_transactions(transactions):
    pass






if __name__ == '__main__':
    
    itemsets = [(1, 2, 3), 
                (1, 2, 4), 
                (1, 3, 4), 
                (1, 3, 5), 
                (2, 3, 4)]
    assert list(apriori_gen(itemsets)) == [(1, 2, 3, 4)]
    
    assert list(join_step([(1, 2, 3), (1, 2, 4), (1, 2, 5)])) == \
    [(1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 4, 5)]
    possible_extensions = list(join_step(itemsets))
    new_itemsets = prune_step(itemsets, possible_extensions)
    print(list(new_itemsets))
    
    transactions = [(1, 2, 3, 4, 5),
                   (1, 3, 2),
                   (1, 4, 2),
                   (1, 4, 2, 3),
                   (3, 2, 1, 5, 7),
                   (4, 2, 3),
                   (1, 2, 3)]
    
    transactions = [tuple(sorted(list(t))) for t in transactions]
    
    for t in transactions:
        print(t)
    print('-----------------------')
        
    apriori_classic(transactions, min_support=2, min_confidence=2)
    
    
    
    for test in range(10):
        
        transactions = [tuple(set([random.randint(0, 14) \
        for i in range(random.randint(1, 10))])) for t in range(random.randint(10, 50))]
        print(f'-------------------------------\nTest number {test}')
        #print(transactions)
        
        apriori_classic(transactions, min_support=2, min_confidence=2)
        
    
    
    