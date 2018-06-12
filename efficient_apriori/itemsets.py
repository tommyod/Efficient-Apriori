#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to itemsets.
"""

import itertools
import collections
import numbers

def join_step(itemsets):
    """
    Join k length itemsets into k + 1 length itemsets.
    Assumes that the list of itemsets, and the itemsets, are sorted.
    
    Examples
    --------
    >>> # This is an example from the 1994 paper by Agrawal et al. 
    >>> itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]
    >>> list(join_step(itemsets))
    [(1, 2, 3, 4), (1, 3, 4, 5)]
    """
    # Iterate over every itemset in the itemsets
    i = 0
    while i < len(itemsets):
        
        # The number of rows to skip
        skip = 1
        
        # Get all but the last item in the itemset, and the last one
        #print(itemsets)
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
    >>> # This is an example from the 1994 paper by Agrawal et al. 
    >>> itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]
    >>> possible_itemsets = list(join_step(itemsets))
    >>> list(prune_step(itemsets, possible_itemsets))
    [(1, 2, 3, 4)]
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
                

def itemsets_from_transactions(transactions, min_support):
    """
    Return a dictionary of itemsets from transactions, where every itemset
    has at least the minimum support.
    
    This algorithm is the simplest one from the 1994 paper by Agrawal et al. 
    
    Examples
    --------
    >>> # This is an example from the 1994 paper by Agrawal et al. 
    >>> transactions = [(1, 3, 4), (2, 3, 5), (1, 2, 3, 5), (2, 5)]
    >>> itemsets = itemsets_from_transactions(transactions, min_support=2/5)
    >>> itemsets[1]
    {(1,): 2, (2,): 3, (3,): 3, (5,): 3}
    >>> itemsets[2]
    {(1, 3): 2, (2, 3): 2, (2, 5): 3, (3, 5): 2}
    >>> itemsets[3]
    {(2, 3, 5): 2}
    """
    # STEP 0 - Sanitize user inputs
    # -----------------------------
    
    # If the transactions are iterable, convert it to sets for faster lookups
    if isinstance(transactions, collections.Iterable):
        transaction_sets = [set(t) for t in transactions if len(t) > 0]
        def transactions():
            return transaction_sets
        
    # If the transactions is a callable, we assume that it returns a generator
    elif isinstance(transactions, collections.Callable):
        pass
    else:
        msg = f'`transactions` must be an iterable or a callable returning an \
                iterable.'
        raise TypeError(msg)
        
    if not (isinstance(min_support, numbers.Number) 
            and (0 <= min_support <= 1)):
        msg = f'`min_support` must be an integer >= 0.'
        raise ValueError('')
        
    use_transaction = collections.defaultdict(lambda: True)
        
    # STEP 1 - Generate all large itemsets of size 1
    # ----------------------------------------------
    counts = collections.defaultdict(int)
    num_transactions = 0
    for transaction in transactions():
        num_transactions += 1
        for item in transaction:
            counts[item] += 1

    large_itemsets = [(i, c) for (i, c) in counts.items() if 
                      (c / num_transactions) >= min_support]
    
    
    #large_itemsets = collections.Counter(i for t in transactions() for i in t)
    #large_itemsets = [(i, c) for (i, c) in large_itemsets.items() 
    #                  if c >= min_support]
    
    #print(large_itemsets)
    # If large itemsets were found, convert to dictionary
    if large_itemsets:
        large_itemsets = {1: {(i, ):c for (i, c) in sorted(large_itemsets)}}
        
    # No large itemsets were found, return immediately
    else: return {}

    # STEP 2 - Build up the size of the itemsets
    # ------------------------------------------
    
    # While there are itemsets of the previous size
    k = 2
    while large_itemsets[k - 1]:
        
        # Generate candidate itemsets of length k from length k - 1 itemsets
        itemsets_list = list(large_itemsets[k - 1].keys())
        #print('->', large_itemsets[k - 1])
        #print('-->', itemsets_list)
    
        possible_extensions = join_step(itemsets_list)
        C_k = list(prune_step(itemsets_list, possible_extensions))
        
        # Create a copy with sets for faster subset checks in the transactions
        C_k_sets = [set(itemset) for itemset in C_k]
        
        # Prepare counts
        candidate_itemset_counts = collections.defaultdict(int)
        for row, transaction in enumerate(transactions()):
            if not use_transaction[row]:
                continue
            
            found_any = False
            for candidate, candidate_set in zip(C_k, C_k_sets):
                
                # This is where most of the time is spent in the algorithm
                # TODO : Look into hash trees to speed this up
                if set.issubset(candidate_set, transaction):
                    candidate_itemset_counts[candidate] += 1
                    found_any = True
                    
            if not found_any:
                use_transaction[row] = False
            
        
        C_k = [(i, c) for (i, c) in candidate_itemset_counts.items() 
               if (c / num_transactions) >= min_support]
        
        # If no candidate itemsets were found, break out of the loop
        if not C_k:
            break
        
        # Candidate itemsets were found, add them and progress the while-loop
        large_itemsets[k] = {i:c for (i, c) in sorted(C_k)}
        k += 1
        
    return large_itemsets


if __name__ == '__main__':
    import pytest
    pytest.main(args=['.', '--doctest-modules', '-v'])
    #pytest.main(args=['tests', '--doctest-modules', '-v'])

def test_speed():
    import random
    import time
    random.seed(123)
    
    def generate_transactions(num_transactions, unique_items, items_row=(1, 100)):
        """
        Generate synthetic transactions.
        """
        
        items = list(range(unique_items))
        
        for transaction in range(num_transactions):
            items_this_row = random.randint(*items_row)
            yield random.sample(items, k=min(unique_items, items_this_row))
          
        
    trans = generate_transactions(500, 25, items_row=(1, 10))
    st = time.perf_counter()
    itemsets_from_transactions(trans, min_support=2/500)
    print(f'Test ran in {round(time.perf_counter() - st,4)} s.')
    
if __name__ == '__main__':
    test_speed()
    
    

        
    
    
    