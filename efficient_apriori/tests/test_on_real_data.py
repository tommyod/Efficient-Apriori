#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

import pytest
from efficient_apriori.apriori import apriori


def test_adult_dataset():
    """
    Test on the Adult dataset, which may be found here:
        https://archive.ics.uci.edu/ml/datasets/adult
        
    Some numeric columns were removed. The age was discretized.
    The purpose of this test is to assure that the algorithm can deal with a
    small 2.2 MB (30k rows) data set reasonably efficiently.
    """
    
    def data_generator(filename):
        """
        Data generator, needs to return a generator to be called several times.
        """
        def data_gen():
            with open(filename) as file:
                for line in file:
                    yield tuple(k.strip() for k in line.split(','))      
        return data_gen

    filename = 'adult_data_cleaned.txt'
    transactions = data_generator(filename)
    itemsets, rules = apriori(transactions, min_support=0.4, 
                              min_confidence=0.4)
    
    
if __name__ == '__main__':
    pytest.main(args=['.', '--doctest-modules', '-v']) 