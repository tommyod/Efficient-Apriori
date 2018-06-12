#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:53:10 2018

@author: tommy



https://www.kaggle.com/c/instacart-market-basket-analysis/data
"""

from efficient_apriori.itemsets import itemsets_from_transactions

def generate_rules_naively():
    """
    Generate rules naively, for testing.
    """
    pass

def genrules():
    """
    The naive algorithm from the original paper.
    """
    pass
    
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
                    'cbf']