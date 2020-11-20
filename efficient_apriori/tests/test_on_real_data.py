#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

import pytest
import os
from efficient_apriori.apriori import apriori
from efficient_apriori.rules import Rule


def test_adult_dataset():
    """
    Test on the Adult dataset, which may be found here:
        https://archive.ics.uci.edu/ml/datasets/adult

    Some numeric columns were removed. The age was discretized.
    The purpose of this test is to assure that the algorithm can deal with a
    small 2.2 MB (30k rows) data set reasonably efficiently.

    Test against R, from the following code
    > library(arules)
    > df <- read.csv("adult_data_cleaned.txt", header = FALSE)
    > df <- data.frame(sapply(df, as.factor))
    > rules <- apriori(df, parameter = list(supp = 0.4, conf = 0.4))
    > inspect(head(rules, by = "confidence"m = 10))

    """

    def data_generator(filename):
        """
        Data generator, needs to return a generator to be called several times.
        """

        def data_gen():
            with open(filename) as file:
                for line in file:
                    yield tuple(k.strip() for k in line.split(","))

        return data_gen

    try:
        base, _ = os.path.split(__file__)
        filename = os.path.join(base, "adult_data_cleaned.txt")
    except NameError:
        filename = "adult_data_cleaned.txt"
    transactions = data_generator(filename)
    itemsets, rules = apriori(transactions, min_support=0.2, min_confidence=0.2)

    # Test that the rules found in R were also found using this implementation
    rules_set = set(rules)
    assert Rule(("Married-civ-spouse", "Husband", "middle-aged"), ("Male",)) in rules_set
    assert (
        Rule(
            ("Married-civ-spouse", "White", "middle-aged", "Male"),
            ("Husband",),
        )
        in rules_set
    )
    assert Rule(("<=50K", "young"), ("Never-married",)) in rules_set
    assert (
        Rule(
            ("Husband", "White", "Male", "middle-aged"),
            ("Married-civ-spouse",),
        )
        in rules_set
    )
    assert Rule(("young",), ("Never-married",)) in rules_set

    # Test results against R package arules
    for rule in rules:
        if rule == Rule(("Married-civ-spouse", "Husband", "middle-aged"), ("Male",)):
            assert abs(rule.support - 0.2356193) < 10e-7
            assert abs(rule.confidence - 0.9998697) < 10e-7
            assert abs(rule.lift - 1.494115) < 10e-7

        if rule == Rule(
            ("Married-civ-spouse", "White", "middle-aged", "Male"),
            ("Husband",),
        ):
            assert abs(rule.support - 0.2123399) < 10e-7
            assert abs(rule.confidence - 0.9938192) < 10e-7
            assert abs(rule.lift - 2.452797) < 10e-7

        if rule == Rule(("<=50K", "young"), ("Never-married",)):
            assert abs(rule.support - 0.2170081) < 10e-7
            assert abs(rule.confidence - 0.7680435) < 10e-7
            assert abs(rule.lift - 2.340940) < 10e-7

        if rule == Rule(
            ("Husband", "White", "Male", "middle-aged"),
            ("Married-civ-spouse",),
        ):
            assert abs(rule.support - 0.2123399) < 10e-7
            assert abs(rule.confidence - 0.9995663) < 10e-7
            assert abs(rule.lift - 2.173269) < 10e-7

        if rule == Rule(("young",), ("Never-married",)):
            assert abs(rule.support - 0.2200792) < 10e-7
            assert abs(rule.confidence - 0.7379261) < 10e-7
            assert abs(rule.lift - 2.249144) < 10e-7


if __name__ == "__main__":
    pytest.main(args=[".", "--doctest-modules", "-v"])
