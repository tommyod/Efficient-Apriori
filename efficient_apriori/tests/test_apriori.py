#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

import pytest
from efficient_apriori.apriori import apriori
from efficient_apriori.rules import Rule


def test_against_R_implementation_1():
    """
    The following R-code was used:

    > install.packages("arules")
    > col1 = c("a", "a", "a", "b", "b", "b", "b")
    > col2 = c("c", "c", "d", "d", "d", "c", "c")
    > col3 = c("e", "e", "e", "e", "f", "f", "f")
    > df = data.frame(col1, col2, col3)
    > df <- data.frame(sapply(df, as.factor))
    > rules <- apriori(df, parameter = list(supp = 0.2, conf = 0.2))
    > inspect(head(rules, by = "confidence"))
    """

    transactions = [
        ("a", "c", "e"),
        ("a", "c", "e"),
        ("a", "d", "e"),
        ("b", "d", "e"),
        ("b", "d", "f"),
        ("b", "c", "f"),
        ("b", "c", "f"),
    ]

    itemsets, rules = apriori(transactions, 0.2, 0.2)

    assert Rule(("a",), ("e",)) in rules

    for rule in rules:
        if rule == Rule(("a",), ("e",)):
            assert abs(rule.support - 0.4285714) < 10e-7
            assert rule.confidence == 1

        if rule == Rule(("c", "e"), ("a",)):
            assert abs(rule.support - 0.2857143) < 10e-7
            assert rule.confidence == 1

        if rule == Rule(("e",), ("a",)):
            assert abs(rule.support - 0.4285714) < 10e-7
            assert rule.confidence == 3 / 4


def test_against_R_implementation_2():
    """
    The following R-code was used:

    > install.packages("arules")
    > col1 = c("b", "b", "c", "b", "a", "a", "b", "c", "b", "b", "a", "b", "a",
    "a", "a", "c", "b", "a", "b", "b", "b", "c", "a", "c", "a", "a", "c", "a",
    "b", "b", "a", "c")
    > col2 = c("e", "f", "e", "e", "f", "e", "d", "f", "e", "e", "e", "d", "e",
    "e", "f", "d", "d", "d", "e", "f", "f", "d", "d", "f", "e", "e", "f", "f",
    "f", "d", "e", "e")
    > col3 = c("g", "i", "j", "i", "i", "j", "i", "h", "g", "j", "g", "h", "i",
    "h", "g", "h", "g", "j", "h", "i", "g", "g", "i", "h", "h", "h", "h", "g",
    "j", "i", "g", "g")
    > df = data.frame(col1, col2, col3)
    > df <- data.frame(sapply(df, as.factor))
    > rules <- apriori(df, parameter = list(supp = 0.2, conf = 0.2))
    > inspect(head(rules, by = "confidence"))
    """

    transactions = [
        ("b", "e", "g"),
        ("b", "f", "i"),
        ("c", "e", "j"),
        ("b", "e", "i"),
        ("a", "f", "i"),
        ("a", "e", "j"),
        ("b", "d", "i"),
        ("c", "f", "h"),
        ("b", "e", "g"),
        ("b", "e", "j"),
        ("a", "e", "g"),
        ("b", "d", "h"),
        ("a", "e", "i"),
        ("a", "e", "h"),
        ("a", "f", "g"),
        ("c", "d", "h"),
        ("b", "d", "g"),
        ("a", "d", "j"),
        ("b", "e", "h"),
        ("b", "f", "i"),
        ("b", "f", "g"),
        ("c", "d", "g"),
        ("a", "d", "i"),
        ("c", "f", "h"),
        ("a", "e", "h"),
        ("a", "e", "h"),
        ("c", "f", "h"),
        ("a", "f", "g"),
        ("b", "f", "j"),
        ("b", "d", "i"),
        ("a", "e", "g"),
        ("c", "e", "g"),
    ]

    itemsets, rules = apriori(transactions, 0.2, 0.2)

    for rule in rules:
        if rule == Rule(("a",), ("e",)):
            assert abs(rule.support - 0.21875) < 10e-7
            assert abs(rule.confidence - 0.5833333) < 10e-7

        if rule == Rule(("e",), ("a",)):
            assert abs(rule.support - 0.21875) < 10e-7
            assert abs(rule.confidence - 0.5000000) < 10e-7


def test_against_R_implementation_3():
    """
    The following R-code was used:

    > install.packages("arules")
    > col1 = c("b", "b", "c", "a", "b", "b", "a", "a", "b", "b", "a", "a", "c",
    "b", "a", "c")
    > col2 = c("e", "d", "e", "e", "e", "e", "d", "e", "e", "e", "d", "e", "e",
    "e", "d", "e")
    > col3 = c("i", "g", "h", "j", "i", "g", "h", "j", "i", "g", "j", "i", "j",
    "j", "i", "i")
    > df = data.frame(col1, col2, col3)
    > df <- data.frame(sapply(df, as.factor))
    > rules <- apriori(df, parameter = list(supp = 0.2, conf = 0.2))
    > inspect(head(rules, by = "confidence"))
    """

    transactions = [
        ("b", "e", "i"),
        ("b", "d", "g"),
        ("c", "e", "h"),
        ("a", "e", "j"),
        ("b", "e", "i"),
        ("b", "e", "g"),
        ("a", "d", "h"),
        ("a", "e", "j"),
        ("b", "e", "i"),
        ("b", "e", "g"),
        ("a", "d", "j"),
        ("a", "e", "i"),
        ("c", "e", "j"),
        ("b", "e", "j"),
        ("a", "d", "i"),
        ("c", "e", "i"),
    ]

    itemsets, rules = apriori(transactions, 0.2, 0.2)

    for rule in rules:
        if rule == Rule(("b",), ("e",)):
            assert abs(rule.support - 0.3750) < 10e-7
            assert abs(rule.confidence - 0.8571429) < 10e-7

        if rule == Rule(("i",), ("e",)):
            assert abs(rule.support - 0.3125) < 10e-7
            assert abs(rule.confidence - 0.8333333) < 10e-7

        if rule == Rule(("j",), ("e",)):
            assert abs(rule.support - 0.2500) < 10e-7
            assert abs(rule.confidence - 0.8000000) < 10e-7

        if rule == Rule(("e",), ("b",)):
            assert abs(rule.support - 0.3750) < 10e-7
            assert abs(rule.confidence - 0.5000000) < 10e-7


def test_minimal_input():
    """
    The with some minimal inputs, and make sure the correct errors are raised.
    """
    transactions = []
    itemsets, rules = apriori(transactions, 0.2, 0.2)
    assert itemsets == {} and rules == []

    with pytest.raises(ValueError):
        itemsets, rules = apriori(transactions, -0.2, 0.2)

    with pytest.raises(ValueError):
        itemsets, rules = apriori(transactions, 0.2, -0.2)

    with pytest.raises(ValueError):
        itemsets, rules = apriori(transactions, "asdf", 1)

    itemsets, rules = apriori([(1, 2), (1, 2), (1, 3)], 1, 1)
    itemsets, rules = apriori([(1, 2), (1, 2), (1, 3)], 1.0, 1.0)


if __name__ == "__main__":
    pytest.main(args=[".", "--doctest-modules", "-v"])
