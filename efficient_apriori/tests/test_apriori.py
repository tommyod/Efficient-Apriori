#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

import pytest
from efficient_apriori.apriori import apriori
from efficient_apriori.rules import Rule
from efficient_apriori.itemsets import ItemsetCount


def test_api():
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

    assert itemsets[1] == {("a",): 3, ("c",): 4, ("e",): 4, ("d",): 3, ("b",): 4, ("f",): 3}
    assert all(isinstance(rule, Rule) for rule in rules)

    for count, itemsets_dict in itemsets.items():
        assert isinstance(itemsets_dict, dict)
        for itemset, count in itemsets_dict.items():
            actual_count = sum(1 if set(itemset).issubset(set(trans)) else 0 for trans in transactions)
            assert count == actual_count

    itemsets, rules = apriori(transactions, 0.2, 0.2, output_transaction_ids=True)
    for count, itemsets_dict in itemsets.items():
        assert isinstance(itemsets_dict, dict)
        for itemset, counter in itemsets_dict.items():
            assert isinstance(counter, ItemsetCount)

            actual_count = sum(1 if set(itemset).issubset(set(trans)) else 0 for trans in transactions)
            assert counter.itemset_count == actual_count


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


def test_iterator_input():
    """
    Minimal test using transactions from iterators.
    """
    empty_iterator = iter(())
    transactions = empty_iterator
    itemsets, rules = apriori(transactions, 0.2, 0.2)
    assert itemsets == {} and rules == []

    transactions = [(1, 2), (1, 2), (1, 3), (1, 4), (1, 3)]
    transactions_iter = iter(transactions)
    itemsets1, rules1 = apriori(transactions_iter, 0.2, 1)
    itemsets2, rules2 = apriori(transactions, 0.2, 1)
    assert len(rules1) == len(rules2)
    for i in range(len(rules1)):
        assert rules1[i] == rules2[i]


def test_empty_H_1():
    """
    An example of the case where there are itemsets without any Rule with
    single item in right hand side that satifies the required minimum confidence.
    The issue is raised in #57.
    """
    # The results are received from commit 01d174379c51758aa2f6d2926b473124928dc631
    true_itemsets_raw = {1: {(1,): 4, (2,): 5, (3,): 4}, 2: {(1, 2): 4, (1, 3): 3, (2, 3): 4}, 3: {(1, 2, 3): 3}}
    true_rules = [
        Rule((2,), (1,), 4, 5, 4, 5),
        Rule((1,), (2,), 4, 4, 5, 5),
        Rule((3,), (2,), 4, 4, 5, 5),
        Rule((2,), (3,), 4, 5, 4, 5),
        Rule((1, 3), (2,), 3, 3, 5, 5),
    ]

    transactions = [(1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2), (2, 3)]
    itemsets_raw, rules = apriori(transactions, 0.4, 0.8)

    assert itemsets_raw == true_itemsets_raw
    assert all(rule == true_rule for rule, true_rule in zip(rules, true_rules))


if __name__ == "__main__":
    pytest.main(args=[".", "--doctest-modules", "-v"])
