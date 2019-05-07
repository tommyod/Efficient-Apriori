# Efficient-Apriori [![Build Status](https://travis-ci.com/tommyod/Efficient-Apriori.svg?branch=master)](https://travis-ci.com/tommyod/Efficient-Apriori) [![PyPI version](https://badge.fury.io/py/efficient-apriori.svg)](https://pypi.org/project/efficient-apriori/) [![Documentation Status](https://readthedocs.org/projects/efficient-apriori/badge/?version=latest)](https://efficient-apriori.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/efficient-apriori)](https://pepy.tech/project/efficient-apriori) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

An efficient pure Python implementation of the Apriori algorithm. Works with Python 3.6+.

The apriori algorithm uncovers hidden structures in categorical data.
The classical example is a database containing purchases from a supermarket.
Every purchase has a number of items associated with it.
We would like to uncover association rules such as `{bread, eggs} -> {bacon}` from the data.
This is the goal of [association rule learning](https://en.wikipedia.org/wiki/Association_rule_learning), and the [Apriori algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm) is arguably the most famous algorithm for this problem.
This repository contains an efficient, well-tested implementation of the apriori algorithm as descriped in the [original paper](https://www.macs.hw.ac.uk/~dwcorne/Teaching/agrawal94fast.pdf) by Agrawal et al, published in 1994.

## Example

Here's a minimal working example.
Notice that in every transaction with `eggs` present, `bacon` is present too.
Therefore, the rule `{eggs} -> {bacon}` is returned with 100 % confidence.

```python
from efficient_apriori import apriori
transactions = [('eggs', 'bacon', 'soup'),
                ('eggs', 'bacon', 'apple'),
                ('soup', 'bacon', 'banana')]
itemsets, rules = apriori(transactions, min_support=0.5,  min_confidence=1)
print(rules)  # [{eggs} -> {bacon}, {soup} -> {bacon}]
```
More examples are included below.

## Installation

The software is available through GitHub, and through [PyPI](https://pypi.org/project/efficient-apriori/).
You may install the software using `pip`.

```bash
pip install efficient-apriori
```

## Contributing

You are very welcome to scrutinize the code and make pull requests if you have suggestions and improvements.
Your submitted code must be PEP8 compliant, and all tests must pass.

## More examples

### Filtering and sorting association rules

It's possible to filter and sort the returned list of association rules.

```python
from efficient_apriori import apriori
transactions = [('eggs', 'bacon', 'soup'),
                ('eggs', 'bacon', 'apple'),
                ('soup', 'bacon', 'banana')]
itemsets, rules = apriori(transactions, min_support=0.2,  min_confidence=1)

# Print out every rule with 2 items on the left hand side,
# 1 item on the right hand side, sorted by lift
rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule) # Prints the rule and its confidence, support, lift, ...
```

### Working with large datasets

If you have data that is too large to fit into memory, you may pass a function returning a generator instead of a list.
The `min_support` will most likely have to be a large value, or the algorithm will take very long before it terminates.
If you have massive amounts of data, this Python implementation is likely not fast enough, and you should consult more specialized implementations.

```python
def data_generator(filename):
  """
  Data generator, needs to return a generator to be called several times.
  """
  def data_gen():
    with open(filename) as file:
      for line in file:
        yield tuple(k.strip() for k in line.split(','))      

  return data_gen

transactions = data_generator('dataset.csv')
itemsets, rules = apriori(transactions, min_support=0.9, min_confidence=0.6)
```
