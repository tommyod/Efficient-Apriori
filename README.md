# Efficient-Apriori [![Build Status](https://travis-ci.com/tommyod/Efficient-Apriori.svg?branch=master)](https://travis-ci.com/tommyod/Efficient-Apriori)

An efficient Python implementation of the Apriori algorithm.

The apriori algorithm finds hidden structures in data.
The classical example is a database containing purchases from a supermarket.
Every receipt has a number of items associated with it.
We would like to uncover association rules such as `{bread, eggs} -> {bacon}`, so we can advertise and stock our supermarket in an advantageous manner.
This is the goal of [association rule learning](https://en.wikipedia.org/wiki/Association_rule_learning), and the [Apriori algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm) is arguably the most famous algorithm for this problem.
This repository contains an efficient, well-tested implementation of the apriori algorithm as descriped in the [original paper](https://www.macs.hw.ac.uk/~dwcorne/Teaching/agrawal94fast.pdf) by Agrawal et al, published in 1994.

## Example

Here's a minimal working example.
Notice that in every transaction with `a` present, `b` is present too.
Therefore, the rule `{a} -> {b}` is returned with 100 % confidence.

```python
from efficient_apriori import apriori
transactions = [('a', 'b', 'c'),
                ('a', 'b', 'd'),
                ('f', 'b', 'g')]
itemsets, rules = apriori(transactions, min_confidence=1)
print(rules)  # [{a} -> {b}]
```

## Installation

Here's how to install from GitHub.

```bash
git clone https://github.com/tommyod/Efficient-Apriori.git
cd Efficient-Apriori
pip install .
```

## Contributing

Pull requests are very welcome.

## More examples

```python
# Example showing confidence, support, itemsets and rules
```

```python
# Example showing how to work with large files
```

## Todo

- [ ] Check if the rule algo must return duplicates
- [ ] Implement faster rule algo
- [ ] Add type hints to every function
- [X] Add tests against R package
- [ ] Consider aprioriTID implementation
