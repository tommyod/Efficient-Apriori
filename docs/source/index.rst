Efficient-Apriori
=================

An efficient pure Python implementation of the Apriori algorithm.

Overview
--------

An efficient pure Python implementation of the Apriori algorithm.
Created for Python 3.6 and 3.7.

The apriori algorithm uncovers hidden structures in categorical data.
The classical example is a database containing purchases from a supermarket.
Every purchase has a number of items associated with it.
We would like to uncover association rules such as `{bread, eggs} -> {bacon}`
from the data. This is the goal of `association rule
learning <https://en.wikipedia.org/wiki/Association_rule_learning>`_, and the
`Apriori algorithm <https://en.wikipedia.org/wiki/Apriori_algorithm>`_ is
arguably the most famous algorithm for this problem. This project contains an
efficient, well-tested implementation of the apriori algorithm as descriped in
the  `original paper <https://www.macs.hw.ac.uk/~dwcorne/Teaching/agrawal94fast.pdf>`_
by Agrawal et al, published in 1994.

Installation
------------

The package is distributed on `PyPI <https://pypi.org/project/efficient-apriori/>`_.
From your terminal, simply run the following command to install the package.

::

    $ pip install efficient-apriori

Notice that the name of the package is ``efficient-apriori`` on PyPI, while it's
imported as ``import efficient_apriori``.

A minimal working example
-------------------------

.. py:currentmodule:: efficient_apriori

Here's a minimal working example.
Notice that in every transaction with `eggs` present, `bacon` is present too.
Therefore, the rule `{eggs} -> {bacon}` is returned with 100 % confidence.

.. code-block:: python

    from efficient_apriori import apriori
    transactions = [('eggs', 'bacon', 'soup'),
                    ('eggs', 'bacon', 'apple'),
                    ('soup', 'bacon', 'banana')]
    itemsets, rules = apriori(transactions, min_support=0.5,  min_confidence=1)
    print(rules)  # [{eggs} -> {bacon}, {soup} -> {bacon}]

See the API documentation for the full signature of :func:`~efficient_apriori.apriori`.
More examples are included below.

More examples
-------------

Filtering and sorting association rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's possible to filter and sort the returned list of association rules.

.. code-block:: python

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


Working with large datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have data that is too large to fit into memory, you may pass a function
returning a generator instead of a list. The `min_support` will most likely
have to be a large value, or the algorithm will take very long before it
terminates. If you have massive amounts of data, this Python implementation is
likely not fast enough, and you should consult more specialized implementations.

.. code-block:: python

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


Contributing
------------

You are very welcome to scrutinize the code and make pull requests if you have
suggestions for improvements. Your submitted code must be PEP8 compliant, and
all tests must pass.
See `tommyod/Efficient-Apriori <https://github.com/tommyod/Efficient-Apriori>`_
on GitHub for more information.

API documentation
-----------------

Although the Apriori algorithm uses many sub-functions, only three functions
are likely of interest to the reader. The :func:`~efficient_apriori.apriori`
returns both the itemsets and the association rules, which is obtained by calling
:func:`~efficient_apriori.itemsets_from_transactions`
and
:func:`~efficient_apriori.generate_rules_apriori`, respectively.
The rules are returned as instances of the :class:`~efficient_apriori.Rule` class,
so reading up on it's basic methods might be useful.

Apriori function
~~~~~~~~~~~~~~~~

.. autofunction:: apriori

Itemsets function
~~~~~~~~~~~~~~~~~

.. autofunction:: itemsets_from_transactions

Association rules function
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: generate_rules_apriori

Rule class
~~~~~~~~~~

.. autoclass:: Rule
