#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 08:42:18 2021
@author: tommy

MIN_SUPPORTS [0.1, 0.05, 0.01]
MAX_LENGTHS [1, 2, 3, 4, 5]
APYRIO: Average time on file 'adult_data_cleaned.txt' was: 0.337
APYRIO: Average time on file 'online-retail.txt' was: 0.725
EFF_AP: Average time on file 'adult_data_cleaned.txt' was: 0.371
EFF_AP: Average time on file 'online-retail.txt' was: 0.736


MIN_SUPPORTS [0.1, 0.05, 0.01]
MAX_LENGTHS [1, 2, 3, 4, 5]
APYRIO: Average time on file 'adult_data_cleaned.txt' was: 0.332
APYRIO: Average time on file 'online-retail.txt' was: 0.714
EFF_AP: Average time on file 'adult_data_cleaned.txt' was: 0.255
EFF_AP: Average time on file 'online-retail.txt' was: 0.7
"""

import itertools
import time
import statistics
import pytest
from efficient_apriori import itemsets_from_transactions


def yield_data(filename):
    with open(filename) as file:
        for line in file:
            yield set(k.strip() for k in line.split(","))


FILENAMES = ["adult_data_cleaned.txt", "online-retail.txt"]
MIN_SUPPORTS = [0.1, 0.05, 0.01]
MAX_LENGTHS = [1, 2, 3, 4, 5]


@pytest.mark.skip(reason="Timing is skipped.")
def test_times_efficient_apriori():
    for filename in FILENAMES:
        # transactions = list(yield_data(filename))

        times = []

        for min_support, max_length in itertools.product(MIN_SUPPORTS, MAX_LENGTHS):
            transactions = list(yield_data(filename))
            start_time = time.perf_counter()
            large_itemsets, num_transactions = itemsets_from_transactions(
                transactions, min_support=min_support, max_length=max_length
            )
            total_time = round(time.perf_counter() - start_time, 3)
            times.append(total_time)
            # print(filename, min_support, max_length, total_time)

        avg_time = round(statistics.mean(times), 3)
        print(f"EFF_AP: Average time on file '{filename}' was: {avg_time}")


@pytest.mark.skip(reason="Timing is skipped.")
def test_times_apyriori():
    from apyori import gen_support_records, TransactionManager

    for filename in FILENAMES:
        # transactions = list(yield_data(filename))

        times = []

        for min_support, max_length in itertools.product(MIN_SUPPORTS, MAX_LENGTHS):
            transactions = list(yield_data(filename))
            start_time = time.perf_counter()
            transaction_manager = TransactionManager.create(transactions)
            list(gen_support_records(transaction_manager, min_support, max_length=max_length))
            total_time = round(time.perf_counter() - start_time, 3)
            times.append(total_time)
            # print(filename, min_support, max_length, total_time)

        avg_time = round(statistics.mean(times), 3)
        print(f"APYRIO: Average time on file '{filename}' was: {avg_time}")


if __name__ == "__main__":
    print("MIN_SUPPORTS", MIN_SUPPORTS)
    print("MAX_LENGTHS", MAX_LENGTHS)
    test_times_apyriori()
    test_times_efficient_apriori()
