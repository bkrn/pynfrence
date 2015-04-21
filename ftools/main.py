# -*- coding: utf-8 -*-

import itertools


def powerset(iterable):
    s = list(iterable)
    for pset in itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    ):
        if pset:
            yield pset
        else:
            pass