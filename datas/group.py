# -*- coding: utf-8 -*-


import numpy as np


class Group(object):

    def __init__(self, datas, dataset):
        self.datas = datas
        self.size = len(datas)
        self.numerics = {}
        self.factors = {}
        self.dataset = dataset
        for data in datas:
            for attr, val in data.numerics.iteritems():
                if not attr in self.numerics:
                    self.numerics[attr] = []
                if not np.isnan(np.float64(val)):
                    self.numerics[attr].append(val)
            for factor in dataset.factors:
                if not factor in self.factors:
                    self.factors[factor] = 0
                self.factors[factor] += int(factor in data.factors)
        for attr, li in self.numerics.iteritems():
            if li:
                self.numerics[attr] = (
                    np.nanmin(li), np.nanmean(li), np.nanmax(li)
                )
            else:
                self.numerics[attr] = (np.nan, np.nan, np.nan)
        for factor, n in self.factors.iteritems():
            self.factors[factor] = (n / float(self.size), int(bool(n)))

    def __add__(self, other):
        return Group((self.datas & other.datas))

    def get_val(self, attr, agg_func):
        if attr in self.numerics:
            funcs = ('min', 'mean', 'max')
            d = self.numerics
        elif attr in self.factors:
            funcs = ('mean', 'exists')
            d = self.factors
        else:
            raise AttributeError()
        if not np.isnan(d[attr]).any():
            return d[attr][funcs.index(agg_func)]
        else:
            return self.dataset.numerics[attr]
