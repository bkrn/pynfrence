# -*- coding: utf-8 -*-

import csv
import cPickle

import numpy as np
from sklearn import linear_model

from errors import ExclusionException
from group import Group


class DataSet(object):

    def __init__(self, row_generator, dataclass):
        self.groups = {}  # Dictionary colname: groupid: Group
        self.numerics = {}  # Dictionary attr: np.array(floats|integers)
        self.factors = {}  # Dictionary factor: integer (for factors occurence)
        self.datas = []
        for row in row_generator():
            try:
                self.datas.append(dataclass(row, self))
            except ExclusionException:
                pass
        for attr, li in self.numerics.iteritems():
            self.numerics[attr] = np.nanmean(li)
        for column, group_dict in self.groups.iteritems():
            for groupid, set_of_datas in group_dict.iteritems():
                self.groups[column][groupid] = Group(set_of_datas, self)
        self.datas = np.array(self.datas)
        self.sorted_factors = sorted(self.factors.keys())

    @classmethod
    def load_csv(cls, file_path, dataclass):

        def row_generator():
            with open(file_path, 'rb') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=',', quotechar='"')
                for row in reader:
                    yield row

        return cls(row_generator, dataclass)

    @classmethod
    def load_pandas(cls, dataframe, dataclass):

        def row_generator():
            for ind in dataframe.index:
                yield dict(dataframe.loc[ind])

        return cls(row_generator, dataclass)

    @classmethod
    def load_sqlite(cls, CON, table, dataclass):

        def row_generator():
            cur = CON.execute('SELECT * FROM %s;' % table)
            cols = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            while not row is None:
                yield dict(zip(cols, row))
                row = cur.fetchone()

        return cls(row_generator, dataclass)

    @classmethod
    def from_pickle(cls, file_path):
        with open(file_path, 'rb') as f:
            return cPickle.load(f)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            return cPickle.dump(self, f)

    def linear_model(self, model_array):
        arr = np.array([data.get_row(model_array) for data in self.datas])
        arr = arr[~np.isnan(arr).any(axis=1)]
        r = linear_model.Ridge()
        y, x = arr[:, 0], arr[:, 1:]
        r.fit(x, y)
        print r.score(x, y)