

import numpy as np

from variables import Variable


class Data(object):

    def __init__(self, row, dataset):
        self.dataset = dataset
        self.numerics = {}
        self.groups = {}
        self.identity = {}
        self.factors = set()
        for attr_name, variable in self.var_generator():
            variable.build(attr_name, row, self)
        self.write_to_dataset()

    def var_generator(self):
        for key, item in self.__class__.__dict__.iteritems():
            if isinstance(item, Variable):
                yield key, item

    def write_to_dataset(self):
        for attr, val in self.numerics.iteritems():
            if not attr in self.dataset.numerics:
                self.dataset.numerics[attr] = []
            self.dataset.numerics[attr].append(val)
        for group_type, group in self.groups.iteritems():
            if not group_type in self.dataset.groups:
                self.dataset.groups[group_type] = {}
            if not group in self.dataset.groups[group_type]:
                self.dataset.groups[group_type][group] = set()
            self.dataset.groups[group_type][group].add(self)
        for factor in self.factors:
            if not factor in self.dataset.factors:
                self.dataset.factors[factor] = 0
            self.dataset.factors[factor] += 1

    def _get_var(self, name):
        if name in self.numerics:
            var_val = self.numerics[name]
        elif name in self.dataset.factors:
            var_val = np.int64((name in self.factors))
        elif name.endswith('()'):
            var_val = eval('self.' + name)
        elif '__' in name:
            group, attr, agg_func = tuple(name.split('__'))
            if group in self.groups:
                groupid = self.groups[group]
                var_val = self.dataset.groups[group][groupid].get_val(
                    attr,
                    agg_func
                )
            else:
                var_val = self.dataset.numerics[attr]
        else:
            raise AttributeError('%s does not exist in data' % name)
        return var_val

    def get_row(self, attrs):
        row = []
        for attr in attrs:
            if isinstance(attr, str):
                if attr.startswith('~'):
                    factor_type = attr[1:]
                    for factor in self.dataset.sorted_factors:
                        if factor.split('.')[0] == (factor_type):
                            row.append(self._get_var(factor))
                else:
                    row.append(self._get_var(attr))
            elif isinstance(attr, tuple):
                func, attr1, attr2 = attr
                row.append(func(self._get_var(attr1), self._get_var(attr2)))
        return np.array(row)
