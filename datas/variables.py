

import numpy as np
import warnings
from errors import BadValueWarning, ExclusionException


class Variable(object):

    def __init__(self, colname, translators={}):
        self.colname = colname
        self.translators = translators

    def build(self, attr_name, row, parent):
        self.parent = parent
        init_val = row[self.colname]
        if init_val in self.translators:
            self.value = self.translators[init_val]
        else:
            self.value = init_val
        self.post_build(attr_name)
        return self.value

    def post_build(self):
        NotImplemented


class FloatVariable(Variable):

    def __init__(self, colname, translators={}, bounds=(-np.inf, np.inf)):
        super(FloatVariable, self).__init__(colname, translators)

    def post_build(self, name):
        try:
            self.value = np.float64(self.value)
        except ValueError:
            try:
                assert np.isnan(self.value)
            except:
                warnings.warn(
                    BadValueWarning(self.colname, self, self.value)
                )
                self.value = np.nan
        self.parent.numerics[name] = self.value


class IntegerVariable(Variable):

    def __init__(self, colname, translators={}, bounds=(-np.inf, np.inf)):
        super(IntegerVariable, self).__init__(colname, translators)

    def post_build(self, name):
        try:
            self.value = np.int64(float(self.value))
        except ValueError:
            try:
                assert np.isnan(self.value)
            except:
                warnings.warn(
                    BadValueWarning(self.colname, self, self.value)
                )
                self.value = np.nan
        self.parent.numerics[name] = self.value


class FactorVariable(Variable):

    def __init__(self, colname, translators={}, delimiter=" "):
        self.delimiter = delimiter
        super(FactorVariable, self).__init__(colname, translators)

    def post_build(self, name):
        if not self.delimiter is None:
            self.value = set([
                '.'.join([name, v]) for v in
                self.value.split(self.delimiter) if v
            ])
        else:
            self.value = set(['.'.join([name, self.value])])
        self.parent.factors |= self.value


class GroupingVariable(Variable):

    def __init__(self, colname, translators={}):
        super(GroupingVariable, self).__init__(colname, translators)

    def post_build(self, name):
        if self.value:
            self.parent.groups[name] = self.value


class ExclusionVariable(Variable):

    def __init__(self, colname, function, translators={}):
        self.check_function = function
        super(ExclusionVariable, self).__init__()

    def post_build(self, name):
        result = self.check_function(self.value)
        if result is True:
            raise ExclusionException()
        elif result is False:
            pass
        else:
            raise AttributeError(
                "Exclusion variable check function must return True or False")


class IdentityVariable(Variable):

    def __init__(self, colname, translators={}):
        super(IdentityVariable, self).__init__(colname, translators)

    def post_build(self, name):
        self.parent.identity[name] = self.value