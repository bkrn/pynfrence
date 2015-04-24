# -*- coding: utf-8 -*-


class ExclusionException(Exception):

    def __init__(self):
        super(ExclusionException, self).__init__()


class BadValueWarning(RuntimeWarning):

    def __init__(self, colname, vartype, value):
        super(BadValueWarning, self).__init__()
        self.value, self.colname, self.vartype = value, colname, vartype

    def __str__(self):
        return "%s (%s) got value %s and set to np.nan" % (
            self.colname, self.vartype.__class__.__name__, self.value
        )