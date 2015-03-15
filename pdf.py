import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import exp, sqrt, pi


class PDF(object):

    def __init__(self, series, start=None, stop=None, bw=.1):
        if not type(series) == pd.Series:
            series = pd.Series(series)
        series = series[series.notnull()]
        self.name = series.name
        self.size = series.count()
        self.values = series.unique()
        self.bw = bw

    def _estimate(self):
        NotImplemented

    def plot(self):
        x = np.array(range(self.ends[0] - 1, self.ends[1] + 1))
        y = [0] + list(self.pdf) + [0]
        plt.step(x, y)
        plt.show()

    def prob(self, li):
        NotImplemented

    def verify_slice(self, li):
        NotImplemented

    @classmethod
    def make(cls, series, pdf_type=None, start=None, stop=None):
        if pdf_type:
            if pdf_type == 'continuous':
                obj = ContinuousPDF(series, start, stop)
            elif pdf_type == 'discrete':
                obj = DiscretePDF(series, start, stop)
            elif pdf_type == 'indexed':
                obj = IndexedPDF(series)
            else:
                raise Exception('%s unrecognized distribution type' % pdf_type)
        else:
            if 'float' in str(series.dtype):
                obj = ContinuousPDF(series, start, stop)
            elif 'int' in str(series.dtype):
                obj = DiscretePDF(series, start, stop)
            else:
                obj = IndexedPDF(series)
        return obj


class ContinuousPDF(PDF):

    def __init__(self, series, start=None, stop=None):
        super(ContinuousPDF, self).__init__(series)
        if not type(series) == pd.Series:
            series = pd.Series(series)
        series = series[series.notnull()]
        if start and stop:
            self.ends = np.array((start, stop))
        else:
            self.ends = np.array((self.values.min(), self.values.max()))
        self.step = (self.ends[1] - self.ends[0]) / 200.
        self.pdf = self._estimate(series)

    def _estimate(self, series):

        def kde(xv, vect, d, bw):
            return sum(exp(-0.5 * ((xv - vect) / bw) ** 2)
                / d)

        d = sqrt(2 * pi * self.bw ** 2)
        r = np.linspace(*self.ends, num=200)
        a = np.array([kde(xv, series, d, self.bw) for xv in r])
        return a * (1 / a.sum())

    def prob(self, li):
        li = np.array(li)
        self.verify_slice(li),
        left = max(int((li.min() - self.ends[0]) / self.step) - 1, 0)
        right = min(int((li.max() - self.ends[0]) / self.step) + 1, 200)
        return self.pdf[range(left, right)].sum()

    def verify_slice(self, li):
        try:
            assert li.min() >= self.ends[0] and li.max() <= self.ends[1]
        except:
            s = '%s has range %s but %s provided'
            s = s % (self.name, self.ends, li)
            raise Exception(s)


class DiscretePDF(PDF):

    def __init__(self, series, start=None, stop=None):
        super(DiscretePDF, self).__init__(series)
        if not type(series) == pd.Series:
            series = pd.Series(series)
        series = series[series.notnull()]
        if start and stop:
            self.ends = np.array((start, stop))
        else:
            self.ends = np.array((self.values.min(), self. values.max() + 1))
        self.pdf = self._estimate(series)

    def _estimate(self, series):

        def kde(xv, vect, d, bw):
            return sum(exp(-0.5 * ((xv - vect) / bw) ** 2)
                / d)

        d = sqrt(2 * pi * self.bw ** 2)
        r = range(*self.ends)
        a = np.array([kde(xv, series, d, self.bw) for xv in r])
        return a * (1 / a.sum())

    def prob(self, li):
        li = np.array(li)
        self.verify_slice(li)
        li = li - self.ends.min()
        return self.pdf[li].sum()

    def verify_slice(self, li):
        try:
            assert all(np.intersect1d(li, np.array(range(*self.ends))) == li)
        except:
            s = '%s has range %s but %s provided'
            s = s % (self.name, self.ends, li)
            raise Exception(s)


class IndexedPDF(PDF):

    def __init__(self, series):
        super(IndexedPDF, self).__init__(series)
        if not type(series) == pd.Series:
            series = pd.Series(series)
        series = series[series.notnull()]
        self.ends = self.values
        self.pdf = self._estimate(series)

    def _estimate(self, series):
        a = np.array([(series == k).sum() for k in self.values])
        return a / float(series.count())

    def prob(self, li):
        li = np.array(li)
        self.verify_slice(li)
        p = 0
        for k in self.values:
            if k in li:
                p += self.pdf[np.argmax(self.values == k)]
        return p

    def plot(self):
        x = np.array(range(-1, self.values.count() + 1))
        y = [0] + list(self.pdf) + [0]
        plt.step(x, y)
        plt.xticks(np.array(range(0, self.values.count())) - .5, self.values)
        plt.show()

    def verify_slice(self, li):
        try:
            assert all(np.intersect1d(li, self.values) == li)
        except:
            s = '%s has range %s but %s provided'
            s = s % (self.name, self.ends, li)
            raise Exception(s)