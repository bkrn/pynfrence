# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pynfrence.ftools import powerset


class LinearModeler(object):

    def __init__(self, datas, trainset=.8, bootstraps=10, bootstrapn=10):
        self.datas = datas
        self.trainset = trainset
        self.bootstraps = bootstraps
        self.bootstrapn = bootstrapn
        self.modelstats = {}
        self.choicemodels = {}
        self.models = {}

    def addmodel(self, modelarray, floor=None, minlen=4):
        modelarray = tuple(modelarray)
        boots = {}
        fullmodel = LinearModel(
            modelarray, self.datas, floor=floor, minlen=minlen, clean=False
        )
        points = fullmodel.points.copy()
        fullmodel.clean()
        self.models[modelarray] = fullmodel
        for key, rsq, rbh in self.doboots(points, modelarray, floor, minlen):
            if not key in boots:
                boots[key] = []
            boots[key].append((rsq, rbh))
        self.addboots(boots, modelarray)
        return self.modelstats[modelarray]['_TOTAL']

    def addboots(self, boots, modelarray):
        self.modelstats[modelarray] = {}
        for key in boots:
            stats = self.makestats(boots[key])
            self.modelstats[modelarray][key] = stats
            if ((not key in self.choicemodels) or
                    (stats[0][1] > self.choicemodels[key][1][0][1]) or
                    (
                        np.isnan(self.choicemodels[key][1][0][1]) and
                        not np.isnan(stats[0][1])
                    )
            ):
                self.choicemodels[key] = (self.models[modelarray], stats)

    def makestats(self, tuparr):
        rsq, rbh = map(np.array, zip(*tuparr))
        std = 1.96 * rsq.std()
        mean = rsq.mean()
        low, high = mean - std, mean + std
        return  (
            (low, mean, high),
            rbh.mean(),
        )

    def doboots(self, points, modelarray, floor, minlen):
        badmodels = 0
        for i in range(self.bootstraps):
            straps = []
            while len(straps) < self.bootstrapn:
                train, test = self.splitpoints(points)
                m = LinearModel(
                    modelarray, train, floor=floor, minlen=minlen, disp=True)
                res = m.testfit(test, points=True)
                if np.isnan(res['_TOTAL'][0]):
                    badmodels += 1
                    if badmodels > 10:
                        raise BaseException('That was a bad model!')
                straps.append(res)
            for key in straps[0]:
                rsqs, rbhs = map(np.array, zip(*[d[key] for d in straps]))
                yield key, rsqs.mean(), rbhs.mean()

    def splitpoints(self, points):
        train, test = {}, {}
        for key in points:
            arr = np.random.permutation(points[key])
            v = int(len(arr) * self.trainset)
            train[key], test[key] = arr[:v], arr[v:]
        return train, test

    def performance(self):
        a = np.array([
            self.choicemodels[key][1][0][1] for key in self.choicemodels
        ])
        a = a[~np.isnan(a)]
        models = set([self.choicemodels[key][0] for key in self.choicemodels])
        return a.min(), a.mean(), a.max(), len(models)


class LinearModel(object):
    """ Creates a best fit multi-variate linear model.
    modelarray: array of strings and (func, string, string) tuples
        strings are names of attributes of the data objects that
        return floats. Tuple''s string are the same with the function called as
        func(dataobj.string1, dataobj.string2) providing the final float.
    datas: An array of objects that return floats for
        get_attr(obj, modelarray string). Overlaod __getattr__ to return methods
        using eval if you want!
    """

    def __init__(self, modelarray, datas,
                 floor=None, minlen=4, disp=False, clean=True):
        self.modelarray = modelarray
        self.floor = floor
        self.minlen = min(len(self.modelarray) - 1, minlen)
        self.dvar = modelarray[0]
        self.xvars = modelarray[1:]
        self.datas = datas
        if disp:
            self.points = self.datas
        else:
            self.setpoints()
        self.setcoefs()
        if clean:
            self.clean()

    def __repr__(self):
        return str(self.modelarray)

    def extract(self, dataobj):
        depobj = getattr(dataobj, self.dvar)
        if isinstance(depobj, float):
            l = [depobj]
            l.extend([self.figure(dataobj, bit) for bit in self.xvars])
            yield np.array(l), 'NOKEY'
        elif isinstance(depobj, dict):
            for key in depobj:
                l = [depobj[key]]
                l.extend([self.figure(dataobj, bit) for bit in self.xvars])
                yield np.array(l), key

    def figure(self, dataobj, bit):
        if isinstance(bit, str):
            att = getattr(dataobj, bit)
        elif isinstance(bit, tuple):
            func = bit[0]
            att1 = getattr(dataobj, bit[1]) if getattr(dataobj, bit[1]) else 0
            att2 = getattr(dataobj, bit[2]) if getattr(dataobj, bit[2]) else 0
            att = func(att1, att2)
        return att

    def setpoints(self):
        points = {}
        for data in self.datas:
            for avars, key in self.extract(data):
                if not key in points:
                    points[key] = []
                points[key].append(avars)
        self.points = points

    def setcoefs(self):
        coefs = {}
        for key in self.points:
            if not key in coefs:
                coefs[key] = {}
            vs = np.array(self.points[key])
            for cset in powerset(range(1, len(self.modelarray))):
                # cset iterates across all possible subsets of the model xvars
                # this allows for using smaller models in the case of missing
                # data --- graceful! Long!
                if len(cset) >= self.minlen:
                    y = vs[:, 0]
                    x = vs[:, cset]  # remove non model columns
                    x = x[~np.isnan(x).any(axis=1)]  # Remove rows with nan
                    y = y[~np.isnan(x).any(axis=1)]  # Remove rows with nan
                    if len(x) > 0:
                        clf = linear_model.Ridge(fit_intercept=False)
                        clf.fit(x, y)
                        coefs[key][frozenset(cset)] = clf.coef_
                        #lsq = np.linalg.lstsq(x, y)
                        #coefs[key][frozenset(cset)] = lsq[0]
        self.coefs = coefs

    def objspredict(self, dataobj):
        results = {}
        for row, key in self.extract(dataobj):
            if key in self.coefs:
                data = np.array(row[1:])
                cbool = ~np.isnan(data)
                cset = np.where(cbool)[0] + 1
                if frozenset(cset) in self.coefs[key]:
                    data = data[cbool]
                    v = (data * self.coefs[key][frozenset(cset)]).sum()
                    results[key] = (v if self.floor is None or v > self.floor
                        else self.floor)
                else:
                    # If the model doesn't exist (to many missing data points).
                    pass
            else:
                # If the dependent variable is unrecognized.
                pass
        return results

    def pointpredict(self, point, key):
        data = np.array(point[1:])
        cbool = ~np.isnan(data)
        cset = np.where(cbool)[0] + 1
        if frozenset(cset) in self.coefs[key]:
            data = data[cbool]
            v = (data * self.coefs[key][frozenset(cset)]).sum()
            return (v if self.floor is None or v > self.floor else self.floor)

    def pointspredict(self, points, key):
        predict = []
        actual = []
        for froset in self.coefs[key]:
            working = np.isnan(points)
            goodcols = list(froset)  # cols in the model
            working[:, goodcols] = ~working[:, goodcols]
            selector = working[:, 1:].all(axis=1)
            if selector.any():
                arr = points[selector]
                actual.extend(list(arr[:, 0]))
                predict.extend((
                    arr[:, goodcols] * self.coefs[key][froset]).sum(axis=1))
        return predict, actual

    def objstest(self, testdatas):
        results = {}
        act = []
        pre = []
        for data in testdatas:
            predic = self.objspredict(data)
            actual = getattr(data, self.dvar)
            for key in predic:
                if not key in results:
                    results[key] = []
                act.append(actual[key])
                pre.append(predic[key])
                results[key].append((actual[key], predic[key]))
        return results, act, pre

    def pointstest(self, points):
        results = {}
        act = []
        pre = []
        for key in points:
            ppre, pact = self.pointspredict(points[key], key)
            act.extend(pact)
            pre.extend(ppre)
            results[key] = zip(act, pre)
        return results, act, pre

    def testplot(self, testdatas):
        results, y, x = self.objstest(testdatas)
        plt.plot(x, y, 'o')
        plt.show()

    def impearsonate(self, x, y):
        x, y = np.array(x), np.array(y)
        rsq = np.corrcoef(x, y)[0][1] ** 2
        rbh = 1 - (1 - rsq) * (
            (len(x) - 1) / float(len(x) - len(self.modelarray) - 2))
        return rsq, rbh

    def testfit(self, testdatas, totonly=False, points=False):
        if not points:
            results, act, pre = self.objstest(testdatas)
        else:
            results, act, pre = self.pointstest(testdatas)
        if not totonly:
            for key in results:
                l = results[key]
                y, x = zip(*l)
                results[key] = self.impearsonate(x, y)
        results['_TOTAL'] = self.impearsonate(pre, act)
        return results

    def clean(self):
        del self.datas
        del self.points