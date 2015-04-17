# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


class LinearModeler(object):

    def __init__(self, datas, trainset=.8, bootstraps=40):
        self.datas = datas
        self.trainset = trainset
        self.bootstraps = bootstraps
        self.modelstats = {}

    def addmodel(self, modelarray, floor=None):
        modelarray = tuple(modelarray)
        rsq, rbh = [], []
        for i in range(self.bootstraps):
            np.random.shuffle(self.datas)
            train = self.datas[:int(self.trainset * len(self.datas))]
            test = self.datas[int(self.trainset * len(self.datas)):]
            m = LinearModel(modelarray, train, floor=floor)
            res = self.testmodel(m, test)
            rsq.append(res[0])
            rbh.append(res[1])
        rsq, rbh = np.array(rsq), np.array(rbh)
        self.modelstats[modelarray] = (rsq.mean(), rbh.mean())
        return self.modelstats[modelarray]

    def testmodel(self, model, test):
        res = model.testfit(test)
        return res['_TOTAL']


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

    def __init__(self, modelarray, datas, floor=None):
        self.modelarray = modelarray
        self.floor = floor
        self.dvar = modelarray[0]
        self.xvars = modelarray[1:]
        self.datas = datas
        self.setpoints()
        self.setcoefs()

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
            return att if not np.isnan(att) else 0
        elif isinstance(bit, tuple):
            func = bit[0]
            att1 = getattr(dataobj, bit[1]) if getattr(dataobj, bit[1]) else 0
            att2 = getattr(dataobj, bit[2]) if getattr(dataobj, bit[2]) else 0
            v = func(att1, att2)
            return v if not np.isnan(v) else 0

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
            vs = np.array(self.points[key])
            y = vs[:, 0]
            x = vs[:, 1:]
            lsq = np.linalg.lstsq(x, y)
            coefs[key] = lsq[0]
        self.coefs = coefs

    def predict(self, dataobj):
        results = {}
        for row, key in self.extract(dataobj):
            if key in self.coefs:
                v = (np.array(row[1:]) * self.coefs[key]).sum()
                results[key] = (v if self.floor is None or v > self.floor
                    else self.floor)
            else:
                pass
        return results

    def runtest(self, testdatas):
        results = {}
        act = []
        pre = []
        for data in testdatas:
            predic = self.predict(data)
            actual = getattr(data, self.dvar)
            for key in predic:
                if not key in results:
                    results[key] = []
                act.append(actual[key])
                pre.append(predic[key])
                results[key].append((actual[key], predic[key]))
        return results, act, pre

    def testplot(self, testdatas):
        results, y, x = self.runtest(testdatas)
        plt.plot(x, y, 'o')
        plt.show()

    def impearsonate(self, x, y):
        x, y = np.array(x), np.array(y)
        sstot = ((y - y.mean()) ** 2).sum()
        ssres = ((y - x) ** 2).sum()
        rsq = 1 - ssres / sstot
        dfe = float(len(x) - len(self.xvars) - 1)
        rbh = 1 - (ssres / dfe) / (sstot / float(len(x) - 1))
        rbh = rbh if rbh <= rsq else np.nan
        return rsq, rbh

    def testfit(self, testdatas):
        results, act, pre = self.runtest(testdatas)
        for key in results:
            l = results[key]
            y, x = zip(*l)
            results[key] = self.impearsonate(x, y)
        results['_TOTAL'] = self.impearsonate(pre, act)
        return results