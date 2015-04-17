# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


class LinearModeler(object):

    def __init__(self, datas, trainset=.8):
        shuffle = np.random.shuffle(datas)
        self.train = shuffle[:int(trainset * len(shuffle))]
        self.test = shuffle[int(trainset * len(shuffle)):]
        self.models = {}
        self.modelstats = {}

    def addmodel(self, modelarray):
        modelarray = tuple(modelarray)
        m = LinearModel(modelarray, self.train)
        self.models[modelarray] = m
        t = self.testmodel(m)
        self.modelstats[modelarray] = t
        return t

    def testmodel(self, model):
        pass


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

    def __init__(self, modelarray, datas):
        self.modelarray = modelarray
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
                results[key] = (np.array(row[1:]) * self.coefs[key]).sum()
            else:
                pass
        return results

    def runtest(self, testdatas):
        results = {}
        for data in testdatas:
            predic = self.predict(data)
            actual = getattr(data, self.dvar)
            for key in predic:
                if not key in results:
                    results[key] = []
                results[key].append((actual[key], predic[key]))
        return results

    def testplot(self, testdatas):
        results = self.runtest(testdatas)
        x = []
        y = []
        for key in results:
            ny, nx = zip(*results[key])
            x.extend(nx)
            y.extend(ny)
        plt.plot(x, y, 'o')
        plt.show()

    def testfit(self, testdatas):
        results = self.runtest(testdatas)
        for key in results:
            l = results[key]
            act, pre = map(np.array, zip(*l))
            sstot = ((act - act.mean()) ** 2).sum()
            ssres = ((act - pre) ** 2).sum()
            rsq = 1 - ssres / sstot
            dfe = float(len(l) - len(self.xvars) - 1)
            rbh = 1 - (ssres / dfe) / (sstot / float(len(l) - 1))
            rbh = rbh if rbh <= rsq else np.nan
            results[key] = (rsq, rbh)  # r^2 adjusted for number of xvars, r
        return results