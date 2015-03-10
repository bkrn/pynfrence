# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from pdf import PDF


class BayesMachine(object):

    def __init__(self, df, clsfn):
        clsfr = df[clsfn]
        self.varsn = list(df.columns)
        self.varsn.remove(clsfn)
        self.clsfn = clsfr.unique()
        self.clsfPDF = PDF.make(clsfr, pdf_type='indexed')
        self.varsPDFs = dict([(var, PDF.make(df[var])) for var in self.varsn])
        self.givnPDFs = dict([(clsf, dict(
            [
                (var, PDF.make(df[clsfr == clsf][var],
                start=df[var].min(),
                stop=df[var].max())) for var in self.varsn
            ])) for clsf in self.clsfn
        ])

    def prob(self, data):
        given = np.array([
            np.prod([
                self.givnPDFs[clsf][var].prob(data[var]) for var in data.keys()
            ]) for clsf in self.clsfn
        ])
        clsfs = np.array([self.clsfPDF.prob(clsf) for clsf in self.clsfn])
        prior = np.array([
            np.prod([
                self.varsPDFs[var].prob(data[var]) for var in data.keys()
            ]) for clsf in self.clsfn
        ])
        v = ((given * clsfs) / prior)
        return pd.DataFrame(
            columns=[str(clsf) for clsf in self.clsfn],
            data=[v * (1 / float(v.sum()))]
        )

    def classify(self, data):
        given = np.array([
            np.prod([
                self.givnPDFs[clsf][var].prob(data[var]) for var in data.keys()
            ]) for clsf in self.clsfn
        ])
        clsfs = np.array([self.clsfPDF.prob(clsf) for clsf in self.clsfn])
        prior = np.array([
            np.prod([
                self.varsPDFs[var].prob(data[var]) for var in data.keys()
            ]) for clsf in self.clsfn
        ])
        v = ((given * clsfs) / prior)
        return self.clsfn[list(v).index(v.max())]

    def ranges(self):
        d = {}
        d['classifications'] = self.clsfn
        for var in self.varsPDFs:
            d[var] = self.varsPDFs[var].ends
        for clsf in self.clsfn:
            for var in self.givnPDFs[clsf]:
                d['%s|%s' % var, clsf] = self.givnPDFs[clsf][var]
        return d

    def test(self, series, n=20, its=1000):
        # THIS DOES NOT WORK
        assert series.count()[0] > 100
        series = series.dropna()

        def tester(series, its, n):
            name = self.clsfPDF.name
            cols = list(series.columns)
            cols.remove(name)
            naiv = np.array([
                self.prob({})[str(c)] for c in self.clsfn]).cumsum()
            while its:
                ind = np.random.choice(range(0, series.count()[0]), size=n)
                samp = series.iloc[ind]
                actu = np.array([(
                    samp[name] == c).sum() / 20. for c in self.clsfn]).cumsum()
                sc = [self.classify(samp[cols][i:i + 1]) for i in range(20)]
                smrt = np.array([
                    sc.count(c) / 20. for c in self.clsfn]).cumsum()
                dif = abs((actu - smrt)).max() - abs((actu - naiv)).max()
                its -= 1
                yield dif

        diffs = np.fromiter(tester(series, its, n), dtype=np.float64)
        pdf = PDF.make(diffs)
        pdf.plot()
        return diffs.mean() / diffs.std()
