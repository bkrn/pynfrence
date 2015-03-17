# -*- coding: utf-8 -*-

import numpy as np

from pdf import *


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
        """data: An {independent_variable: [vals]} dictionary
        Returns a {classifer_string: probabilty} dictionary
        """
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
        return dict(
            zip(
                [str(clsf) for clsf in self.clsfn],
                list(v * (1 / float(v.sum())))
            )
        )

    def classify(self, data):
        """data: An {independent variable: [vals]} dictionary
        Returns the string identifier for the most likely classification of
        the data
        """
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
        """ Returns a dictionary of {PDF name: range}.
        The PDFs labeled X|Y are the PDF for the variable X given result Y.
        """
        d = {}
        d['classifications'] = self.clsfn
        for var in self.varsPDFs:
            d[var] = self.varsPDFs[var].ends
        for clsf in self.clsfn:
            for var in self.givnPDFs[clsf]:
                d[('%s|%s' % (var, clsf))] = self.givnPDFs[clsf][var].ends
        return d

    def sweep_1d(self, data, target):
        """ data: An {independent variable: [vals]} dictionary
        target: String name of an independent variable to sweep
        Shows a plot of classifier probabilities given values in data and the
        whole range or target's possible values.
        """

        if target in data:
            raise BaseException('Target should not be in data')
        tgtpdf = self.varsPDFs[target]
        if isinstance(tgtpdf, IndexedPDF):
            x = tgtpdf.values
        elif isinstance(tgtpdf, DiscretePDF):
            x = range(*tgtpdf.ends)
        elif isinstance(tgtpdf, ContinuousPDF):
            x = np.linspace(tgtpdf.ends[0], tgtpdf.ends[1], 50)
        ys = []
        for v in x:
            data[target] = v
            ys.append(self.prob(data))
        for key in [str(clsf) for clsf in self.clsfn]:
            y = [r[key] for r in ys]
            plt.plot(x, y, 'o', label=key)
        plt.show()

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
