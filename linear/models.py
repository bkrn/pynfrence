

from sklearn import linear_model


class LinearModel(object):

    def __init__(self):
        self.modeler = linear_model.Ridge()