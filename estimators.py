import numpy as np
from sklearn.dummy import DummyRegressor

class HourWeekdayBinModel(DummyRegressor):

    def __init__(self, strategy='mean'):
        self._model = None
        self.strategy = strategy

    def fit(self, X, y):
        a = np.zeros((24, 7))
        hours = X[:, 1]
        weekdays = X[:, 2]

        if self.strategy == 'mean':
            counts = a.copy()
            for i, row in enumerate(zip(hours, weekdays)):
                hour = int(row[0])
                day = int(row[1])
                counts[hour, day] += 1
                a[hour, day] += y[i]

            self._model = a / counts
        elif self.strategy == 'median':

            # this is a 3d array 
            groups = [[[] for i in range(7)] for j in range(24)]

            for i, row in enumerate(zip(hours, weekdays)):
                hour = int(row[0])
                day = int(row[1])
                groups[hour][day].append(y[i])
            for i, j in np.ndindex((24, 7)):
                a[i,j] = np.median(groups[i][j])
            self._model = a

        #from matplotlib import pyplot as plt
        #plt.imshow(self._model)
        #plt.show()

        return self

    def predict(self, X):
        hours = X[:, 1]
        weekdays = X[:, 2]
        prediction = map(lambda x: self._model[x[0], x[1]], zip(hours, weekdays))
        return np.array(prediction)
