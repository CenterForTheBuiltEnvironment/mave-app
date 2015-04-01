"""
Building Energy Prediction

This software reads an input file (a required argument) containing 
building energy data in a format similar to example file. 
It then trains a model and estimates the error associated
with predictions using the model.

@author Paul Raftery <p.raftery@berkeley.edu>
@author Tyler Hoyt <thoyt@berkeley.edu>
"""

import os, csv, pickle
import dateutil.parser
import numpy as np
from datetime import datetime, timedelta
from sklearn import preprocessing, cross_validation
import trainers

class Preprocessor(object):

    HOLIDAYS_PICKLE_FILENAME = os.path.join('holidays', 'USFederalHolidays.p')
    DATETIME_COLUMN_NAME = 'time.LOCAL'
    HISTORICAL_DATA_COLUMN_NAMES = ['dboatF']
    TARGET_COLUMN_NAMES = ['wbelectricitykWh']

    def __init__(self, 
                 input_file, 
                 use_holidays=True,
                 start_frac=0.0,
                 end_frac=1.0,
                 ):

        self.reader = csv.reader(input_file, delimiter=',')
        self.headers, country = self.process_headers()
        input_file.seek(0) # rewind the file so we don't have to open it again

        self.holidays = []
        if country == 'us' and use_holidays:
            with open(self.HOLIDAYS_PICKLE_FILENAME, 'r') as fp:
                self.holidays = pickle.load(fp)
        
        input_data = np.genfromtxt(input_file, 
                            comments='#', 
                            delimiter=',',
                            dtype=None, 
                            skip_header=len(self.headers)-1, 
                            names=True, 
                            missing_values='NA')
        dcn = self.DATETIME_COLUMN_NAME.replace(".", "")

        input_data_L = len(input_data)
        start_index = int(start_frac * input_data_L)
        end_index = int(end_frac * input_data_L)
        input_data = input_data[ start_index : end_index ]

        try: 
            datetimes = map(lambda d: datetime.strptime(d, "%m/%d/%y %H:%M"), input_data[dcn])
        except ValueError:
            datetimes = map(lambda d: dateutil.parser.parse(d, dayfirst=False), input_data[dcn])

        dtypes = input_data.dtype.descr
        dtypes[0] = dtypes[0][0], '|S16' # force S16 datetimes
        input_data = input_data.astype(dtypes)

        input_data, self.datetimes = self.interpolate_datetime(input_data, datetimes)

        use_month = True if (self.datetimes[0] - self.datetimes[-1]).days > 360 else False
        vectorized_process_datetime = np.vectorize(self.process_datetime)
        d = np.column_stack(vectorized_process_datetime(self.datetimes, use_month))
        self.num_dt_features = np.shape(d)[1]
        # minute, hour, weekday, holiday, and (month)

        input_data, target_column_index = self.append_input_features(input_data, d)
        self.X, self.y, self.datetimes = \
                self.clean_missing_data(input_data, self.datetimes, target_column_index)

    def clean_missing_data(self, d, datetimes, target_column_index):
        # remove any row with missing data
        # filter the datetimes and data arrays so the match up
        keep_inds = ~np.isnan(d).any(axis=1)
        num_to_del = len(keep_inds[~keep_inds])
        if num_to_del > 0: 
            datetimes = datetimes[keep_inds]
            d = d[keep_inds]

	    if (d[:,0] != np.array([dt.minute for dt in datetimes])).any() or \
			(d[:,1] != np.array([dt.hour for dt in datetimes])).any():
	        raise Error(" - The datetimes in the datetimes array do not \
			  match those in the input features array")

	    # split into input and target arrays
        X, target_data = np.hsplit(d, np.array([target_column_index]))

        return X, target_data, datetimes

    def append_input_features(self, data, d0, historical_data_points=0):
        column_names = data.dtype.names[1:] # no need to include datetime column
        d = d0
        for s in column_names:
            if s in self.HISTORICAL_DATA_COLUMN_NAMES:
                d = np.column_stack( (d, data[s]) )
                if historical_data_points > 0:
                    # create input features using historical data 
                    # at the intervals defined by n_vals_in_past_day
                    for v in range(1, historical_data_points + 1):
                        past_hours = v * 24 / (historical_data_points + 1)
                        n_vals = past_hours * vals_per_hr
                        past_data = np.roll(data[s], n_vals)
                        # for the first day in the data file there will be no historical data
                        # use the data from the next day as a rough estimate
                        past_data[0:n_vals] = past_data[ 24 * vals_per_hr: 24 * vals_per_hr + n_vals ]
            elif not s in self.TARGET_COLUMN_NAMES:
                # just add the column as an input feature without historical data
                d = np.column_stack( (d, data[s]) )

        # add the target data
        split = d.shape[1]
        for s in self.TARGET_COLUMN_NAMES:
            if s in column_names:
                d = np.column_stack( (d, data[s]) )
        return d, split

    def interpolate_datetime(self, data, datetimes):

        start = datetimes[0]
        second_val = datetimes[1]
        end = datetimes[-1]

        # calculate the interval between datetimes
        interval = second_val - start
        vals_per_hr = 3600 / interval.seconds
        assert (3600 % interval.seconds) == 0,  \
          'Interval between datetimes must divide evenly into an hour'

        # check to ensure that the timedelta between datetimes is
        # uniform through out the array
        row_length = len(data[0])
        diffs = np.diff(datetimes)
        gaps = np.greater(diffs, interval)
        gap_inds = np.nonzero(gaps)[0] # gap_inds contains the left indices of the gaps
        NN = 0 # accumulate offset of gap indices as you add entries
        for i in gap_inds:
            gap = diffs[i]
            gap_start = datetimes[i + NN]
            gap_end = datetimes[i + NN + 1]
            N = gap.seconds / interval.seconds - 1 # number of entries to add
            for j in range(1, N+1):
                new_dt = gap_start + j * interval
                new_row = np.array([(new_dt,) + (np.nan,) * (row_length - 1)], dtype=data.dtype)
                print ("-- Missing datetime interval between %s and %s" % (gap_start, gap_end))
                data = np.append(data, new_row)
                datetimes = np.append(datetimes, new_dt) 
                datetimes_ind = np.argsort(datetimes) # returns indices that would sort datetimes
            data = data[datetimes_ind] # sorts arr by sorted datetimes object indices
            datetimes = datetimes[datetimes_ind] # sorts datetimes
            NN += N

        return data, datetimes

    def process_headers(self):
        # reads up to the first 100 lines of a file and returns
        # the headers, and the country in which the building is located
        headers = []
        country = None
        for _ in range(100):
            row = self.reader.next()
            headers.append(row)
            for i, val in enumerate(row):
                if val.lower().strip() == 'country':
                    row = self.reader.next()
                    headers.append(row)
                    country = row[i]
            if row[0] == self.DATETIME_COLUMN_NAME: break
        return headers, country

    def process_datetime(self, dt, use_month):
        # takes a datetime and returns a tuple of:
        # minute, hour, weekday, holiday, and (month)
        w = float(dt.weekday())
        rv = float(dt.minute), float(dt.hour), w
        if self.holidays:
            if dt.date() in self.holidays:
                hol = 3.0 # this day is a holiday
            elif (dt + timedelta(1,0)).date() in self.holidays:
                hol = 2.0 # next day is a holiday
            elif (dt - timedelta(1,0)).date() in self.holidays:
                hol = 1.0 # previous day was a holiday
            else:
                hol = 0.0 # this day is not near a holiday
            rv += hol,

        if use_month: rv += float(dt.month),
        return rv


class ModelAggregator(object):

    def __init__(self, bpe0, prediction_fraction=0.33):
        self.bpe0 = bpe0
        X = bpe0.X
        y = np.ravel(bpe0.y)

        self.X_standardizer = preprocessing.StandardScaler().fit(X)
        self.y_standardizer = preprocessing.StandardScaler().fit(y)
        self.X_s = self.X_standardizer.transform(X)
        self.y_s = self.y_standardizer.transform(y)

        #self.X_s, self.X_test_s, self.y_s, self.y_test_s = \
        #        cross_validation.train_test_split(X_s, y_s, \
        #            test_size=prediction_fraction, random_state=0)

        self.models = []
        self.best_model = None

    def train_dummy(self):
        dummy_trainer = trainers.DummyTrainer()
        dummy_trainer.train(self.X_s, self.y_s)
        self.models.append(dummy_trainer.model)
        return dummy_trainer.model

    def train_hour_weekday(self):
        hour_weekday_trainer = trainers.HourWeekdayBinModelTrainer()
        hour_weekday_trainer.train(self.X_s, self.y_s)
        self.models.append(hour_weekday_trainer.model)
        return hour_weekday_trainer.model

    def train_all(self):
        dummy_trainer = trainers.DummyTrainer()
        dummy_trainer.train(self.X_s, self.y_s)
        self.models.append(dummy_trainer.model)
        print "trained dummy"

        hour_weekday_trainer = trainers.HourWeekdayBinModelTrainer()
        hour_weekday_trainer.train(self.X_s, self.y_s)
        self.models.append(hour_weekday_trainer.model)
        print "trained hour-weekday"

        kneighbors_trainer = trainers.KNeighborsTrainer()
        kneighbors_trainer.train(self.X_s, self.y_s)
        self.models.append(kneighbors_trainer.model)
        print "trained k-nearest neighbors"

        #svr_trainer = trainers.SVRTrainer(search_iterations=0)
        #svr_trainer.train(self.X_s, self.y_s)
        #self.models.append(svr_trainer.model)
        #print "trained svr"

        random_forest_trainer = trainers.RandomForestTrainer(search_iterations=20)
        random_forest_trainer.train(self.X_s, self.y_s)
        self.models.append(random_forest_trainer.model)
        print "trained random forest"

        #gradient_boosting_trainer = trainers.GradientBoostingTrainer(search_iterations=2)
        #gradient_boosting_trainer.train(self.X_s, self.y_s)
        #self.models.append(gradient_boosting_trainer.model)
        #print "trained gradient boosting"

        extra_trees_trainer = trainers.ExtraTreesTrainer(search_iterations=20)
        extra_trees_trainer.train(self.X_s, self.y_s)
        self.models.append(extra_trees_trainer.model)
        print "trained extra trees"
        
if __name__=='__main__': 

    f = open('data/6_P_cbe_02.csv', 'Ur')
    bpe0 = Preprocessor(f, 
            start_frac=0.2,
            end_frac=0.8)

    m = ModelAggregator(bpe0)
    m.train_all()
    print map(lambda m: m.best_score_, m.models)
