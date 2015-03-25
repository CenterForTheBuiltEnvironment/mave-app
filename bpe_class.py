"""
Building Energy Prediction

This software reads an input file (a required argument) containing 
building energy data in a format similar to example file. 
It then trains a model and estimates the error associated
with predictions using the model.

@author Paul Raftery <p.raftery@berkeley.edu>
@author Tyler Hoyt <thoyt@berkeley.edu>
"""

import pdb
import os, csv, sys, pickle, time
import dateutil.parser
import numpy as np
import estimators 
from datetime import datetime, date, timedelta
from scipy.stats import randint as sp_randint
from sklearn import preprocessing, cross_validation, svm, grid_search, \
        ensemble, neighbors, dummy
from comparison_functions import ne, nmbe, rmse, cvrmse, plot_comparison,\
        print_overview, write_model_results

class BPE(object):

    HOLIDAYS_PICKLE_FILENAME = os.path.join('holidays', 'USFederalHolidays.p')
    DATETIME_COLUMN_NAME = 'time.LOCAL'

    def __init__(self, 
                 training_filename, 
                 use_holidays=True,
                 training_start_frac=0.0,
                 training_end_frac=1.0,
                 ):

        training_f = open(training_filename, 'Ur')
        self.reader = csv.reader(training_f, delimiter=',')
        self.headers, country = self.process_headers()
        training_f.seek(0) # rewind the file so we don't have to open it again

        self.holidays = []
        if country == 'us' and use_holidays:
            with open(self.HOLIDAYS_PICKLE_FILENAME, 'r') as fp:
                self.holidays = pickle.load(fp)
        
        training_data = np.genfromtxt(training_f, 
                            comments='#', 
                            delimiter=',',
                            dtype=None, 
                            skip_header=len(self.headers)-1, 
                            names=True, 
                            missing_values='NA')
        dcn = self.DATETIME_COLUMN_NAME.replace(".", "")

        training_data_L = len(training_data)
        training_start_index = int(training_start_frac * training_data_L)
        training_end_index = int(training_end_frac * training_data_L)
        training_data = training_data[ training_start_index : training_end_index ]
        
        try: 
            datetimes = map(lambda d: datetime.strptime(d, "%m/%d/%y %H:%M"), training_data[dcn])
        except ValueError:
            datetimes = map(lambda d: dateutil.parser.parse(d, dayfirst=False), training_data[dcn])

        dtypes = training_data.dtype.descr
        dtypes[0] = dtypes[0][0], '|S16' # force S16 datetimes
        training_data = training_data.astype(dtypes)

        self.training_data = self.clean_data(training_data, datetimes)

        print self.training_data
   
    def clean_data(self, data, datetimes):

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

        return data

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

if __name__=='__main__':

    bpe0 = BPE('data/6_P_cbe_02.csv', 
            training_start_frac=0.2,
            training_end_frac=0.8)
    print bpe0

