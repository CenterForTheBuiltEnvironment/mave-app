"""
Building Energy Prediction

This software reads an input file (a required argument) containing buidling energy data in a format
similar to example file. It then trains a model and estimates the error associated
with predictions using the model.

@author Paul Raftery <p.raftery@berkeley.edu>
"""

import os, csv, sys, pickle, argparse, time
import dateutil.parser
import numpy as np
from datetime import datetime, date, timedelta
from scipy.stats import randint as sp_randint
from sklearn import preprocessing, cross_validation, svm, grid_search, ensemble, neighbors, dummy
from comparison_functions import ne, nmbe, rmse, cvrmse, plot_comparison, print_overview, write_model_results

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", 
                        help="filename for input data - REQUIRED ")
    parser.add_argument("-pi", "--prediction_input_filename",
                        help="filename containing input prediction data")
    parser.add_argument("-po", "--prediction_output_filename", default='Prediction.csv',
                        help="filename for the prediction output (default = Prediction.csv)")
    parser.add_argument("-of","--output_folder", default='Results',
                        help="folder name for result files (default=Results)")
    parser.add_argument("-n", "--n_jobs", type=int, default=1,
                        help="maximum number of jobs to run in parallel (default=1) (all cores = -1)")
    parser.add_argument("-k", "--k_folds", type=int, default=10,
                        help="number of folds used to cross-validate (default=10)")
    parser.add_argument("-c", "--comp_time", type=float, default=1.0,
                        help=" approx linear scalar for comp time spent performing random grid search (default=1.0)")
    parser.add_argument("-pf", "--prediction_fraction", type=float, default=0.33,
                        help="if no prediction file is given, fraction of the input file to use for prediction purposes (default=0.33)")
    parser.add_argument("-rs", "--reduce_size", type=float, default=None,
                        help="reduce the size of the input file data to analyze (useful to test model fit on just part of a file). e.g. -rs 0.25 uses half the first 25% of the file")    
    parser.add_argument("-rps", "--random_prediction_dataset", action="store_true",
                        help="chooses the validation set randomly, instead of from file end")
    parser.add_argument("-mmn", "--min_max_normalization", action="store_true", default=False,
                        help="normalize the data to a range of 2 about 0, instead of mean & std. dev")
    parser.add_argument("-nv", "--n_vals_in_past_day", type=int, default=2,
                        help="number of values (equal time intervals) in past day to use as additional input feature (default=2)")
    parser.add_argument("-hol", "--holidays_flag", action="store_false", default=True,
                        help="do not use holidays as an input feature")
    parser.add_argument("-j", "--join_holidays", action="store_true", default=False,
                        help="join holidays with weekday input feature, (i.e. do not use as separate feature)")
    parser.add_argument("-mth", "--month_flag", action="store_false", default=True,
                        help="do not use month as an input feature")
    parser.add_argument("-knr", "--knr_flag", action="store_true", default=False,
                        help=" use k neighbours regression")
    parser.add_argument("-svr", "--svr_flag", action="store_true", default=False,
                        help=" use support vector regression")
    parser.add_argument("-rfr", "--rfr_flag", action="store_false", default=True,
                        help="do not use random forest regression")
    parser.add_argument("-gbr", "--gbr_flag", action="store_true", default=False,
                        help="use gradient boosting regression")
    parser.add_argument("-etr", "--etr_flag", action="store_false", default=True,
                        help="do not use extra trees regression")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save detailed results")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="plot training and validation results")
    args = parser.parse_args()

    # general parameters
    ## define column names for the target data
    targetColNames = ['wbelectricity.kWh'] # currently this tool only predicts whole building elec
    ### other options:
    ##targetColNames = {'wbelectricity.kWh', 'wbgas.kBTU', 'chw.kBTU', 'hw.kBTU',
    ##           'steam.kBTU', 'coolingelectricity.kWh', 'coolinggas.kBTU',
    ##           'heatingelectricity.kWh', 'heatinggas.kBTU',
    ##           'ventilationelectricity.kWh', 'lightingelectricity.kWh'}

    inputColNames = ['dboat.F'] ## add other desired input features

    # remove special characters
    inputColNames = [s.replace(".", "") for s in inputColNames]
    targetColNames = [s.replace(".", "") for s in targetColNames]
    ## define datetime column name
    datetime_col_name = 'time.LOCAL'

    # define run characteristics
    n_jobs = args.n_jobs # number of cores (-1 = all cores)
    if n_jobs < -1 or n_jobs == 0:
        n_jobs = 1 # handle case where user enters an unsupported int
        print "--- Number of cores must be either -1, or any value >0 ---"
        print "--- Number of jobs reset to default (1)"
    k = args.k_folds # number of folds used in k-fold cross validation
    # roughly linear scalar for comp time spent performing random grid search
    # for optimal paramter values for each regression model
    # reasonable results for values >= 0.1, recommended 1 (or greater)
    comp_time = args.comp_time
    verbose = args.verbose

    # define input file with training data
    fp = args.input_file
    # define test set
    # if prediction_input_filename contains a string value
    # the model will predict results based on a new file
    # using the prediction_input_filename value as filename
    prediction_input_filename = args.prediction_input_filename
    prediction_output_filename = args.prediction_output_filename
    # if prediction_input_filename is None (False), the model will
    # predict results based on a fraction taken from input training file
    prediction_fraction = args.prediction_fraction
    # input standardization to use
    # uses mean & std dev by default, set min_max_normalization = True to force min and max range
    min_max_normalization = args.min_max_normalization
    random_prediction_dataset = args.random_prediction_dataset # whether to randomly select, or select from file end
    n_vals_in_past_day = args.n_vals_in_past_day
    reduce_size = args.reduce_size
    holidays_flag = args.holidays_flag
    join_holidays = args.join_holidays
    month_flag = args.month_flag
    knr_flag = args.knr_flag
    svr_flag = args.svr_flag
    rfr_flag = args.rfr_flag
    gbr_flag = args.gbr_flag
    etr_flag = args.etr_flag

    save = args.save # saves each the main numpy objects and the trained models to files
    plot = args.plot # plots the prediction results   

    # define output folder from parameters
    op = os.path.join(args.output_folder)
    
    # creates a sub folder output name that is detailed summary of the
    # parameters used for these runs and saves the detailed output in
    # folder
    if save and not prediction_input_filename:
        op = os.path.join(op,fp)
        folder = 'rs'+ str(reduce_size) + \
                 '_pf' + str(prediction_fraction) + \
                 '_rps'+ str(random_prediction_dataset)[0]
        op = os.path.join(op,folder)
        folder = 'c' + str(comp_time)+ \
                 '_k' + str(k) + \
                 '_mmn' + str(min_max_normalization)[0] + \
                 '_mth' + str(month_flag)[0] + \
                 '_hol' + str(holidays_flag)[0] + \
                 '_nv' + str(n_vals_in_past_day) + \
                '_j' + str(join_holidays)
        op = os.path.join(op,folder)

    if not os.path.exists(op): os.makedirs(op)

    # list to contain each trained models
    models = []
    
    def process_headers(fp):
        # reads up to the first 100 lines of a file and returns
        # the headers, and the country in which the building is located
        headers =[]
        line = 0
        country = None
        with open(fp, 'Ur') as f:
            r = csv.reader(f, delimiter=',')
            while line <100: 
                row = r.next()
                headers.append(row)
                for i, val in enumerate(row):
                    if val.lower().strip() == 'country':
                        row = r.next()
                        headers.append(row)
                        country = row[i]
                if row[0] == datetime_col_name: break
                line +=1
            if country == None:
                raise Exception("The country could not be identified from the input file header lines")
                sys.exit(-1)
        return headers, country

    def process_datetime(dt_as_string, use_month):
        # takes a string, converts it to a datetime,
        # and returns a tuple of minute, hour, weekday, holiday, and (month)
        dt = dateutil.parser.parse(dt_as_string, dayfirst=False)
        w = float(dt.weekday())
        if join_holidays:
            if dt.date() in holidays:
                w = 7.0 # weekday is an int in range 0-6, 7 indicates a holiday
        else:
            if dt.date() in holidays:
                hol = 3.0 # this day is a holiday
            elif (dt - timedelta(1,0)).date() in holidays:
                hol = 1.0 # previous day was a holiday
            elif (dt + timedelta(1,0)).date() in holidays:
                hol = 2.0 # next day is a holiday                
            else:
                hol = 0.0 # this day is not near a holiday
        if holidays_flag and not join_holidays:
            if use_month:
                rv = float(dt.minute), float(dt.hour), w, hol, float(dt.month)
            else:
                rv = float(dt.minute), float(dt.hour), w, hol
        else:
            if use_month:
                rv = float(dt.minute), float(dt.hour), w, float(dt.month)
            else:
                rv = float(dt.minute), float(dt.hour), w
        return rv

    def process_input_data(fp, force_month=False, prediction_file=False): 
        with open(fp, 'Ur')  as f:
            arr = np.genfromtxt(f, comments='#', delimiter=',',
                                     dtype=None, skip_header=len(headers)-1, names=True, missing_values='NA')
        dcn = datetime_col_name.replace(".", "")
        L = len(arr)
        # reduce the size of the array if requested
        # (for testing a small period from one file covering a long period)
        if reduce_size:
            arr = arr[:int(reduce_size * L)]
            
        datetimes = arr[dcn]
        start = dateutil.parser.parse(arr[dcn][0], dayfirst=False)
        second_val = dateutil.parser.parse(arr[dcn][1], dayfirst=False)
        end = dateutil.parser.parse(arr[dcn][-1], dayfirst=False)

        # handle case where there is no datetime with 16 characters
        # in the input file (e.g. from 1/1/2000 00:00 to 10/9/2000 00:00
        # in this case numpy.genfromtxt sets the type of the column to S15
        # which causes problems when a longer datetime string is needed
        # - e.g. 10/10/2000 00:00
        dtypes = arr.dtype.descr
        dtypes[0] = dtypes[0][0], '|S16'
        arr = arr.astype(dtypes)
        
        # calculate the interval between datetimes
        interval = (second_val-start).seconds
        vals_per_hr = 3600/interval
        assert 3600%interval==0, 'Interval between datetimes must divide evenly into an hour'
        # check to ensure that the timedelta between datetimes is
        # uniform through out the array
        prev = start
        i = 0
        row_length = len(arr[0])

        while i < L-1:
            i += 1
	    if isinstance(arr[dcn][i], datetime):
	        cur = dateutil.parser.parse(arr[dcn][i], dayfirst=False)
                if (cur-prev).seconds%interval != 0:
                    raise Exception("Irregular datetime interval identified between " \
                      + str(prev) + " and " + str(cur) + ". This is not supported")
                    sys.exit(-1)
                if (cur-prev).seconds/interval > 1:
                    if verbose: print '-- Missing datetime interval after ' \
                      + str(prev)
                    # add blank row(s) to replace the missing datetime
                    empty_row = (np.nan,) * row_length
                    arr = np.insert(arr,i,empty_row,axis=0)
                    # create new datetime and add to the array
                    new_dt = prev + timedelta(0,interval)              
                    arr[dcn][i] = new_dt.strftime("%m/%d/%Y %H:%M")
                prev = dateutil.parser.parse(arr[dcn][i], dayfirst=False)

        # identify if month of year is a viable training feature
        if prediction_input_filename and not random_prediction_dataset:
            # all of the input file is representative training data
            end_train = end
        else:
            # only a portion of the input file is training data
            train_size = int(L*(1-prediction_fraction))
            end_train = dateutil.parser.parse(arr[dcn][train_size], dayfirst=False)
        more_than_12_months_data = True if (end_train - start).days > 360 else False
        if prediction_file:
            use_month = force_month
        else:
            use_month = True if more_than_12_months_data and month_flag else False
            
        if verbose: print "-- Generating training features from datetimes"
        vectorized_process_datetime = np.vectorize(process_datetime)
        d = np.column_stack(vectorized_process_datetime(arr[dcn], use_month))  

        # add other selected input features if present in textfile
        if verbose: print "-- Adding other input data as training features"    
        for s in inputColNames:
            if s in arr.dtype.names:
                d = np.column_stack((d,arr[s]))
                if n_vals_in_past_day >0:
                    # create historical data at the intervals defined by n_vals_in_past_day
                    for v in range(1,n_vals_in_past_day+1):
                        past_hours = v*24/(n_vals_in_past_day+1)
                        n_vals = past_hours*vals_per_hr
                        past_data = np.roll(arr[s],n_vals)
                        # for the first day in the data file there will be no historical data
                        # use the data from the next day as an estimate
                        past_data[0:n_vals] = past_data[24*vals_per_hr:24*vals_per_hr+n_vals]
                        d = np.column_stack((d,past_data))
                else:
                    d = np.column_stack((d,arr[s]))
        # add the target data
        split = d.shape[1]
        for s in targetColNames:
            if s in arr.dtype.names:
                d = np.column_stack((d,arr[s]))
                    
        # remove any row with missing data
        if verbose: print "-- Removing training examples with missing values"
        d = d[~np.isnan(d).any(axis=1)]
        # split into input and target arrays
        inputData, targetData = np.hsplit(d, np.array([split]))        
        return inputData, targetData, headers, arr[dcn], use_month

    def trainer(model, name, param_dist, search_iterations):
        # trains a model to the training data
        # using a random grid search assessed using k-fold cross validation
        if verbose: print "\n-- Training " + name + " regressors"
        # scale number of iterations according to requested computation time
        search_iterations = int(search_iterations*comp_time) 
        model = grid_search.RandomizedSearchCV(model, param_distributions=param_dist,
                                               n_iter=search_iterations,
                                               n_jobs=n_jobs,
                                               cv=k,
                                               verbose=verbose)
        model.fit(X_s, y_s)
        if verbose:
            if hasattr(model.best_estimator_, 'feature_importances_'):
                print 'Feature importances: ' + \
                      str(model.best_estimator_.feature_importances_ ) 

        
        if verbose: print "Best score: " + str(model.best_score_)
        return model

    print "\n...making prediction - please wait...\n"
    if verbose: print "\n=== Processing input file ==="
    
    headers, country = process_headers(fp)
    if country == 'us' and holidays_flag:
        with open(os.path.join('Holidays','USFederalHolidays.p'), 'r') as fh:
            holidays = pickle.load(fh)
    else:
        holidays = []

    X, y, headers, datetimes, force_month = process_input_data(fp)
    y = np.ravel(y)
    if verbose: print "-- Using " + str(X.shape[1]) + " input features"
    if verbose: print "-- " + str(X.shape[1]-(n_vals_in_past_day+1)) + \
       " are date or time related, the remainder are weather related"
    if verbose: print "-- Using " + str(X.shape[0]) + \
       " valid samples (i.e. no missing values)"    

    if prediction_input_filename:
        # predict results from a separate file containing only input
        # data (no training data)
        if verbose: print "\n=== Processing prediction input file ==="
        X_test, y_test, headers, predict_datetimes, x = \
                process_input_data(prediction_input_filename, force_month, prediction_file=True)
        X_test = np.array(X_test).astype(np.float)
        y_test = np.array(y_test).astype(np.float)
        y_test = np.ravel(y_test)
    else:
        # create a test set from input training data file
        # kept independent throughout the whole process
        if random_prediction_dataset:
            # randomly select        
            X, X_test, y, y_test = cross_validation.train_test_split(
                X, y, test_size=prediction_fraction, random_state=0)
        else:
            # select from end of file (i.e. forward prediction from one input file)
            train_size = int(len(X)*(1-prediction_fraction))
            X_test = X[train_size+1:]
            y_test = y[train_size+1:]
            X = X[:train_size]
            y = y[:train_size]
    
    if verbose: print "-- Normalizing training data"
    # standardize inputs
    if min_max_normalization:
        X_standardizer = preprocessing.MinMaxScaler().fit(X)
        y_standardizer = preprocessing.MinMaxScaler().fit(y)  
    else:
        X_standardizer = preprocessing.StandardScaler().fit(X)
        y_standardizer = preprocessing.StandardScaler().fit(y)
    X_s = X_standardizer.transform(X)
    y_s = y_standardizer.transform(y)
    X_test_s = X_standardizer.transform(X_test)
    y_test_s = y_standardizer.transform(y_test)

    if verbose: print "\n=== Training multiple regression models using a randomized"
    if verbose: print " grid search to identify the best parameters for each regressor"
    if verbose: print " and evaluating model fit using k-fold cross validation"

    param_dist = {"strategy": ['mean', 'median']}
    models.append(trainer(dummy.DummyRegressor(),
                          "dummy", param_dist,4/comp_time))
    
    param_dist = {"p": [1,2],
                  "n_neighbors": sp_randint(6, 40),
                  "leaf_size": np.logspace(1, 2.5, 1000)}        
    if knr_flag: models.append(trainer(neighbors.KNeighborsRegressor(),
                                            "k neighbours", param_dist,
                                            50))
    # note: probably don't need that many iterations for knr (total approx 200?
    # alos looks like p=1 always performs poorer than p=2 for this type of data
    
    param_dist = {"C": np.logspace(-3, 1, 1000),
                  "epsilon": np.logspace(-3, 0.5, 1000),
                  "degree": [2,3,4],
                  "gamma": np.logspace(-3, 2, 1000),
                  "max_iter": [20000]}
    if svr_flag: models.append(trainer(svm.SVR(),
                                            "support vector", param_dist,
                                            5))
    
    param_dist = {"max_depth": [4, 5, 6, 7, 8, 9, 10, None],
                  "max_features": sp_randint(3, X.shape[1]+1),
                  "min_samples_split": sp_randint(5, 500),
                  "min_samples_leaf": sp_randint(5, 500),
                  "bootstrap": [True,False]}
    if rfr_flag: models.append(trainer(ensemble.RandomForestRegressor(),
                                            "random forest", param_dist,
                                            200))

    param_dist = {"max_depth": [4, 5, 6, 7, 8, 9, 10, None],
                  "n_estimators": np.logspace(1.5, 4, 1000).astype(int),
                  "max_features": sp_randint(3, X.shape[1]+1),
                  "min_samples_split": sp_randint(5, 50),
                  "min_samples_leaf": sp_randint(5, 50),
                  "subsample": [0.8, 1.0],
                  "learning_rate": [0.05, 0.1, 0.2, 0.5]}    
    if gbr_flag: models.append(trainer(ensemble.GradientBoostingRegressor(),
                                            "gradiant boosting", param_dist,
                                            50))
    
    param_dist = {"max_depth": [4, 5, 6, 7, 8, 9, 10, None],
                  "n_estimators": sp_randint(5, 50),
                  "max_features": sp_randint(3, X.shape[1]+1),
                  "min_samples_split": sp_randint(5, 50),
                  "min_samples_leaf": sp_randint(5, 50),
                  "bootstrap": [True,False]}
    if etr_flag: models.append(trainer(ensemble.ExtraTreesRegressor(),
                                            "extra trees", param_dist,
                                            200))

    if verbose: print "\n=== Finding best regressor ==="
    best_model = []
    for m in models:
        if not best_model or best_model.best_score_ < m.best_score_: best_model = m
    if verbose: print "-- Predicting results for training and test set using best regressor"
    out_s = best_model.predict(X_s)
    out = y_standardizer.inverse_transform(out_s)
    out_test_s = best_model.predict(X_test_s)
    out_test = y_standardizer.inverse_transform(out_test_s)

    if verbose:
        print "\n=== Evaluating results ==="
        print "-- Best model parameters"
        print best_model.best_estimator_
        print "\n-- Best model evaluation on entire training dataset"
        print "Score: " + str(best_model.score(X_s, y_s))
        print_overview(y,out)
        if not prediction_input_filename:
            print "\n-- Best model evaluation on prediction (test) data"
            print "Score: " + str(best_model.score(X_test_s, y_test_s))
            print_overview(y_test,out_test)

##    # Write the results to a single file
##    if not prediction_input_filename:
##        with open('ResultsLog.csv', 'a') as fo:
##            fo.seek(0, 2)
##            summary = [fp,prediction_fraction,k,comp_time,
##                       random_prediction_dataset,min_max_normalization,
##                       holidays_flag,join_holidays,month_flag,n_vals_in_past_day,reduce_size]
##            summary.append(str(best_model.best_estimator_).split('(')[0])
##            summary.append(best_model.best_score_)
##            if prediction_input_filename:
##                summary.append('NA - prediction only')
##            else:
##                summary.append(best_model.score(X_test_s, y_test_s))
##            for m in models:
##                summary.append(str(m.best_estimator_).split('(')[0])
##                summary.append(m.best_score_)
##            fo.write(','.join(map(str, summary)))
##            fo.write('\n')
    
    if prediction_input_filename:
        if verbose: print "\n=== Writing prediction file ==="
        with open(os.path.join(op,prediction_output_filename), 'w') as fo:
            headers.pop() # remove timestamp row
            for i in range(len(headers)):
                fo.write(','.join(map(str, headers[i])))
                fo.write('\n')
            fo.write(datetime_col_name + ',load,\n')
            for i in range(len(predict_datetimes)):
                fo.write(str(predict_datetimes[i]) + ',' + str(out_test[i]) + ',\n')
    
    if save:
        write_model_results(models, op)
        data_folder = os.path.join(op,'Data')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        np.savetxt(os.path.join(data_folder,'Training_Input.txt'), X)
        np.savetxt(os.path.join(data_folder,'Training_Target.txt'), y)
        np.savetxt(os.path.join(data_folder,'Normalized_Training_Input.txt'), X_s)
        np.savetxt(os.path.join(data_folder,'Normalized_Training_Target.txt'), y_s)
        np.savetxt(os.path.join(data_folder,'Prediction_Input.txt'), X_test)
        np.savetxt(os.path.join(data_folder,'Prediction_Output.txt'), out_test)
        np.savetxt(os.path.join(data_folder,'Normalized_Prediction_Input.txt'), X_test_s)
        np.savetxt(os.path.join(data_folder,'Normalized_Prediction_Output.txt'), out_test_s)
        if not prediction_input_filename:
            np.savetxt(os.path.join(data_folder,'Prediction_Target.txt'), y_test)
            np.savetxt(os.path.join(data_folder,'Normalized_Prediction_Target.txt'), y_test_s)

    if plot and not prediction_input_filename: plot_comparison(out_test, y_test)

    print "\nDone!\n"