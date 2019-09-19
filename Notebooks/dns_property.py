# Data & Stuff - Machine Learning for Property Valuation
'''
This script contains helper functions for optimising models for C-Value.
'''
# Import required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


# * --------*
# | METRICS |
# * --------*
# RMSE metric
def rmse(y_true, preds):
    '''
    This function computes Root Mean Squared Error (RMSE) using `y_true` and `predictions`.
    '''
    return np.sqrt(mean_squared_error(y_true, preds))

# MAPE metric
def mape(y_true, preds):
    '''
    This function computes Mean Absolute Percentage Error (MAPE) using `y_true` and `predictions`.
    '''
    return np.mean(np.abs((preds - y_true) / y_true))


# * --------------------*
# | FEATURE ENGINEERING |
# * --------------------*

# Generate month lag statistics
def fe_mth(df, feat, mths_lag):
    
    # Generate forward month
    dat = df.copy()
    dat['mth_l'] = dat.mth + pd.to_timedelta(31*mths_lag, unit='d')
    dat['mth_l'] = pd.to_datetime(dat.mth_l.dt.year.astype(str) + '-' + dat.mth_l.dt.month.astype(str) + '-01')
    
    # Compute and append median
    temp_median = dat.groupby(['mth_l', feat])['target'].median()
    temp_median.index.names = ['mth', feat]
    temp_median = temp_median.rename('m' + str(mths_lag) + '_' + feat + '_median')
    dat = dat.merge(temp_median, on=['mth', feat], how='left').fillna(0)
    
    # Compute and append standard deviation
    temp_sd = dat.groupby(['mth_l', feat])['target'].std()
    temp_sd.index.names = ['mth', feat]
    temp_sd = temp_sd.rename('m' + str(mths_lag) + '_' + feat + '_sd')
    dat = dat.merge(temp_sd, on=['mth', feat], how='left').fillna(0)

    # Output
    return dat

# Generate quarter lag statistics
def fe_qtr(df, feat, qtrs_lag):

    # Create date lookup
    dat = df.copy()
    
    if not 'qtr_l' in dat.columns:
        qtrs = pd.DataFrame(dat.set_index('mth').groupby('mth').resample('Q', label='left')['target'].median())
        qtrs.index.names = ['mth', 'qtr']
        qtrs = qtrs.reset_index()

        # Lag date
        qtrs['qtr_l'] = pd.to_datetime((qtrs.qtr + pd.to_timedelta(3*qtrs_lag, unit='M')).dt.date)

        # Standardise quarter as start of month
        qtrs['qtr'] = pd.to_datetime(qtrs.qtr.dt.year.astype(str) + '-' + qtrs.qtr.dt.month.astype(str) + '-01')
        qtrs['qtr_l'] = pd.to_datetime(qtrs.qtr_l.dt.year.astype(str) + '-' + qtrs.qtr_l.dt.month.astype(str) + '-01')

        # Merge lagged quarter with main dataset and rename
        qtr_map = qtrs.set_index('mth').drop('target', axis=1)
        dat = dat.merge(qtr_map, on='mth')

    # Compute and append median
    temp_median = pd.DataFrame(dat.groupby(['qtr_l', feat])['target'].median())
    temp_median.index.names = ['qtr', feat]
    temp_median.columns = ['q' + str(qtrs_lag) + '_' + feat + '_median']
    dat = dat.merge(temp_median, on=['qtr', feat], how='left').fillna(0)
    
    # Compute and append median
    temp_sd = pd.DataFrame(dat.groupby(['qtr_l', feat])['target'].std())
    temp_sd.index.names = ['qtr', feat]
    temp_sd.columns = ['q' + str(qtrs_lag) + '_' + feat + '_sd']
    dat = dat.merge(temp_sd, on=['qtr', feat], how='left').fillna(0)
    
    # Output
    return dat


# * -----------------*
# | MODEL EVALUATION |
# * -----------------*

# CV function
def custom_cv(rgr, X, y, cv_obj, tfidf_cols, stop_words, norm=False, ppd=True):
    
    '''
    This function takes the following arguments:
    - rgr: Regressor (`sklearn` API only)
    - X: All predictors
    - y: Target
    - cv_obj: Cross-validation object from `sklearn`
    - tfidf_cols: Columns to convert into TF-IDF binary features
    - stop_words: Stop words to exclude
    - norm: Boolean on whether to normalise features (extremely important for K-NN)
    - ppd: Whether to compute (absolute) purchase price deviation or MAPE
    
    It returns the following for all iterations of CV:
    - R-squared
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)
    - Mean (absolute) Purchase Price Deviation (PPD)
    - Percentage of predictions that were within the following thresholds of deviation from true price:
        - 5%
        - 10%
        - 20%
    - Detailed (absolute) PPD from all iterations (for computation of C-Value equivalent to Zoom Value)
    
    '''
    
    # Print
    print('# ---- COMPUTING C-VALUE ---- #')
    
    # Initialise lists
    all_r2 = []
    all_rmse = []
    all_mae = []
    all_mape = []
    all_ppd = []
    all_plus05 = []
    all_plus10 = []
    all_plus20 = []
    ppd_data = []
    
    counter = 1
    
    # Count iterations
    total_iter = cv_obj.get_n_splits()
    
    for tr_idx, te_idx in cv_obj.split(X, y):
        
        # Update
        print('Running iteration %s of %s...' % (counter, total_iter))
        counter += 1
        
        # Split data
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        
        # Generate TF-IDF binary features
        for col in tfidf_cols:
        
            # Initialise TF-IDF vectoriser
            tfv = TfidfVectorizer(stop_words=stop_words, use_idf=False, smooth_idf=False, norm=False, ngram_range=(1,1), lowercase=True, binary=True)
            
            # Fit data without numbers
            tfv.fit(X_tr[col].str.replace('[0-9]', ''))
            
            # Generate features
            st_tr = pd.DataFrame(tfv.transform(X_tr[col]).A, columns=pd.DataFrame([tfv.vocabulary_]).columns)
            st_te = pd.DataFrame(tfv.transform(X_te[col]).A, columns=pd.DataFrame([tfv.vocabulary_]).columns)
            X_tr = X_tr.drop(col, axis=1)
            X_te = X_te.drop(col, axis=1)
            
            # Remove words with a count less than 20
            st_counts = st_tr.sum()
            drop_vars = st_counts.index[(st_counts < 20)]
            
            # Concatenate TF-IDF features to main training data
            X_tr = pd.concat([X_tr.reset_index(drop=True), st_tr], axis=1)
            X_te = pd.concat([X_te.reset_index(drop=True), st_te], axis=1)
            X_tr = X_tr.drop(drop_vars, axis=1)
            X_te = X_te.drop(drop_vars, axis=1)
        
        # Normalise data
        if norm:
            mms = MinMaxScaler()
            X_tr_scaled = mms.fit_transform(X_tr)
            X_te_scaled = mms.transform(X_te)
        else:
            X_tr_scaled = X_tr
            X_te_scaled = X_te
        
        # Fit model
        rgr.fit(X_tr_scaled, y_tr)

        # Predict
        temp_preds = rgr.predict(X_te_scaled)
        
        # Compute MAPE / PPD
        if ppd:
            temp_apreds = np.abs((y_te - temp_preds) / temp_preds)
        else:
            temp_apreds = np.abs((temp_preds - y_te) / y_te)
        
        # Compute ppd
        temp_dev = np.abs((y_te - temp_preds) / temp_preds)
        
        # Compute metrics
        temp_r2 = r2_score(y_te, temp_preds)
        temp_rmse = rmse(y_te, temp_preds)
        temp_mae = mean_absolute_error(y_te, temp_preds)
        temp_mape = mape(y_te, temp_preds)
        temp_ppd = np.median(temp_dev)
        
        # Append
        all_r2.append(temp_r2)
        all_rmse.append(temp_rmse)
        all_mae.append(temp_mae)
        all_mape.append(temp_mape)
        all_ppd.append(temp_ppd)
        all_plus05.append(np.sum(temp_apreds < 0.05)/temp_apreds.shape[0])
        all_plus10.append(np.sum(temp_apreds < 0.10)/temp_apreds.shape[0])
        all_plus20.append(np.sum(temp_apreds < 0.20)/temp_apreds.shape[0])
        ppd_data.append(temp_dev)
    
    # Print results
    print()
    print('[MODEL STATISTICS]')
    print('M-PPD:', '{:.2f}%'.format(100*np.mean(all_ppd)))
    print('PPD Deviation:')
    print('  +-5.0%:', '{:.2f}%'.format(100*np.mean(all_plus05)))
    print('  +-10%:', '{:.2f}%'.format(100*np.mean(all_plus10)))
    print('  +-20%:', '{:.2f}%'.format(100*np.mean(all_plus20)))
    print()

    print('MAPE:', '{:.2f}%'.format(100*np.mean(all_mape)))
    print('MAE:', '${:.0f}'.format(np.round(np.mean(all_mae))))
    print('RMSE:', '${:.0f}'.format(np.round(np.mean(all_rmse))))
    print('R2:', '{:.2f}%'.format(100*np.mean(all_r2)))
    
    # Output
    return (all_r2, all_rmse, all_mae, all_mape, all_ppd, all_plus05, all_plus10, all_plus20, ppd_data)

# Predictions for Time Series
def custom_ts(rgr, X, y, timevar, start, tfidf_cols, stop_words, dropout=0.1, norm=False, ppd=True, random_state=123):
    
    '''
    This function takes the following arguments:
    - rgr: Regressor (`sklearn` API only)
    - X: All predictors
    - y: Target
    - timevar: Feature representing the timesteps
    - start: Start date for test sets
    - tfidf_cols: Columns to convert into TF-IDF binary features
    - stop_words: Stop words to exclude
    - dropout: percentage of samples to drop from each month in dataset
    - norm: Boolean on whether to normalise features (extremely important for K-NN)
    - ppd: Boolean on whether to compute (absolute) purchase price deviation or MAPE
    - random_state: Random seed for dropout
    
    It plots and returns the following for all test sets:
    - Mean (absolute) Purchase Price Deviation (PPD)
    - Mean Absolute Percentage Error (MAPE)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R-squared
    
    - Percentage of predictions that were within the following thresholds of deviation from true price:
        - 5%
        - 10%
        - 20%
    - Detailed (absolute) PPD from test iterations in chronological order (for computation of C-Value equivalent to Zoom Value)
    
    '''
    
    # Print
    print('# ---- COMPUTING C-VALUE ---- #')

    # Initialise lists
    all_r2 = []
    all_rmse = []
    all_mae = []
    all_mape = []
    all_ppd = []
    all_plus05 = []
    all_plus10 = []
    all_plus20 = []
    ppd_data = []

    # Sort data
    df_full = pd.concat([X, y], axis=1)
    df_full = df_full.reset_index(drop=True).sort_values(timevar).reset_index(drop=True)

    # Configure train, test and validation sets
    ts_all = df_full[timevar].unique()
    ts_train = df_full[timevar][df_full[timevar] < start].unique()
    # ts_val = X_sorted[timevar][X_sorted[timevar] > start].unique()[:-1]
    ts_test = df_full[timevar][df_full[timevar] >= start].unique()

    # Initialise new dataframe
    df_drop = pd.DataFrame()

    # Generate random numbers
    np.random.seed(random_state)
    seeds = np.random.randint(1, 100, len(ts_all))

    # Drop data
    for i in range(len(seeds)):
        df_drop = pd.concat([df_drop, df_full[df_full[timevar] == ts_all[i]].sample(frac=1-dropout, random_state=seeds[i])])

    # Split data
    X_new = df_drop.drop('target', axis=1).reset_index(drop=True)
    y_new = df_drop['target'].reset_index(drop=True)

    # Initialise statuses
    counter = 1
    total_iter = len(ts_test)

    # Train and predict on each timestep
    for t in ts_test:

        # Update
        print('Running iteration %s of %s...' % (counter, total_iter))
        counter += 1

        # Split data
        X_tr = X_new[X_new[timevar] < t].drop(timevar, axis=1)
        X_te = X_new[X_new[timevar] == t].drop(timevar, axis=1)
        y_tr = y_new[X_new[timevar] < t]
        y_te = y_new[X_new[timevar] == t]

        # Generate TF-IDF binary features
        for col in tfidf_cols:

            # Initialise TF-IDF vectoriser
            tfv = TfidfVectorizer(stop_words=stop_words, use_idf=False, smooth_idf=False, norm=False, ngram_range=(1,1), lowercase=True, binary=True)

            # Fit data without numbers
            tfv.fit(X_tr[col].str.replace('[0-9]', ''))

            # Generate features
            st_tr = pd.DataFrame(tfv.transform(X_tr[col]).A, columns=pd.DataFrame([tfv.vocabulary_]).columns)
            st_te = pd.DataFrame(tfv.transform(X_te[col]).A, columns=pd.DataFrame([tfv.vocabulary_]).columns)
            X_tr = X_tr.drop(col, axis=1)
            X_te = X_te.drop(col, axis=1)

            # Remove words with a count less than 20
            st_counts = st_tr.sum()
            drop_vars = st_counts.index[(st_counts < 20)]

            # Concatenate TF-IDF features to main training data
            X_tr = pd.concat([X_tr.reset_index(drop=True), st_tr], axis=1)
            X_te = pd.concat([X_te.reset_index(drop=True), st_te], axis=1)
            X_tr = X_tr.drop(drop_vars, axis=1)
            X_te = X_te.drop(drop_vars, axis=1)

        # Normalise data
        if norm:
            mms = MinMaxScaler()
            X_tr_scaled = mms.fit_transform(X_tr)
            X_te_scaled = mms.transform(X_te)
        else:
            X_tr_scaled = X_tr
            X_te_scaled = X_te

        # Fit model
        rgr.fit(X_tr_scaled, y_tr)

        # Predict
        temp_preds = rgr.predict(X_te_scaled)

        # Compute MAPE / PPD
        if ppd:
            temp_apreds = np.abs((y_te - temp_preds) / temp_preds)
        else:
            temp_apreds = np.abs((temp_preds - y_te) / y_te)

        # Compute ppd
        temp_dev = (y_te - temp_preds) / temp_preds

        # Compute metrics
        temp_r2 = r2_score(y_te, temp_preds)
        temp_rmse = rmse(y_te, temp_preds)
        temp_mae = mean_absolute_error(y_te, temp_preds)
        temp_mape = mape(y_te, temp_preds)
        temp_ppd = np.median(temp_apreds)

        # Append
        all_r2.append(temp_r2)
        all_rmse.append(temp_rmse)
        all_mae.append(temp_mae)
        all_mape.append(temp_mape)
        all_ppd.append(temp_ppd)
        all_plus05.append(np.sum(temp_apreds < 0.05)/temp_apreds.shape[0])
        all_plus10.append(np.sum(temp_apreds < 0.10)/temp_apreds.shape[0])
        all_plus20.append(np.sum(temp_apreds < 0.20)/temp_apreds.shape[0])
        ppd_data.append(temp_dev)

    # Combine percentage price deviations
    ppd_hist = pd.Series()
    for i in range(len(ppd_data)):
        ppd_hist = ppd_hist.append(ppd_data[i])
    
    # Line spacing
    print()
    
    # Set up figure
    fig = plt.figure(figsize=(15,15))
    gs = plt.GridSpec(3, 4)
    plt.subplots_adjust(bottom=0.0001)

    # Plot PPD
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(ts_test, np.array(all_ppd)*100, color='#133056')
    ax1.set_title('Purchase Price Deviation (%)\nMean: ' + '{:.2f}%'.format(np.mean(np.array(all_ppd)*100)))
    ax1.hlines(y=np.mean(np.array(all_ppd)*100), xmin=np.min(ts_test), xmax=np.max(ts_test), linestyle='dotted', color='red')

    # Plot MAPE
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.plot(ts_test, np.array(all_mape)*100, color='#133056')
    ax2.set_title('Mean Absolute Percentage Error (%)\nMean: ' + '{:.2f}%'.format(np.mean(np.array(all_mape)*100)))
    ax2.hlines(y=np.mean(np.array(all_mape)*100), xmin=np.min(ts_test), xmax=np.max(ts_test), linestyle='dotted', color='red')

    # Plot MAE
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(ts_test, all_mae, color='#133056')
    ax3.set_title('Mean Absolute Error ($)\nMean: ' + '${:,.2f}'.format(np.mean(np.array(all_mae))))
    ax3.hlines(y=np.mean(np.array(all_mae)), xmin=np.min(ts_test), xmax=np.max(ts_test), linestyle='dotted', color='red')

    # Plot RMSE
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.plot(ts_test, all_rmse, color='#133056')
    ax4.set_title('Root Mean Squared Error ($)\nMean: ' + '${:,.2f}'.format(np.mean(np.array(all_rmse))))
    ax4.hlines(y=np.mean(np.array(all_rmse)), xmin=np.min(ts_test), xmax=np.max(ts_test), linestyle='dotted', color='red')
    
    
    
    # Plot histogram
    ax5 = fig.add_subplot(gs[2:4, :])
    yhist, _,  _ = ax5.hist(ppd_hist*100, bins=60, color='#6fceb0', alpha=0.7)
    ymax = np.max(yhist)
    ax5.set_title('Purchase Price Deviation (%)')
    ax5.vlines(x=5, ymin=0, ymax=ymax, linestyle='dashed', linewidth=1, color='#133056', label='Within 5%: ' + '{:.2f}%'.format(np.mean(np.abs(ppd_hist) < 0.05)*100))
    ax5.vlines(x=-5, ymin=0, ymax=ymax, linestyle='dashed', linewidth=1, color='#133056')
    ax5.vlines(x=10, ymin=0, ymax=ymax, linestyle='dotted', color='#f85b74', label='Within 10%: ' + '{:.2f}%'.format(np.mean(np.abs(ppd_hist) < 0.1)*100))
    ax5.vlines(x=-10, ymin=0, ymax=ymax, linestyle='dotted', color='#f85b74')
    ax5.vlines(x=20, ymin=0, ymax=ymax, linestyle='dotted', color='#ff9966', label='Within 20%: ' + '{:.2f}%'.format(np.mean(np.abs(ppd_hist)< 0.2)*100))
    ax5.vlines(x=-20, ymin=0, ymax=ymax, linestyle='dotted', color='#ff9966')
    ax5.legend()
    
    # Title
    fig.suptitle('Model Results', fontsize=20, fontweight='bold')
    plt.show()

    # Print results
    print('[MODEL STATISTICS]')
    print('M-PPD:', '{:.2f}%'.format(100*np.mean(all_ppd)))
    print('PPD Deviation:')
    print('  +-5.0%:', '{:.2f}%'.format(100*np.mean(all_plus05)))
    print('  +-10%:', '{:.2f}%'.format(100*np.mean(all_plus10)))
    print('  +-20%:', '{:.2f}%'.format(100*np.mean(all_plus20)))
    print()

    print('MAPE:', '{:.2f}%'.format(100*np.mean(all_mape)))
    print('MAE:', '${:.0f}'.format(np.round(np.mean(all_mae))))
    print('RMSE:', '${:.0f}'.format(np.round(np.mean(all_rmse))))
    print('R2:', '{:.2f}%'.format(100*np.mean(all_r2)))

    # Output
    return (all_r2, all_rmse, all_mae, all_mape, all_ppd, all_plus05, all_plus10, all_plus20, ppd_hist)