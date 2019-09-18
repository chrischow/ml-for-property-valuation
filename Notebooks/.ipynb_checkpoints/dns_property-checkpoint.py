# Data & Stuff - Machine Learning for Property Valuation
'''
This script contains helper functions for optimising models for C-Value.
'''
# Import required modules
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
def custom_cv(rgr, X, y, cv_obj, tfidf_cols, norm=False, ppd=True):
    
    '''
    This function takes the following arguments:
    - Regressor (`sklearn` API only)
    - All predictors (X)
    - Target (y)
    - Cross-validation object from `sklearn`
    - Columns to convert into TF-IDF binary features
    - Norm: Boolean on whether to normalise features (extremely important for K-NN)
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
    
    # Configure stop words
    stop_words = [
            'road', 'avenue', 'street', 'drive', 'lorong', 'crescent', 'view', 'rise'
            'jalan', 'lane', 'walk', 'link', 'bukit',
            'rd', 'ave', 'st', 'dr', 'lor', 'cres', 'jln', 'pl'
        ]
    
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
    print('R2:', '{:.2f}%'.format(100*np.mean(all_r2)))
    print('MAE:', '${:.0f}'.format(np.round(np.mean(all_mae))))
    print('RMSE:', '${:.0f}'.format(np.round(np.mean(all_rmse))))
    print('MAPE:', '{:.2f}%'.format(100*np.mean(all_mape)))
    print('M-PPD:', '{:.2f}%'.format(100*np.mean(all_ppd)))
    print('+-5.0%:', '{:.2f}%'.format(100*np.mean(all_plus05)))
    print('+-10%:', '{:.2f}%'.format(100*np.mean(all_plus10)))
    print('+-20%:', '{:.2f}%'.format(100*np.mean(all_plus20)))
    
    # Output
    return (all_r2, all_rmse, all_mae, all_mape, all_ppd, all_plus05, all_plus10, all_plus20, ppd_data)