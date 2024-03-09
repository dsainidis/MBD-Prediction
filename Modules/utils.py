def read_data(file_path, drop_columns = [], exclude_dtypes = [], exclude_years = [], year_column = 'year'):
    
    import pandas as pd
    
    dataset = pd.read_csv(file_path)
    
    if len(exclude_dtypes) != 0:
        dataset = dataset.select_dtypes(exclude=exclude_dtypes)

    if len(exclude_years) > 0:
        dataset = dataset[~dataset[year_column].isin(exclude_years)]
        
    if len(drop_columns) != 0:
        dataset.drop(columns = drop_columns, inplace = True)
        
    dataset.reset_index(drop = True, inplace = True)
        
    return dataset


def np_nth_smallest(array, n):
    import numpy as np
    
    return np.partition(array, n-1)[n-1]


def plot_imbalance(dataframe, target_column = 'case', x_label = 'Number of cases', y_label = ['non_case', 'case'], tick_size = 14, label_size = 18, text_size = 12, figure_size = (12, 4)):
    
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    data_year_non_cases = dataframe.loc[dataframe[target_column] == 0]
    data_year_cases = dataframe.loc[dataframe[target_column] != 0]
    
    counts0 = len(data_year_non_cases)
    counts1 = data_year_cases[target_column].sum()
    
    fig, ax = plt.subplots(figsize = figure_size)
    
    x_max = math.ceil(max(counts0, counts1) + (max(counts0, counts1) * .1))
    step = round(math.ceil(x_max * .1) / 1000) * 1000
    if step == 0:
        step = 500
    
    plt.xlabel(x_label, size = label_size)
    
    plt.xticks(np.arange(0, x_max, step = step), size = tick_size, rotation = 45)
    plt.xlim([0, x_max])
    
    imbalance = round((max(counts0, counts1)/min(counts0, counts1)))
    
    plt.text(max(counts0, counts1) - (max(counts0, counts1) * .15), .45, f'imbalance ~ {imbalance}:1', size=text_size)
    
    bars = ax.barh(y_label, (counts0, counts1))
    ax.bar_label(bars, label_type= 'edge', size = text_size)
    plt.show()

def dataframe_describe(dataframe, by_column = 'year', target_column = 'case'):
    
    import pandas as pd
    
    result = pd.DataFrame(columns = ['year', 'non-case', 'case', 'total', '0/1 ratio'])
    
    for year in dataframe[by_column].drop_duplicates().sort_values():
        data_year = dataframe.loc[dataframe[by_column] == year]
        data_year_non_cases = data_year.loc[data_year[target_column] == 0]
        data_year_cases = data_year.loc[data_year[target_column] != 0]
    
        counts0 = len(data_year_non_cases)
        counts1 = data_year_cases[target_column].sum()

           
        ratio = counts0/counts1

        new_row = {by_column: year,
                    'non-case': counts0,
                    'case': counts1,
                    'total': (counts0 + counts1),
                    '0/1 ratio': f'{ratio:.2f}%'}
        
        result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
        
        
    non_case_tot = result['non-case'].sum()
    case_tot = result['case'].sum()
    total_tot = result['total'].sum()
    ratio_tot = non_case_tot/case_tot

    new_row_tot = {by_column: 'Total:',
                   'non-case' : non_case_tot,
                   'case': case_tot,
                   'total': total_tot,
                   '0/1 ratio': f'{ratio_tot:.2f}%'}
    
    result = pd.concat([result, pd.DataFrame([new_row_tot])], ignore_index=True)
        
    return result


def yearCV_split(dataframe, exclude_years_train = [], year_column = 'year', target_column = 'case'):
    
    years_sorted = dataframe[year_column].drop_duplicates().sort_values()
    
    for year in years_sorted:

        data_train = dataframe.loc[~dataframe[year_column].isin([year]+exclude_years_train)]

        # data_train = dataframe.loc[dataframe[year_column] != year]
        # if len(exclude_years_train) > 0:
        #      data_train = data_train.loc[~data_train[year_column].isin(exclude_years_train)]

        data_test = dataframe.loc[dataframe[year_column] == year]
        #data_test = data_test.drop_duplicates(keep='first')
        
        yield data_train.index, data_test.index

def yearCV_split_vector(dataframe, exclude_years_train = [], year_column = 'year', target_column = 'case'):
    
    years_sorted = dataframe[year_column].drop_duplicates().sort_values()
    
    for year in years_sorted:

        data_train = dataframe.loc[~dataframe[year_column].isin([year]+exclude_years_train)]

        # data_train = dataframe.loc[dataframe[year_column] != year]
        # if len(exclude_years_train) > 0:
        #      data_train = data_train.loc[~data_train[year_column].isin(exclude_years_train)]

        data_test = dataframe.loc[dataframe[year_column] == year]
        #data_test = data_test.drop_duplicates(keep='first')
        
        yield data_train.index, data_test.index


def transform_data(X_train, X_test, y_train, y_test, random_r, nearmiss, smote, scaler_key = 'minmax', exclude_features = []):
    
    import math
    import random
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
    from sklearn.impute import KNNImputer
    from imblearn.under_sampling import NearMiss 
    from imblearn.over_sampling import BorderlineSMOTE
    
    scalers = {'minmax': MinMaxScaler(),
               'standard': StandardScaler(),
               'robust': RobustScaler(),
               'abs': MaxAbsScaler()}

    columns_Xtrain = X_train.columns
    
    if(random_r != 0):
        nan_index = []
        zero_index = []
        
        for index, row in X_train.iterrows():
            if row.isnull().values.any():
                nan_index.append(index)
                if y_train.iloc[index] == 0:
                    zero_index.append(index)
            
        nan_set = set(nan_index)
        zero_set = set(zero_index)
        
        intersection = list(nan_set.intersection(zero_set))
        sampling_size = math.ceil(len(intersection) * random_r)
        removed = random.sample(intersection, sampling_size)
        
        X_train = X_train.drop(index = removed, axis = 1)
        y_train = y_train.drop(index = removed, axis = 1)
    
    
    scaler = scalers.get(scaler_key, MinMaxScaler())
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
        
    imputer = KNNImputer()
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
        
    try:
        class_0_counts = y_train.value_counts()[0]
    except KeyError:
        class_0_counts = 0
    try:
        class_1_counts = y_train.value_counts()[1]
    except KeyError:
        class_1_counts = 0
        
    if (nearmiss != 0):
        nm = NearMiss(sampling_strategy = {0: math.ceil(class_0_counts * (1 - nearmiss)), 1: class_1_counts}, version = 1, n_jobs = -1)
        X_train, y_train = nm.fit_resample(X_train, y_train)
    
    if (smote != 0):
        class_0_counts = y_train.value_counts()[0]
        sm = BorderlineSMOTE(sampling_strategy = {0: class_0_counts, 1: math.ceil(class_1_counts * (1 + smote))}, n_jobs = -1, random_state = 0)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    
    X_train_inversed = scaler.inverse_transform(X_train)
    X_train_df = pd.DataFrame(X_train_inversed, columns = columns_Xtrain)
    try:
        X_train_df = X_train_df.astype({'day':'int', 'month':'int', 'year':'int'})
    except KeyError:
        X_train_df = X_train_df.astype({'year':'int'})

    indicies_to_remove = []
    for item in exclude_features:
        indicies_to_remove.append(columns_Xtrain.get_loc(item))

    for index in indicies_to_remove:
        X_train = np.delete(X_train, index, axis=1)
        X_test = np.delete(X_test, index, axis=1)
        
    return X_train, X_test, y_train, y_test, X_train_df, scaler, imputer

def transform_data_new(X_train, X_test, y_train, y_test, random_r, nearmiss, smote, scaler_key = 'minmax', majority_class = 0, exclude_features = []):
    
    import math
    import random
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
    from sklearn.impute import KNNImputer
    from imblearn.under_sampling import NearMiss 
    from imblearn.over_sampling import BorderlineSMOTE
    
    scalers = {'minmax': MinMaxScaler(),
               'standard': StandardScaler(),
               'robust': RobustScaler(),
               'abs': MaxAbsScaler()}

    columns_Xtrain = X_train.columns
    
    if(random_r != 0):
        nan_index = []
        target_index = []
        
        for feature, row in X_train.iterrows():
            if row.isnull().values.any():
                nan_index.append(feature)
                if y_train.iloc[feature] == majority_class:
                    target_index.append(feature)
            
        nan_set = set(nan_index)
        zero_set = set(target_index)
        
        intersection = list(nan_set.intersection(zero_set))
        sampling_size = math.ceil(len(intersection) * random_r)
        removed = random.sample(intersection, sampling_size)
        
        X_train = X_train.drop(index = removed, axis = 1)
        y_train = y_train.drop(index = removed, axis = 1)
    
    
    scaler = scalers.get(scaler_key, MinMaxScaler())
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
        
    imputer = KNNImputer()
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    y_train_value_counts = y_train.value_counts()
    class_0_counts = y_train_value_counts.get(key = 0) if y_train_value_counts.get(key = 0) is not None else 0
    class_1_counts = y_train_value_counts.get(key = 1) if y_train_value_counts.get(key = 1) is not None else 0
     
    if (nearmiss != 0):

        if majority_class == 0:
            nm = NearMiss(sampling_strategy = {0: math.ceil(class_0_counts * (1 - nearmiss)), 1: class_1_counts}, version = 1, n_jobs = -1)
        elif majority_class == 1:
            nm = NearMiss(sampling_strategy = {0: class_0_counts, 1: math.ceil(class_1_counts * (1 - nearmiss))}, version = 1, n_jobs = -1)

        X_train, y_train = nm.fit_resample(X_train, y_train)
    
    if (smote != 0):
        y_train_value_counts = y_train.value_counts()
        class_0_counts = y_train_value_counts.get(key = 0) if y_train_value_counts.get(key = 0) is not None else 0
        class_1_counts = y_train_value_counts.get(key = 1) if y_train_value_counts.get(key = 1) is not None else 0
        
        if majority_class == 0:
            sm = BorderlineSMOTE(sampling_strategy = {0: class_0_counts, 1: math.ceil(class_1_counts * (1 + smote))}, n_jobs = -1, random_state = 0)
        elif majority_class == 1:
            sm = BorderlineSMOTE(sampling_strategy = {0: math.ceil(class_0_counts * (1 + smote)), 1: class_1_counts}, n_jobs = -1, random_state = 0)

        X_train, y_train = sm.fit_resample(X_train, y_train)
    
    X_train_inversed = scaler.inverse_transform(X_train)
    X_train_df = pd.DataFrame(X_train_inversed, columns = columns_Xtrain)
    try:
        X_train_df = X_train_df.astype({'day':'int', 'month':'int', 'year':'int'})
    except KeyError:
        X_train_df = X_train_df.astype({'year':'int'})

    features_to_remove = []
    for item in exclude_features:
        features_to_remove.append(columns_Xtrain.get_loc(item))

    for feature in features_to_remove:
        X_train = np.delete(X_train, feature, axis=1)
        X_test = np.delete(X_test, feature, axis=1)
        
    return X_train, X_test, y_train, y_test, X_train_df, scaler, imputer

def train_test_split_transform(data_train, data_test, nearmiss, smote, target_col = 'case', scaler_key = 'minmax', majority_class = 0, exclude_features = []):
    
    import math
    import random
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
    from sklearn.impute import KNNImputer
    from imblearn.under_sampling import NearMiss 
    from imblearn.over_sampling import BorderlineSMOTE
    
    scalers = {'minmax': MinMaxScaler(),
               'standard': StandardScaler(),
               'robust': RobustScaler(),
               'abs': MaxAbsScaler()}

    object_cols = data_train.select_dtypes(include=['object']).columns.to_list()
    removed_cols = exclude_features

    X_train = data_train.drop(columns = [target_col] + object_cols + removed_cols)
    y_train = data_train[target_col]
    X_test = data_test.drop(columns = [target_col] + object_cols + removed_cols)
    y_test = data_test[target_col]

    scaler = scalers.get(scaler_key, MinMaxScaler())
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
        
    imputer = KNNImputer()
    imputer.fit(X_train)
    X_train_imputed = imputer.transform(X_train_scaled)
    X_test_imputed = imputer.transform(X_test_scaled)

    y_train_value_counts = y_train.value_counts()
    class_0_counts = y_train_value_counts.get(key = 0) if y_train_value_counts.get(key = 0) is not None else 0
    class_1_counts = y_train_value_counts.get(key = 1) if y_train_value_counts.get(key = 1) is not None else 0
        
    if (nearmiss != 0):
        if majority_class == 0:
            nm = NearMiss(sampling_strategy = {0: math.ceil(class_0_counts * (1 - nearmiss)), 1: class_1_counts}, version = 1, n_jobs = -1)
        elif majority_class == 1:
            nm = NearMiss(sampling_strategy = {0: class_0_counts, 1: math.ceil(class_1_counts * (1 - nearmiss))}, version = 1, n_jobs = -1)

        X_train_resampled, y_train_resampled = nm.fit_resample(X_train_imputed, y_train)

        y_train_value_counts = y_train_resampled.value_counts()
        class_0_counts = y_train_value_counts.get(key = 0) if y_train_value_counts.get(key = 0) is not None else 0
        class_1_counts = y_train_value_counts.get(key = 1) if y_train_value_counts.get(key = 1) is not None else 0
        

    if (smote != 0):
        if majority_class == 0:
            sm = BorderlineSMOTE(sampling_strategy = {0: class_0_counts, 1: math.ceil(class_1_counts * (1 + smote))}, n_jobs = -1, random_state = 0)
        elif majority_class == 1:
            sm = BorderlineSMOTE(sampling_strategy = {0: math.ceil(class_0_counts * (1 + smote)), 1: class_1_counts}, n_jobs = -1, random_state = 0)

        if (nearmiss != 0):
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train_resampled, y_train_resampled)
        else:
            X_train_resampled, y_train_resampled = sm.fit_resample(X_train_imputed, y_train)

    X_train_returned = X_train_imputed if (nearmiss == 0 and smote == 0) else X_train_resampled
    y_train_returned = y_train if (nearmiss == 0 and smote == 0) else y_train_resampled
    X_test_returned = X_test_imputed
    y_test_returned = y_test
        
    return X_train_returned, X_test_returned, y_train_returned, y_test_returned, scaler, imputer

def calculate_weights(training_set):
    import math
    
    (non_cases, cases) = training_set.value_counts()
    w0 = 1
    try:
        w1 = math.ceil(non_cases/cases)
    except ZeroDivisionError:
        w1 = 10
        
    return w0,w1

def calculate_weights_new(training_set, multiplier = 1):
    import math

    value_counts = training_set.value_counts()
    non_cases = value_counts.get(key = 0) if value_counts.get(key = 0) is not None else 0
    cases = value_counts.get(key = 1) if value_counts.get(key = 1) is not None else 0
    
    if non_cases >= cases:
        w0 = 1
        try:
            w1 = round(non_cases/cases)
        except ZeroDivisionError:
            w1 = 10
        w1 = w1 * multiplier
    else:
        w1 = 1
        try:
            w0 = round(cases/non_cases)
        except ZeroDivisionError:
            w0 = 10
        w0 = w0 * multiplier
        
    return w0,w1

def calculate_scale_pos_weight(training_set, multiplier = 1):
    import math

    value_counts = training_set.value_counts()
    non_cases = value_counts.get(key = 0) if value_counts.get(key = 0) is not None else 0
    cases = value_counts.get(key = 1) if value_counts.get(key = 1) is not None else 0

    if non_cases == 0:
        w = 0.1
    elif cases == 0:
        w = 10
    else:
        w = non_cases/cases
    
    w = w * multiplier
    return w

def train_lin_model(model, X_train, y_train, explain = False):
    import shap

    model.fit(X_train, y_train)

    if explain:
        masker = shap.maskers.Independent(data = X_train)
        explainer = shap.LinearExplainer(model, masker = masker)
        shap_values = explainer.shap_values(X_train)
        return model, shap_values
    else:
        return model
    
def train_tree_model(model, X_train, y_train, explain = False):
    import shap

    model.fit(X_train, y_train)

    if explain:
        #masker = shap.maskers.Independent(data = X_train)
        #explainer = shap.TreeExplainer(model, masker = masker)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        return model, shap_values
    else:
        return model
    
def train_model(model, X_train, y_train, explain = False, explainer_key = 'default'):
    import shap

    trained_model = model.fit(X_train, y_train)

    if explain:
        #masker = shap.maskers.Independent(data = X_train)
        #explainer = shap.TreeExplainer(model, masker = masker)
        explainers = {'default': shap.Explainer(trained_model),
                      'linear': shap.LinearExplainer(trained_model),
                      'tree': shap.TreeExplainer(trained_model),
                  }
        explainer = explainers.get(explainer_key, shap.Explainer(trained_model))
        shap_values = explainer.shap_values(X_train)
        return trained_model, shap_values
    else:
        return trained_model

def predict_lin_model_new(trained_model, X_train, y_train, X_test, y_test, data_test, spatial_col = 'lau1', day_col = 'day', month_col = 'month', year_col = 'year', score_col = 'score', target_col = 'case'):
    import pandas as pd
    import numpy as np

    data_train_unique = np.unique(np.concatenate((X_train, y_train.to_numpy().reshape(-1, 1)), axis=1), axis = 0)

    X_train_unique = data_train_unique[:, :-1]
    y_train_unique = data_train_unique[:, -1:]

    train_probas = trained_model.predict_proba(X_train_unique)
    test_probas = trained_model.predict_proba(X_test)
    model_coef = trained_model.coef_

    train_result = pd.DataFrame()
    train_result[target_col] = pd.Series(y_train_unique.squeeze()).astype('int')
    train_result[score_col] = train_probas[:, 1].tolist()
    train_result.sort_values(by=[score_col], ascending = False, ignore_index = True, inplace = True)
    
    test_result = pd.DataFrame()
    test_result[spatial_col] = data_test[spatial_col].reset_index(drop = True)
    if day_col is not None:
        test_result[day_col] = data_test[day_col].reset_index(drop = True)
    if month_col is not None:
        test_result[month_col] = data_test[month_col].reset_index(drop = True)
    test_result[year_col] = data_test[year_col].reset_index(drop = True)
    test_result[target_col] = y_test.reset_index(drop = True).astype('int')
    test_result[score_col] = test_probas[:, 1].tolist()
    test_result.sort_values(by=[score_col], ascending = False, ignore_index = True, inplace = True)

    return train_result, test_result, model_coef

def predict_xgb_model(trained_model, X_train, y_train, X_test, y_test, data_test, spatial_col = 'lau1', day_col = 'day', month_col = 'month', year_col = 'year', score_col = 'score', target_col = 'case'):
    import pandas as pd
    import numpy as np

    data_train_unique = np.unique(np.concatenate((X_train, y_train.to_numpy().reshape(-1, 1)), axis=1), axis = 0)

    X_train_unique = data_train_unique[:, :-1]
    y_train_unique = data_train_unique[:, -1:]

    train_probas = trained_model.predict_proba(X_train_unique)
    test_probas = trained_model.predict_proba(X_test)
    model_ft = trained_model.feature_importances_

    train_result = pd.DataFrame()
    train_result[target_col] = pd.Series(y_train_unique.squeeze()).astype('int')
    train_result[score_col] = train_probas[:, 1].tolist()
    train_result.sort_values(by=[score_col], ascending = False, ignore_index = True, inplace = True)
    
    test_result = pd.DataFrame()
    test_result[spatial_col] = data_test[spatial_col].reset_index(drop = True)
    if day_col is not None:
        test_result[day_col] = data_test[day_col].reset_index(drop = True)
    if month_col is not None:
        test_result[month_col] = data_test[month_col].reset_index(drop = True)
    test_result[year_col] = data_test[year_col].reset_index(drop = True)
    test_result[target_col] = y_test.reset_index(drop = True).astype('int')
    test_result[score_col] = test_probas[:, 1].tolist()
    test_result.sort_values(by=[score_col], ascending = False, ignore_index = True, inplace = True)

    return train_result, test_result, model_ft

def predict_lin_model(trained_model, data_train, data_test, X_train, y_train, X_test, y_test, spatial_col = 'lau1', day_col = 'day', month_col = 'month', year_col = 'year', score_col = 'score', target_col = 'case'):
    import pandas as pd

    train_probas = trained_model.predict_proba(X_train)
    test_probas = trained_model.predict_proba(X_test)
    model_coef = trained_model.coef_

    train_df = pd.DataFrame()
    train_df['x'] = data_train['x'].reset_index(drop = True)
    train_df['y'] = data_train['y'].reset_index(drop = True)
    #train_df['municipality'] = data_train['lau1'].reset_index(drop = True)
    if day_col is not None:
        train_df[day_col] = data_train[day_col].reset_index(drop = True)
    if month_col is not None:    
        train_df[month_col] = data_train[month_col].reset_index(drop = True)
    train_df[year_col] = data_train[year_col].reset_index(drop = True)
    train_df[target_col] = y_train.reset_index(drop = True).astype('int')
    train_df[score_col] = train_probas[:, 1].tolist()
    train_df.sort_values(by=[score_col], ascending = False, ignore_index = True, inplace = True)
    
    test_df = pd.DataFrame()
    test_df['x'] = data_test['x'].reset_index(drop = True)
    test_df['y'] = data_test['y'].reset_index(drop = True)
    test_df[spatial_col] = data_test[spatial_col].reset_index(drop = True)
    if day_col is not None:
        test_df[day_col] = data_test[day_col].reset_index(drop = True)
    if month_col is not None:
        test_df[month_col] = data_test[month_col].reset_index(drop = True)
    test_df[year_col] = data_test[year_col].reset_index(drop = True)
    test_df[target_col] = y_test.reset_index(drop = True).astype('int')
    test_df[score_col] = test_probas[:, 1].tolist()
    test_df.sort_values(by=[score_col], ascending = False, ignore_index = True, inplace = True)
    
    return train_df, test_df, model_coef

def inference_lin_model(trained_model, data_test, X_test, y_test, spatial_col = 'lau1', day_col = 'day', month_col = 'month', year_col = 'year', score_col = 'score', target_col = 'case'):
    import pandas as pd

    test_probas = trained_model.predict_proba(X_test)
    

    test_df = pd.DataFrame()
    test_df['x'] = data_test['x'].reset_index(drop = True)
    test_df['y'] = data_test['y'].reset_index(drop = True)
    test_df[spatial_col] = data_test[spatial_col].reset_index(drop = True)
    if day_col is not None:
        test_df[day_col] = data_test[day_col].reset_index(drop = True)
    if month_col is not None:
        test_df[month_col] = data_test[month_col].reset_index(drop = True)
    test_df[year_col] = data_test[year_col].reset_index(drop = True)
    test_df[target_col] = y_test.reset_index(drop = True).astype('int')
    test_df[score_col] = test_probas[:, 1].tolist()
    test_df.sort_values(by=[score_col], ascending = False, ignore_index = True, inplace = True)
    
    return test_df

def train_and_predict(model, data_train, data_test, X_train, y_train, X_test, y_test, spatial_col = 'lau1', day_col = 'day', month_col = 'month', target_col = 'case'):

    import pandas as pd

    model.fit(X_train, y_train)
    
    train_probas = model.predict_proba(X_train)
    test_probas = model.predict_proba(X_test)
    model_coef = model.coef_
    
    train_df = pd.DataFrame()
    train_df['x'] = data_train['x'].reset_index(drop = True)
    train_df['y'] = data_train['y'].reset_index(drop = True)
    #train_df['municipality'] = data_train['lau1'].reset_index(drop = True)
    if day_col is not None:
        train_df[day_col] = data_train[day_col].reset_index(drop = True)
    if month_col is not None:    
        train_df[month_col] = data_train[month_col].reset_index(drop = True)
    train_df['year'] = data_train['year'].reset_index(drop = True)
    train_df[target_col] = y_train.reset_index(drop = True).astype('int')
    train_df['score'] = train_probas[:, 1].tolist()
    train_df.sort_values(by=['score'], ascending = False, ignore_index = True, inplace = True)
    
    test_df = pd.DataFrame()
    test_df['x'] = data_test['x'].reset_index(drop = True)
    test_df['y'] = data_test['y'].reset_index(drop = True)
    test_df[spatial_col] = data_test[spatial_col].reset_index(drop = True)
    if day_col is not None:
        test_df[day_col] = data_test[day_col].reset_index(drop = True)
    if month_col is not None:
        test_df[month_col] = data_test[month_col].reset_index(drop = True)
    test_df['year'] = data_test['year'].reset_index(drop = True)
    test_df[target_col] = y_test.reset_index(drop = True).astype('int')
    test_df['score'] = test_probas[:, 1].tolist()
    test_df.sort_values(by=['score'], ascending = False, ignore_index = True, inplace = True)
    
    return train_df, test_df, model_coef

def plot_feature_importance(model_weights, feature_names, top = 0, title = None, x_label = 'Importance', y_label = 'Features', title_size = 22, tick_size = 14, label_size = 18, figure_size = (8, 18)):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    
    if (len(model_weights) > 1):
        weights = [sum(sub_list) / len(sub_list) for sub_list in zip(*model_weights)]
    else:
        weights = model_weights
    weights_df = pd.DataFrame({ 'features': list(feature_names),'weights': list(weights[0])})
    weights_df = weights_df.reindex(weights_df['weights'].abs().sort_values(ascending = False).index)  

    plt.figure(num = None, figsize = figure_size, dpi = 100, facecolor='w', edgecolor='b')
    if top == 0:
        sns.barplot(x = "weights", y = "features", data = weights_df)
    else:
        sns.barplot(x = "weights", y = "features", data = weights_df.head(top))

    if title != None:
        plt.title(title, size = title_size)
    plt.xticks(size = tick_size)
    plt.yticks(size = tick_size)
    plt.xlabel(x_label, size = label_size)
    plt.ylabel(y_label, size = label_size)

    return weights_df


def plot_trend_curve(results, score_col = 'score', cases = None, plot_min = False, plot_max = False, plot_avg = True, plot_cases = True, x_label = 'Year', y_label = 'Average Probability', tick_size = 14, label_size = 18, legend_size = 18, text_size = 12, figure_size = (8, 6), rotation_ang = 45):
    import numpy as np
    import matplotlib.pyplot as plt
    
    years = []
    p_min = []
    p_avg = []
    p_max = []
    if plot_cases:
        cases_norm = [float(i)/sum(cases) for i in cases]

    for year in results['year'].drop_duplicates().sort_values():
        years.append(year)
        p_min.append(results[results['year'] == year][score_col].min())
        p_avg.append(results[results['year'] == year][score_col].mean())
        p_max.append(results[results['year'] == year][score_col].max())  

    plt.figure(num = None, figsize = figure_size, dpi = 100, facecolor='w', edgecolor='b')
    
    if plot_min:
        plt.plot(years, p_min, 'green', label='Min Score')
        
    if plot_max:
        plt.plot(years, p_max, 'red', label='Max Score')
        
    if plot_avg:
        plt.plot(years, p_avg, 'blue', label='Avg Score')

    if plot_cases:    
        plt.plot(years, cases_norm, 'black', label='Cases (normalized)')
    
    plt.xticks(np.arange(min(years), max(years)+1), size = tick_size, rotation=rotation_ang)
    plt.yticks(np.arange(0, 1.1, step = 0.1), size = tick_size)
    plt.xlabel(x_label, size = label_size)
    plt.ylabel(y_label, size = label_size)
    plt.grid()
    plt.legend(prop={'size': legend_size})
    plt.show()

def plot_probability_curve(results, score_column = 'score', target_col = 'case', augment_samples = False, x_label = 'Probability', y_label = 'Ground Truth', tick_size = 14, label_size = 18, legend_size = 18, text_size = 18, marker_size = 0.2, marker_shape ='*', figure_size = (8, 8)):
    
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    results.sort_values(by=[score_column], ascending = False, ignore_index = True, inplace = True)
    non_case = results[target_col].value_counts().get(0)
    case = results[target_col].value_counts().get(1)
    factor = math.floor((non_case/case)/10)*10

    if augment_samples:
        subset = results[results[target_col] == 1]
        copies = [subset] * factor
        results_norm = pd.concat([results] + copies, ignore_index=True)
    else:
        results_norm = results

    x = np.array(results_norm[score_column])
    y = np.array(results_norm[target_col]).astype(int)
    slope, intercept = np.polyfit(x, y, 1)

    line = slope*x+intercept
    # mask = line <= 1
    # line_plot = line[mask]
    
    plt.figure(num = None, figsize = figure_size, facecolor='w', edgecolor='b')
    plt.scatter(x, y, color='purple', s=marker_size, marker=marker_shape)
    plt.plot(x, line, color='steelblue', linestyle='--', linewidth=2)
    plt.yticks(np.arange(0, 1.1, step = 0.1), size = tick_size)
    plt.xticks(np.arange(0, 1.1, step = 0.1), size = tick_size)
    plt.xlabel(x_label, size = label_size)
    plt.ylabel(y_label, size = label_size)
    plt.ylim([-0.05,1.05])
    plt.grid(True)
    plt.text(0.61, 0.13, 'y = ' + '{:.2f}'.format(intercept) + ' + {:.2f}'.format(slope) + 'x', size = text_size)
    plt.show()

    return intercept, slope


def plot_roc_curve(results, score_col = 'score', target_col = 'case', plot_gmean = False, clip_factor = 1e-8, model_name = 'ML Model', x_label = 'False Positive Rate', y_label = 'True Positive Rate', tick_size = 14, label_size = 18, legend_size = 18, text_size = 18, figure_size = (8, 8)):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    y_true = np.array(results[target_col], dtype='int64')
    y_prob = np.array(results[score_col])

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fpr = np.clip(fpr, clip_factor, 1 - clip_factor)
    tpr = np.clip(tpr, clip_factor, 1 - clip_factor)

    roc_auc = auc(fpr, tpr)

    gmean = np.sqrt(tpr * (1 - fpr))
    g_max_index = np.argmax(gmean)
    optimal_threshold_roc = round(thresholds[g_max_index], ndigits = 4)
    optimal_gmean = round(gmean[g_max_index], ndigits = 4)
    optimal_fpr = round(fpr[g_max_index], ndigits = 4)
    optimal_tpr = round(tpr[g_max_index], ndigits = 4)

    plt.subplots(1, figsize=figure_size)
    plt.plot(fpr, tpr, color = 'steelblue', label = model_name)
    plt.plot([0, 1],  color = 'orange', ls="--", label = 'baseline')
    if plot_gmean:
        plt.plot(optimal_fpr, optimal_tpr, marker='o', markersize = 10, color = 'red', label = f'threshold = {optimal_threshold_roc:.2f}')
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, step = 0.1), size=tick_size)
    plt.xticks(np.arange(0, 1.1, step = 0.1), size=tick_size)
    plt.text(0.7, 0.22, 'AUC = ' + '{:.3f}'.format(roc_auc), size=text_size)
    plt.ylabel(x_label, size=label_size)
    plt.xlabel(y_label, size=label_size)
    plt.ylim([-0.05,1.05])
    plt.legend(prop={'size': legend_size})
    plt.show()

    return optimal_fpr, optimal_tpr, optimal_gmean, optimal_threshold_roc 


def plot_pr_curve(results, score_col = 'score', target_col = 'case', beta = 2, plot_fbeta = False, clip_factor = 1e-8, model_name = 'ML Model', x_label = 'Recall', y_label = 'Precision', tick_size = 14, label_size = 18, legend_size = 18, text_size = 18, figure_size = (8, 8)):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc
    
    y_true = np.array(results[target_col], dtype='int64')
    y_prob = np.array(results[score_col])

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = np.clip(precision, clip_factor, 1 - clip_factor)
    recall = np.clip(recall, clip_factor, 1 - clip_factor)

    pr_auc = auc(recall, precision)

    fb_score = ((1 + beta**2) * (precision * recall))/((beta**2 * precision) + recall)
    counts = np.array(np.unique(y_true, return_counts=True)).T
    pos = counts[1][1]
    neg = counts[0][1]
    baseline = pos/(pos+neg)

    fb_max_index = np.argmax(fb_score)
    optimal_threshold_pr = round(thresholds[fb_max_index], ndigits = 2)
    optimal_fbeta = round(fb_score[fb_max_index], ndigits = 4)
    optimal_recall = round(recall[fb_max_index], ndigits = 4)
    optimal_precision = round(precision[fb_max_index], ndigits = 2)

    plt.subplots(1, figsize=figure_size)
    plt.plot(recall, precision, color = 'steelblue', label = model_name)
    plt.plot([baseline, baseline], ls="--", color = 'orange', label = 'baseline')
    if plot_fbeta:
        plt.plot(optimal_recall, optimal_precision, marker='o', markersize = 10, color = 'red', label = f'threshold={optimal_threshold_pr:.2f}')
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, step = 0.1), size=tick_size)
    plt.xticks(np.arange(0, 1.1, step = 0.1), size=tick_size)
    plt.text(0.7, 0.22, 'AUC = ' + '{:.3f}'.format(pr_auc), size=text_size)
    plt.text(0, baseline + 0.01, 'y = ' + '{:.3f}'.format(baseline), size=18)
    plt.ylabel(y_label, size=label_size)
    plt.xlabel(x_label, size=label_size)
    plt.ylim([-0.05,1.05])
    plt.legend(prop={'size': legend_size})
    plt.show()

    return optimal_recall, optimal_precision, optimal_fbeta, optimal_threshold_pr 


def evaluate_operational(results, k, score_col = 'score', target_col = 'case', prob_threshold = -1, ouput_random = False, sampling_number = 10):

    #TODO: write code when output_random == False
    import numpy as np
    import pandas as pd
    import random
    
    days = np.sort(results['day'].unique())
    months = np.sort(results['month'].unique())
    years = np.sort(results['year'].unique())
    
    operational_df = pd.DataFrame()
    cumulative_df = pd.DataFrame()
    
    for year in years:
        for month in months:
            for day in days:
                prediction_df = results.loc[(results['year'] == year) & 
                                            (results['month'] == month) & 
                                            (results['day'] == day)]

                sorted_df = prediction_df.sort_values(by=[score_col], ascending=False)

                date = f'{day:02}/{month:02}/{year}'

                infected = len(prediction_df.loc[prediction_df[target_col] == 1])

                if (prob_threshold <= 0):
                    discovered = len(sorted_df.head(k).loc[(sorted_df[target_col] == 1)])
                else:
                    discovered = len(sorted_df.head(k).loc[(sorted_df[target_col] == 1) & (sorted_df[score_col] >= prob_threshold)])
                
                if ouput_random:
                    if infected != 0:
                        idx_list = prediction_df.index.to_list()
                        random_discovery_list = []

                        for _ in range(sampling_number):
                            random_sample = random.sample(idx_list, k)
                            random_df = prediction_df[prediction_df.index.isin(random_sample)]
                            random_hits = len(random_df.loc[(random_df[target_col] == 1)])
                            random_discovery_list.append(random_hits)
                        
                        random_discovered = sum(random_discovery_list)/len(random_discovery_list)
                    else:
                        random_discovered = 0

                if infected == 0:
                    outof = 0
                    percentage = np.nan
                    random_percentage = np.nan
                elif infected > k:
                    outof = k
                    percentage = discovered/k
                    random_percentage = random_discovered/k
                else:
                    outof = infected
                    percentage = discovered/infected
                    random_percentage = random_discovered/infected
                    
                operational_df = operational_df.append({'Prediction Date': date,
                                                        'Infected': infected,
                                                        'Discovered': discovered,
                                                        'Random Discovered': random_discovered,
                                                        'Out of': outof,
                                                        'Percentage': 'NaN' if (pd.isna(percentage)) else f'{percentage*100:.2f} %',
                                                        'percentage' : f'{percentage:.4f}',
                                                        'Random Percentage' : 'NaN' if (pd.isna(random_percentage)) else f'{random_percentage*100:.2f} %',
                                                        'random percentage' : f'{random_percentage:.4f}',}, ignore_index = True)
                
                operational_df = operational_df.astype({'Infected':'int', 'Discovered':'int', 'Random Discovered':'float', 'Out of':'int', 'percentage':'float', 'random percentage':'float'})
        
        annual = operational_df.loc[operational_df['Prediction Date'].str.endswith(str(year))]
        all_infected = annual['Infected'].sum()
        all_discovered = annual['Discovered'].sum()
        all_random_discovered = annual['Random Discovered'].sum()
        all_outof = annual['Out of'].sum()
        #all_percentage = annual['percentage'].mean()

        if all_discovered == 0 and all_outof == 0:
            all_percentage = np.nan
        else:
            all_percentage = all_discovered/all_outof

        if all_random_discovered == 0 and all_outof == 0:
            all_random_percentage = np.nan
        else:
            all_random_percentage = all_random_discovered/all_outof
        
        cumulative_df = cumulative_df.append({'Year': year,
                                              'Infected': all_infected,
                                              'Discovered': all_discovered,
                                              'Random Discovered' : all_random_discovered,
                                              'Out Of': all_outof,
                                              'Percentage': f'{all_percentage*100:.2f} %',
                                              'percentage': f'{all_percentage:.4f}',
                                              'Random Percentage': f'{all_random_percentage*100:.2f} %',
                                              'random percentage': f'{all_random_percentage:.4f}'}, ignore_index = True)
        
    
    cumulative_df = cumulative_df.astype({'Year': 'int', 'Infected': 'int', 'Discovered': 'int', 'Random Discovered':'float', 'Out Of':'int', 'percentage': 'float', 'random percentage': 'float'})
    
    avg = cumulative_df['percentage'].mean()
    w_avg = np.average(cumulative_df['percentage'].fillna(0), weights = list(cumulative_df['Infected']), axis = 0)
    r_avg = cumulative_df['random percentage'].mean()
    r_w_avg = np.average(cumulative_df['random percentage'].fillna(0), weights = list(cumulative_df['Infected']), axis = 0)
    
    #return operational_df.iloc[: , :-1], cumulative_df.iloc[: , :-1], avg, w_avg
    return operational_df, cumulative_df, avg, w_avg, r_avg, r_w_avg
     

def operational_year(data, year):
    out = data.loc[data['Prediction Date'].str.endswith(str(year))]
    print(out)

def classification_report_new(results_train, results_test, score_col = 'score', target_col = 'case', threshold = 0.5, beta = 1, round_factor = 4):
    
    import numpy as np
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, fbeta_score, log_loss, confusion_matrix 
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    years = np.sort(results_test['year'].unique())
    
    report_train = pd.DataFrame()
    report_test = pd.DataFrame()
    
    fbeta_label = f'F{beta} Score'

    #metrics on training set
    probabilities_train = results_train[score_col]
    y_train = results_train[target_col].values.astype(int)
    predictions_train = np.where(probabilities_train > threshold, 1, 0)
        
    bal_acc_train = balanced_accuracy_score(y_train, predictions_train)
    precision_train = precision_score(y_train, predictions_train, labels = [0,1], zero_division = 0)
    recall_train = recall_score(y_train, predictions_train, labels = [0,1], zero_division = 0)
    fb_score_train = fbeta_score(y_train, predictions_train, beta = beta, labels = [0,1], zero_division = 0)
    try:
        log_loss_train = log_loss(y_train, probabilities_train, labels = [0,1])
    except ValueError:
        log_loss_train = 0
        
    report_train = report_train.append({'Accuracy': round(bal_acc_train, round_factor),
                                        'Precision': round(precision_train, round_factor),
                                        'Recall': round(recall_train, round_factor), 
                                        fbeta_label : round(fb_score_train, round_factor),
                                        'Loss' : round(log_loss_train, round_factor)}, ignore_index = True)
    
    for year in years:
        #metrics on test set
        probabilities_test = results_test.loc[results_test['year'] == year][score_col]
        y_test = results_test.loc[results_test['year'] == year][target_col].values.astype(int)
        _, counts = np.unique(y_test, return_counts=True)
        if (len(counts) == 2):
            infected = counts[1]
        else:
            infected = 0
            
        predictions_test = np.where(probabilities_test > threshold, 1, 0)
        
        cm = confusion_matrix(y_test, predictions_test, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        positive_rate = (fp+tp) / (tp + fn + fp + tn)
    
        bal_acc_test = balanced_accuracy_score(y_test, predictions_test)
        precision_test = precision_score(y_test, predictions_test, labels = [0,1], zero_division = 0)
        recall_test = recall_score(y_test, predictions_test, labels = [0,1], zero_division = 0)
        fb_score_test = fbeta_score(y_test, predictions_test, beta = beta, labels = [0,1], zero_division = 0)
        log_loss_test = log_loss(y_test, probabilities_test, labels = [0,1])
        
        report_test = report_test.append({'Year': f'{year}'.split('.')[0],
                                          'Infected': infected,
                                          'Positive Rate': round(positive_rate, round_factor),
                                          'Accuracy': round(bal_acc_test, round_factor),
                                          'Precision': round(precision_test, round_factor),
                                          'Recall': round(recall_test, round_factor), 
                                          fbeta_label : round(fb_score_test, round_factor),
                                          'Loss' : round(log_loss_test, round_factor)}, ignore_index = True)
        
    # report_train = report_train.append({'Year': 'Mean',
    #                                     'Accuracy': round(report_train["Accuracy"].mean(), round_factor),
    #                                     'Precision': round(report_train["Precision"].mean(), round_factor),
    #                                     'Recall': round(report_train["Recall"].mean(), round_factor), 
    #                                     fbeta_label : round(report_train[fbeta_label].mean(), round_factor),
    #                                     'Loss' : round(report_train["Loss"].mean(), round_factor)}, ignore_index = True)
    
    # report_train = report_train.append({'Year': 'Median',
    #                                     'Accuracy': round(report_train["Accuracy"].median(), round_factor),
    #                                     'Precision': round(report_train["Precision"].median(), round_factor),
    #                                     'Recall': round(report_train["Recall"].median(), round_factor), 
    #                                     fbeta_label : round(report_train[fbeta_label].median(), round_factor),
    #                                     'Loss' : round(report_train["Loss"].median(), round_factor)}, ignore_index = True)    
    
    report_test = report_test.append({'Year': 'Mean',
                                      'Infected': round(report_test["Infected"].mean(), round_factor),
                                      'Positive Rate': round(report_test["Positive Rate"].mean(), round_factor),
                                      'Accuracy': round(report_test["Accuracy"].mean(), round_factor),
                                      'Precision': round(report_test["Precision"].mean(), round_factor),
                                      'Recall': round(report_test["Recall"].mean(), round_factor), 
                                      fbeta_label : round(report_test[fbeta_label].mean(), round_factor),
                                      'Loss' : round(report_test["Loss"].mean(), round_factor)}, ignore_index = True)

    report_test = report_test.append({'Year': 'Median',
                                      'Infected': round(report_test["Infected"].median(), round_factor),
                                      'Positive Rate': round(report_test["Positive Rate"].median(), round_factor),
                                      'Accuracy': round(report_test["Accuracy"].median(), round_factor),
                                      'Precision': round(report_test["Precision"].median(), round_factor),
                                      'Recall': round(report_test["Recall"].median(), round_factor), 
                                      fbeta_label : round(report_test[fbeta_label].median(), round_factor),
                                      'Loss' : round(report_test["Loss"].median(), round_factor)}, ignore_index = True)
        
    return report_train, report_test

def classification_report(results_train, results_test, score_col = 'score', target_col = 'case', threshold = 0.5, beta = 1, round_factor = 4):
    
    import numpy as np
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, fbeta_score, log_loss, confusion_matrix 
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    years = np.sort(results_test['year'].unique())
    
    report_train = pd.DataFrame()
    report_test = pd.DataFrame()
    
    fbeta_label = f'F{beta} Score'
    
    for year in years:
        #metrics on training set
        probabilities_train = results_train.loc[results_train['year'] == year][score_col]
        y_train = results_train.loc[results_train['year'] == year][target_col].values.astype(int)
        predictions_train = np.where(probabilities_train > threshold, 1, 0)
        
        bal_acc_train = balanced_accuracy_score(y_train, predictions_train)
        precision_train = precision_score(y_train, predictions_train, labels = [0,1], zero_division = 0)
        recall_train = recall_score(y_train, predictions_train, labels = [0,1], zero_division = 0)
        fb_score_train = fbeta_score(y_train, predictions_train, beta = beta, labels = [0,1], zero_division = 0)
        try:
            log_loss_train = log_loss(y_train, probabilities_train, labels = [0,1])
        except ValueError:
            log_loss_train = 0
        
        report_train = report_train.append({'Year': f'{year}'.split('.')[0],
                                            'Accuracy': round(bal_acc_train, round_factor),
                                            'Precision': round(precision_train, round_factor),
                                            'Recall': round(recall_train, round_factor), 
                                            fbeta_label : round(fb_score_train, round_factor),
                                            'Loss' : round(log_loss_train, round_factor)}, ignore_index = True)
        
        #metrics on test set
        probabilities_test = results_test.loc[results_test['year'] == year][score_col]
        y_test = results_test.loc[results_test['year'] == year][target_col].values.astype(int)
        _, counts = np.unique(y_test, return_counts=True)
        if (len(counts) == 2):
            infected = counts[1]
        else:
            infected = 0
            
        predictions_test = np.where(probabilities_test > threshold, 1, 0)
        
        cm = confusion_matrix(y_test, predictions_test, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        positive_rate = (fp+tp) / (tp + fn + fp + tn)
    
        bal_acc_test = balanced_accuracy_score(y_test, predictions_test)
        precision_test = precision_score(y_test, predictions_test, labels = [0,1], zero_division = 0)
        recall_test = recall_score(y_test, predictions_test, labels = [0,1], zero_division = 0)
        fb_score_test = fbeta_score(y_test, predictions_test, beta = beta, labels = [0,1], zero_division = 0)
        log_loss_test = log_loss(y_test, probabilities_test, labels = [0,1])
        
        report_test = report_test.append({'Year': f'{year}'.split('.')[0],
                                          'Infected': infected,
                                          'Positive Rate': round(positive_rate, round_factor),
                                          'Accuracy': round(bal_acc_test, round_factor),
                                          'Precision': round(precision_test, round_factor),
                                          'Recall': round(recall_test, round_factor), 
                                          fbeta_label : round(fb_score_test, round_factor),
                                          'Loss' : round(log_loss_test, round_factor)}, ignore_index = True)
        
    report_train = report_train.append({'Year': 'Mean',
                                        'Accuracy': round(report_train["Accuracy"].mean(), round_factor),
                                        'Precision': round(report_train["Precision"].mean(), round_factor),
                                        'Recall': round(report_train["Recall"].mean(), round_factor), 
                                        fbeta_label : round(report_train[fbeta_label].mean(), round_factor),
                                        'Loss' : round(report_train["Loss"].mean(), round_factor)}, ignore_index = True)
    
    report_train = report_train.append({'Year': 'Median',
                                        'Accuracy': round(report_train["Accuracy"].median(), round_factor),
                                        'Precision': round(report_train["Precision"].median(), round_factor),
                                        'Recall': round(report_train["Recall"].median(), round_factor), 
                                        fbeta_label : round(report_train[fbeta_label].median(), round_factor),
                                        'Loss' : round(report_train["Loss"].median(), round_factor)}, ignore_index = True)    
    
    report_test = report_test.append({'Year': 'Mean',
                                      'Infected': round(report_test["Infected"].mean(), round_factor),
                                      'Positive Rate': round(report_test["Positive Rate"].mean(), round_factor),
                                      'Accuracy': round(report_test["Accuracy"].mean(), round_factor),
                                      'Precision': round(report_test["Precision"].mean(), round_factor),
                                      'Recall': round(report_test["Recall"].mean(), round_factor), 
                                      fbeta_label : round(report_test[fbeta_label].mean(), round_factor),
                                      'Loss' : round(report_test["Loss"].mean(), round_factor)}, ignore_index = True)

    report_test = report_test.append({'Year': 'Median',
                                      'Infected': round(report_test["Infected"].median(), round_factor),
                                      'Positive Rate': round(report_test["Positive Rate"].median(), round_factor),
                                      'Accuracy': round(report_test["Accuracy"].median(), round_factor),
                                      'Precision': round(report_test["Precision"].median(), round_factor),
                                      'Recall': round(report_test["Recall"].median(), round_factor), 
                                      fbeta_label : round(report_test[fbeta_label].median(), round_factor),
                                      'Loss' : round(report_test["Loss"].median(), round_factor)}, ignore_index = True)
        
    return report_train, report_test

def create_grid(river_shapes, data_shapes, col_municipal='NAME', col_geometry='geometry', dimension=2):

    import shapely
    import math
    import numpy as np
    import geopandas as gpd
    from fiona.crs import from_epsg
    
    """Creates the cell grid for the desired area
    
    Parameters
    --------
    river_shapes : dataframe,
        A GeoDataFrame containing the rivers of the area (to avoid creating cells on them)
                
    data_shapes : dataframe
        A GeoDataFrame containing the polygons (shapes) and the names of municipalities
        
    col_municipal : str, optional
        The column that contains the name for each municipality (default = 'NAME')
        
    col_geometry : string, optional
        The name of the column with the geometry of the river (default = 'geometry')
        
    dimension : int, optional
        The dimension of the square cell in km (eg. 2km x 2km, 1km x 1km) (default = 2)
        
    Returns
    ----------
    cell_df: dataframe
        A dataframe containing the generated cells for each municipality and the centers of each cell
    
    """
    river_shapes_list =  river_shapes[col_geometry].geometry.explode().tolist()
    # Convert cell height dimension to km
    cell_height = (dimension / 6378) * (180 / math.pi)
    
    grid_cells = []
    grid_cells2 = []
    municipal = []
    centers = []
    # Iterate over municipalities to create a grid for each one
    for i in range(data_shapes.shape[0]):
        area_cells = []
        center = []
        # Keep the geometry of municipality
        g_country = data_shapes[col_geometry][i]
        # Extract the bounds of municipality
        xmin, ymin, xmax, ymax = data_shapes[col_geometry][i:i+1].total_bounds
        # Convert cell width dimension to km (dependent to y)
        cell_width  = (dimension / 6378) * (180 / math.pi) / math.cos(ymin  * math.pi/180)
    
        # Iterate to municipality bounds to create the grid's cells
        for x0 in np.arange(xmin, xmax+cell_width, cell_width ):
            for y0 in np.arange(ymin, ymax+cell_height, cell_height):
                y1 = y0+cell_height
                x1 = x0+cell_width
                new_cell = shapely.geometry.box(x0, y0, x1, y1)
                # Check if new cell intersects the municipality
                if new_cell.covered_by(g_country):
                    # Check if new cell intersects previously generated cells
                    delete_cell_boundary = [True for cell in grid_cells2 if new_cell.intersects(cell)]
                    # Check if new cell intersects with rivers
                    delete_cell_river = [True for cell in river_shapes_list if new_cell.intersects(cell)]
                    # Save new cell if it doesn't intersects rivers or previous cells 
                    if not delete_cell_boundary and not delete_cell_river:
                        center.append(new_cell.centroid)
                        area_cells.append(new_cell)
                else:
                    # Continue to next cell generation
                    pass
        # Save the final cells for each municipality
        grid_cells.append(area_cells)
        # Save center of each cell
        centers.append(center)
        # Save municipality name for each generated cell
        municipal.append([data_shapes[col_municipal][i]]*len(area_cells))
        # Save the previous generated cells for intersection check
        grid_cells2 = []
        grid_cells2 = [item for sublist in grid_cells for item in sublist]

    # Flatten lists
    grid_cells_list = [item for sublist in grid_cells for item in sublist]
    centers_list = [item for sublist in centers for item in sublist]
    municipal_list = [item for sublist in municipal for item in sublist]
    
    # Create final GeoDataFrame with municipalities and their cells
    cell_df = gpd.GeoDataFrame(grid_cells_list, columns=[col_geometry], crs=from_epsg(4326))
    cell_df['centers'] = centers_list
    cell_df['NAME'] = municipal_list
    
    x = []
    y = []
    # Extract coordinates from centers
    for index,row in cell_df.iterrows():
        x.append(round(row['centers'].x,5))
        y.append(round(row['centers'].y,5))  
        
    cell_df['x'] = x
    cell_df['y'] = y
    
    return cell_df

def read_river_shapes(col_id, path_rivers, col_geometry='geometry', filter=False, num_region=0, path_region='', *args):

    import geopandas as gpd

    """Reads rivers shapefile and returns a geopandas dataframe
    
    Parameters
    --------
    col_id : string,
        The name of the column with the id of the river
                
    path_rivers : string
        Path to shapefile of rivers
        
    col_geometry : string, optional
        The name of the column with the geometry of the river (default = 'geometry')
        
    filter : boolean, optional
        If false all rivers will be return (default = False)
        
    num_region : int,optional
        The number of the row of the desired region in GeoDataFrame (default = 0)
        
    path_region : string, optional
        Path to shapefile of region that its' rivers will be kept
        
    data_shapes : dataframe, optional (pass with *args)
        A GeoDataFrame containing the polygons (shapes) and the names of municipalities

    Returns
    ----------
    data: dataframe
        A dataframe containing the desired river shapes
    
    """
    try:
        # reading as shp file
        river_shapes = gpd.read_file(path_rivers, encoding="utf_8")
    except: 
        raise NotImplementedError("Sorry, give me a .shp file.")
    if filter:
        indexes = []
        if path_region:
            try:
                # reading as shp file
                region = gpd.read_file(path_region, encoding="utf_8")
            except: 
                raise NotImplementedError("Sorry, give me a .shp file.")
            for index, row in river_shapes.iterrows():
                if row[col_geometry].intersects(region[col_geometry][num_region]):
                    indexes.append(index)
        else:
            data_shapes=args[0]
            for index, row in river_shapes.iterrows():
                for index2, row2 in data_shapes.iterrows():
                    if row[col_geometry].intersects(row2[col_geometry]):
                        indexes.append(index)
                
        river_shapes = river_shapes.loc[indexes].reset_index(drop=True)
    river_shapes = river_shapes[[col_id, col_geometry]]
    return river_shapes

def remove_accents(input_str):

    import unicodedata

    """Removes langueage accents from strings
    
    Parameters
    --------
    input_str : string,
        The string with accents
                
    Returns
    ----------
    string: the string without accents 
    """
    
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def process_greek(frame, column):
    import re

    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    frame[column] = frame[column].apply(lambda x : x.strip())
    frame[column] = frame[column].apply(lambda x : x.lower())
    frame[column] = frame[column].apply(lambda x : x.replace('ά','α'))
    frame[column] = frame[column].apply(lambda x : x.replace('έ','ε'))
    frame[column] = frame[column].apply(lambda x : x.replace('ή','η'))
    frame[column] = frame[column].apply(lambda x : x.replace('ί','ι'))
    frame[column] = frame[column].apply(lambda x : x.replace('ϊ','ι'))
    frame[column] = frame[column].apply(lambda x : x.replace('ΐ','ι'))
    frame[column] = frame[column].apply(lambda x : x.replace('ύ','υ'))
    frame[column] = frame[column].apply(lambda x : x.replace('ϋ','υ'))
    frame[column] = frame[column].apply(lambda x : x.replace('ΰ','υ'))
    frame[column] = frame[column].apply(lambda x : x.replace('ό','ο'))
    frame[column] = frame[column].apply(lambda x : x.replace('ώ','ω'))
    frame[column] = frame[column].apply(lambda x : x.replace('a','α'))
    frame[column] = frame[column].apply(lambda x : x.replace('-',' '))
    frame[column] = frame[column].apply(lambda x : x.replace('–',' '))
    frame[column] = frame[column].apply(lambda x : x.replace('&','και'))
    frame[column] = frame[column].apply(lambda x : x.replace('ν. ',''))

    
    frame[column] = frame[column].apply(lambda x : _RE_COMBINE_WHITESPACE.sub(" ", x).strip())

def process_italic(frame, column):

    import re

    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    frame[column] = frame[column].apply(lambda x : x.lower())
    frame[column] = frame[column].apply(lambda x : _RE_COMBINE_WHITESPACE.sub(" ", x).strip())
    frame[column] = frame[column].apply(lambda x : re.sub(r'\([^)]*\)', '', x))
    frame[column] = frame[column].apply(lambda x : remove_accents(x))
    frame[column] = frame[column].apply(lambda x : x.replace("'", " " ))
    frame[column] = frame[column].apply(lambda x : x.replace("-", " " ))
    frame[column] = frame[column].apply(lambda x : x.replace("’", " " ))
    
    frame[column] = frame[column].apply(lambda x : _RE_COMBINE_WHITESPACE.sub(" ", x).strip())
    frame[column] = frame[column].apply(lambda x : x.replace("s. ", "saint " ))
    frame[column] = frame[column].apply(lambda x : x.replace("san ", "saint " ))
    frame[column] = frame[column].apply(lambda x : x.replace("santa ", "saint " ))
    frame[column] = frame[column].apply(lambda x : x.replace("sant ", "saint " ))
    frame[column] = frame[column].apply(lambda x : x.replace("santo ", "saint " ))

    frame[column] = frame[column].apply(lambda x : x.replace(" d ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" di ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" del ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" dell ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" delle ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" della ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" da ", " da " ))
    frame[column] = frame[column].apply(lambda x : x.replace(" de ", " da " ))


def fillna(dataframe,fill_list):

    for i in list(fill_list.keys()):
        if i not in dataframe.columns:
            raise KeyError('Column(s) not in index')
        else:
            dataframe[i] = dataframe[i].fillna(fill_list[i])
    return dataframe

# def describe_dataframe(dataframe, rounding_factor = 4, remove_zeros = False):

#     import pandas as pd

#     df_desc = pd.DataFrame(columns=['Column', 'Missing'])

#     for col in dataframe.columns:
#         missing_percentage = dataframe[col].isnull().sum() * 100 / len(dataframe[col])
    
#         df_desc = df_desc.append({'Column': col, 
#                                 'Missing': round(missing_percentage, rounding_factor)}, ignore_index = True)

#         if remove_zeros:
#             df_desc = df_desc[df_desc["Missing"] != 0]

#     return df_desc

def describe_dataframe(dataframe, rounding_factor=4, remove_zeros=False):
    import pandas as pd

    missing_data = {'Column': [], 'Missing': []}

    for col in dataframe.columns:
        missing_percentage = dataframe[col].isnull().mean() * 100
        missing_data['Column'].append(col)
        missing_data['Missing'].append(round(missing_percentage, rounding_factor))

    df_desc = pd.DataFrame(missing_data)

    if remove_zeros:
        df_desc = df_desc[df_desc["Missing"] != 0]

    return df_desc

def lau1_mapping(dataframe, x0, y0, lau1_col = 'lau1', map_col = 'lau1_id', dist_col = 'eq_distance'):

    import math

    data = dataframe
    data[dist_col] = data.apply(lambda row : round(math.sqrt(math.pow((row['x']-x0) , 2) + math.pow((row['y']-y0) , 2)),5), axis = 1)
    data_sub = data[[lau1_col, dist_col]]
    data_sorted = data_sub.sort_values(dist_col).drop_duplicates(ignore_index = True)
    data_sorted[map_col] = data_sorted.index
    
    mapping = {}
    for index, row in data_sorted.iterrows():
        mapping[row[lau1_col]] = row[map_col]
        
    return mapping

def encode_cyclical(dataframe, col, max_val):

    import numpy as np

    dataframe[col + '_sin'] = np.sin(2 * np.pi * dataframe[col]/max_val)
    dataframe[col + '_cos'] = np.cos(2 * np.pi * dataframe[col]/max_val)

    return dataframe

def dataset_mamoth_preparation(dataframe, columns, del_months = None, month_col = 'month', mean_lst = True):

    from statistics import mean

    dataset_mamoth = dataframe.copy()
    if mean_lst:
        dataset_mamoth['lst_jan_mean'] = dataset_mamoth.apply(lambda row: mean([row['lst_jan_day_mean'],row['lst_jan_night_mean']]), axis=1)
        dataset_mamoth['lst_feb_mean'] = dataset_mamoth.apply(lambda row: mean([row['lst_feb_day_mean'],row['lst_feb_night_mean']]), axis=1)
        dataset_mamoth['lst_mar_mean'] = dataset_mamoth.apply(lambda row: mean([row['lst_mar_day_mean'],row['lst_mar_night_mean']]), axis=1)
        dataset_mamoth['lst_apr_mean'] = dataset_mamoth.apply(lambda row: mean([row['lst_apr_day_mean'],row['lst_apr_night_mean']]), axis=1)
    dataset_mamoth['mosq_now'] = 0

    if del_months != None:
        dataset_mamoth = dataset_mamoth[~dataset_mamoth[month_col].isin(del_months)]

    dataset_mamoth = dataset_mamoth[columns]

    dataset_mamoth.reset_index(inplace=True, drop=True)

    return dataset_mamoth

def calculate_mosq_previous(dataframe, lau1_col = 'lau1', date_col = 'dt_placement', mosq_prediction_col = 'mosq_pred', result_col = 'mosq_previous'):

    for municipality in dataframe[lau1_col].drop_duplicates():

        data = dataframe.loc[dataframe[lau1_col] == municipality].copy()
        data.sort_values(by=[date_col], inplace = True)
        data.reset_index(inplace = True, drop = True)
        
        for index, row in data.iterrows():
        
            try:
                mosq_minus_2 = data.loc[index-2, mosq_prediction_col]
            except KeyError:
                mosq_minus_2 = 0
        
            try:
                mosq_minus_1 = data.loc[index-1, mosq_prediction_col]
            except KeyError:
                mosq_minus_1 = 0
            
            mosq_sum = mosq_minus_2 + mosq_minus_1
        
            index_at_dataframe = dataframe.index[(dataframe[lau1_col] == row[lau1_col]) & (dataframe[date_col] == row[date_col])].tolist()
            if (len(index_at_dataframe) != 1):
                raise(IndexError("Duplicate Index Found"))
            
            dataframe.at[index_at_dataframe[0], result_col] = mosq_sum

    return dataframe


def convert_temperature(dataframe, result_col = 'lst'):
    temprature_cols = dataframe.filter(regex='^lst_*').columns.tolist()

    dataframe[temprature_cols] = dataframe[temprature_cols].apply(lambda x : x * 0.02 - 273.15)
    dataframe[result_col] = (dataframe['lst_day'] + dataframe['lst_night'])/2

    return dataframe

def convert_multiple_cases(dataframe, target_col = 'case'):
    import pandas as pd

    case_counts = dataframe[target_col].value_counts().index

    for case_count in case_counts:
        if (case_count >= 2):
            # dataframe = dataframe.append([dataframe[dataframe[target_col] == case_count]] * (case_count - 1), ignore_index=True)
            subset = dataframe[dataframe[target_col] == case_count]
            copies = [subset] * (case_count - 1)
            dataframe = pd.concat([dataframe] + copies, ignore_index=True)
            
    dataframe[target_col] = dataframe[target_col].apply(lambda x : 1 if(x > 0) else 0)

    return dataframe

def convert_multiple_cases_array_weighted(dataframe, target_col = 'case'):
    import numpy as np
    import pandas as pd

    unique_arrays = dataframe[target_col].value_counts().index

    for arr in unique_arrays:
        gt2 = np.unique(arr[arr >= 2])
        if (gt2.size != 0):
            temp_pd_series = dataframe[target_col]
            matching_rows = temp_pd_series[temp_pd_series.apply(lambda x: np.array_equal(x, arr))]
            new_rows = dataframe.iloc[matching_rows.index]
            multiplier = (np.min(gt2) - 1)
            dataframe = pd.concat([dataframe] + [new_rows] * multiplier, ignore_index=True)
            
    dataframe[target_col] = dataframe[target_col].apply(lambda x: np.where(x == 0, 0, 1))

    return dataframe

def convert_multiple_cases_array(dataframe, target_col = 'case'):
    import numpy as np

    dataframe[target_col] = dataframe[target_col].apply(lambda x: np.where(x == 0, 0, 1))

    return dataframe

def calculate_nearest_topological(data, topological, neighbors=1):
    import numpy as np
    from statistics import mean
    from sklearn.neighbors import BallTree

    topological['x_rad'] = topological['x'].apply(lambda x: np.deg2rad(x))
    topological['y_rad'] = topological['y'].apply(lambda x: np.deg2rad(x))
    
    data['x_rad'] = data['x'].apply(lambda x: np.deg2rad(x))
    data['y_rad'] = data['y'].apply(lambda x: np.deg2rad(x))
    
    ball = BallTree(topological[["y_rad", "x_rad"]].values, metric='haversine')
    distances, indices = ball.query(data[["y_rad", "x_rad"]].values, k = neighbors)
    
    if neighbors>1:
        distances = [mean(d) for d in distances]
        distances = [(d * 6371) for d in distances]
    else:
        distances = [(d * 6371).tolist()[0] for d in distances]
        
    indices = indices.tolist()
    indices = [i[0] for i in indices]
    del data['x_rad']
    del data['y_rad']
    del topological['x_rad']
    del topological['y_rad']
    
    return distances, indices

def calculate_nearest_point_temporal(data, topological, date_col = 'dt_placement', neighbors=1):

    import numpy as np
    from scipy.spatial.distance import cdist

    topological['x_rad'] = np.deg2rad(topological['x'])
    topological['y_rad'] = np.deg2rad(topological['y'])

    data['x_rad'] = np.deg2rad(data['x'])
    data['y_rad'] = np.deg2rad(data['y'])

    topological['timestamp'] = topological[date_col].view('int64') // 10**9
    data['timestamp'] = data[date_col].view('int64') // 10**9

    coordinates_topological = topological[['y_rad', 'x_rad']].values
    coordinates_data = data[['y_rad', 'x_rad']].values

    distances = cdist(coordinates_data, coordinates_topological, haversine_distance)

    if neighbors > 1:
        distances = np.mean(distances, axis=1)
    else:
        distances = distances[:, 0]

    indices = np.argsort(distances)
    distances = distances[indices]
    indices = indices[:neighbors]

    del data['x_rad']
    del data['y_rad']
    del data['timestamp']
    del topological['x_rad']
    del topological['y_rad']
    del topological['timestamp']

    return distances.tolist(), indices.tolist()

def haversine_distance(coords1, coords2):

    from math import radians, sin, cos, sqrt, atan2

    lat1, lon1 = coords1
    lat2, lon2 = coords2

    R = 6371  # Earth's radius in kilometers

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def fuzzy_merge(df1, df2, on, date_column, cutoff, scorer, limit=1):

    import pandas as pd
    from fuzzywuzzy import process

    df_merged = pd.merge(df1, df2, on=on, how='outer')
    df_merged['Score'] = df_merged.apply(lambda row: scorer(row[on[0]], row[on[1]]), axis=1)
    df_merged = df_merged[df_merged['Score'] >= cutoff]
    df_merged[date_column] = df_merged.apply(lambda row: process.extractOne(row[date_column + '_x'], df2[date_column], scorer=scorer)[0], axis=1)
    df_merged = df_merged.groupby(df_merged.columns.tolist()).apply(lambda x: x.head(limit))
    df_merged.reset_index(drop=True, inplace=True)
    df_merged.drop('Score', axis=1, inplace=True)
    return df_merged

def compute_distance(row, x1 = 'x1', y1 = 'y1', x2 = 'x2', y2 = 'y2'):

    from geopy.distance import geodesic
    
    start_point = (row[x1], row[y1])
    end_point = (row[x2], row[y2])
    distance = geodesic(start_point, end_point).meters
    return distance

def log_loss_vector(y_true, y_pred):
    import numpy as np

    e = 1e-15
    y_pred = np.clip(y_pred, e, 1 - e)
    log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return log_loss


def optimal_threashold_fbeta_vector(y_prob, y_true, beta = 1, clip_factor = 1e-8, round_factor = 2):
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    if y_prob.shape != y_true.shape:
        raise Exception("Probability and Predictions matricies are not the same shape")

    threshold_per_month = []
    column_range = y_prob.shape[1]

    for col in range(column_range):
        precision, recall, thresholds = precision_recall_curve(y_true[:,col], y_prob[:,col])
        precision = np.clip(precision, clip_factor, 1 - clip_factor)
        recall = np.clip(recall, clip_factor, 1 - clip_factor)
        fb_score = ((1 + beta**2) * (precision * recall))/((beta**2 * precision) + recall)
        #fb_score = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)
        fb_max_index = np.argmax(fb_score)
        optimal_threshold_fb = round(thresholds[fb_max_index], ndigits = round_factor)
        threshold_per_month.append(optimal_threshold_fb)

    return np.array(threshold_per_month)

def optimal_threashold_fbeta_scalar(y_prob, y_true, beta = 1, clip_factor = 1e-8, round_factor = 2):
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    if y_prob.shape != y_true.shape:
        raise Exception("Probability and Predictions matricies are not the same shape")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = np.clip(precision, clip_factor, 1 - clip_factor)
    recall = np.clip(recall, clip_factor, 1 - clip_factor)
    fb_score = ((1 + beta**2) * (precision * recall))/((beta**2 * precision) + recall)
    #fb_score = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)
    fb_max_index = np.argmax(fb_score)
    optimal_threshold_fb = round(thresholds[fb_max_index], ndigits = round_factor)

    return optimal_threshold_fb


def optimal_threashold_fbeta_vector_custom(y_prob, y_true, beta = 1, clip_factor = 1e-8, round_factor = 2):
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, fbeta_score
    
    if y_prob.shape != y_true.shape:
        raise Exception("Probability and Predictions matricies are not the same shape")

    threshold_per_month = []
    column_range = y_prob.shape[1]
    thresholds_pool = np.arange(0,1.01,0.01)

    for col in range(column_range):
        y_true_col = y_true[:,col]
        y_prob_col = y_prob[:,col]
        precision_col = []
        recall_col = []
        fbeta_col = []
        for threashold in thresholds_pool:
            y_pred_col_th = np.where(y_prob_col < threashold, 0, 1)
            precision_col_th = precision_score(y_true_col, y_pred_col_th)
            recall_col_th = recall_score(y_true_col, y_pred_col_th)
            fbeta_col_th = fbeta_score(y_true_col, y_pred_col_th, beta = beta)

            precision_col.append(precision_col_th)
            recall_col.append(recall_col_th)
            fbeta_col.append(fbeta_col_th)

        fb_max_index = np.argmax(fbeta_col)
        optimal_threshold_fb = round(thresholds_pool[fb_max_index], ndigits = round_factor)
        threshold_per_month.append(optimal_threshold_fb)

    return np.array(threshold_per_month)