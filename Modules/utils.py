def read_data(file_path, drop_columns = [], exclude_dtypes = [], exclude_years = [], year_column = 'year'):
    
    import pandas as pd
    
    dataset = pd.read_csv(file_path)
    
    if len(exclude_dtypes) != 0:
        dataset = dataset.select_dtypes(exclude=exclude_dtypes)
        
    if len(drop_columns) != 0:
        dataset.drop(columns = drop_columns, inplace = True)
    
    if len(exclude_years) > 0:
        dataset = dataset[~dataset[year_column].isin(exclude_years)]
        
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
    
    value_counts = dataframe[target_column].value_counts().values
    
    fig, ax = plt.subplots(figsize = figure_size)
    
    x_max = math.ceil(max(value_counts) + (max(value_counts) * .1))
    step = round(math.ceil(x_max * .1) / 1000) * 1000
    if step == 0:
        step = 500
    
    plt.xlabel(x_label, size = label_size)
    
    plt.xticks(np.arange(0, x_max, step = step), size = tick_size, rotation = 45)
    plt.xlim([0, x_max])
    
    big_index = np.argmax(value_counts)
    imbalance = math.ceil((value_counts/min(value_counts))[big_index])
    
    plt.text(value_counts[big_index] - (max(value_counts) * .15), .9, f'imbalance ~ {imbalance}:1', size=text_size)
    
    bars = ax.barh(y_label, value_counts)
    ax.bar_label(bars, label_type= 'edge', size = text_size)
    plt.show()


def dataframe_describe(dataframe, year_column = 'year', target_column = 'case'):
    
    import pandas as pd
    
    result = pd.DataFrame(columns = ['year', 'non-case', 'case', 'total', 'percentage'])
    
    for year in dataframe[year_column].drop_duplicates().sort_values():
        data_train = dataframe.loc[dataframe['year'] != year]
        data_test = dataframe.loc[dataframe['year'] == year]
        try:
            (counts0, counts1) = data_test['case'].value_counts()
        except ValueError:
            counts0 = data_test['case'].value_counts()[0]
            counts1 = 0
           
        percentage = len(data_test)/len(dataframe)*100
        
        result = result.append({'year': year,
                                'non-case': counts0,
                                'case': counts1,
                                'total': (counts0 + counts1),
                                'percentage': f'{percentage:.2f}%'}, 
                                ignore_index = True)
        
    return result


def yearCV_split(dataframe, year_column = 'year', target_column = 'case'):
    
    years_sorted = dataframe[year_column].drop_duplicates().sort_values()
    
    for year in years_sorted:
        data_train = dataframe.loc[dataframe[year_column] != year]
        data_test = dataframe.loc[dataframe[year_column] == year]
        data_test = data_test.drop_duplicates(keep='first')
        
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
    X_train_df = X_train_df.astype({'day':'int', 'month':'int', 'year':'int'})

    indicies_to_remove = []
    for item in exclude_features:
        indicies_to_remove.append(columns_Xtrain.get_loc(item))

    for index in indicies_to_remove:
        X_train = np.delete(X_train, index, axis=1)
        X_test = np.delete(X_test, index, axis=1)
        
    return X_train, X_test, y_train, y_test, X_train_df
    

def calculate_weights(training_set):
    import math
    
    (non_cases, cases) = training_set.value_counts()
    w0 = 1
    try:
        w1 = math.ceil(non_cases/cases)
    except ZeroDivisionError:
        w1 = 10
        
    return w0,w1


def train_and_predict(model, data_train, data_test, X_train, y_train, X_test, y_test):

    import pandas as pd

    model.fit(X_train, y_train)
    
    train_probas = model.predict_proba(X_train)
    test_probas = model.predict_proba(X_test)
    model_coef = model.coef_
    
    train_df = pd.DataFrame()
    train_df['x'] = data_train['x'].reset_index(drop = True)
    train_df['y'] = data_train['y'].reset_index(drop = True)
    #train_df['municipality'] = data_train['lau1'].reset_index(drop = True)
    train_df['day'] = data_train['day'].reset_index(drop = True)
    train_df['month'] = data_train['month'].reset_index(drop = True)
    train_df['year'] = data_train['year'].reset_index(drop = True)
    train_df['case'] = y_train.reset_index(drop = True).astype('int')
    train_df['probability'] = train_probas[:, 1].tolist()
    train_df.sort_values(by=['probability'], ascending = False, ignore_index = True, inplace = True)
    
    test_df = pd.DataFrame()
    test_df['x'] = data_test['x'].reset_index(drop = True)
    test_df['y'] = data_test['y'].reset_index(drop = True)
    test_df['municipality'] = data_test['lau1'].reset_index(drop = True)
    test_df['day'] = data_test['day'].reset_index(drop = True)
    test_df['month'] = data_test['month'].reset_index(drop = True)
    test_df['year'] = data_test['year'].reset_index(drop = True)
    test_df['case'] = y_test.reset_index(drop = True).astype('int')
    test_df['probability'] = test_probas[:, 1].tolist()
    test_df.sort_values(by=['probability'], ascending = False, ignore_index = True, inplace = True)
    
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


def plot_trend_curve(results, cases, plot_min = False, plot_max = False, plot_avg = True, x_label = 'Year', y_label = 'Average Probability', tick_size = 14, label_size = 18, legend_size = 18, text_size = 12, figure_size = (8, 6)):
    import numpy as np
    import matplotlib.pyplot as plt
    
    years = []
    p_min = []
    p_avg = []
    p_max = []
    cases_norm = [float(i)/sum(cases) for i in cases]

    for year in results['year'].drop_duplicates().sort_values():
        years.append(year)
        p_min.append(results[results['year'] == year]['probability'].min())
        p_avg.append(results[results['year'] == year]['probability'].mean())
        p_max.append(results[results['year'] == year]['probability'].max())  

    plt.figure(num = None, figsize = figure_size, dpi = 100, facecolor='w', edgecolor='b')
    
    if plot_min:
        plt.plot(years, p_min, 'green', label='P min')
        
    if plot_max:
        plt.plot(years, p_max, 'red', label='P max')
        
    if plot_avg:
        plt.plot(years, p_avg, 'blue', label='P Avg')
        
    plt.plot(years, cases_norm, 'black', label='Cases')
    
    plt.xticks(np.arange(min(years), max(years)+1), size = tick_size, rotation=45)
    plt.yticks(np.arange(0, 1.1, step = 0.1), size = tick_size)
    plt.xlabel(x_label, size = label_size)
    plt.ylabel(y_label, size = label_size)
    plt.grid()
    plt.legend(prop={'size': legend_size})
    plt.show()

def plot_probability_curve(results, x_label = 'Probability', y_label = 'Ground Truth', tick_size = 14, label_size = 18, legend_size = 18, text_size = 18, figure_size = (8, 8)):
    
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    
    results.sort_values(by=['probability'], ascending = False, ignore_index = True, inplace = True)
    (non_case, case) = results['case'].value_counts()
    factor = math.floor((non_case/case)/10)*10
    
    results_norm = results.append([results[results['case'] == 1]] * factor, ignore_index=True)

    x = np.array(results_norm['probability'])
    y = np.array(results_norm['case']).astype(int)
    a, b = np.polyfit(x, y, 1)
    
    plt.figure(num = None, figsize = figure_size, facecolor='w', edgecolor='b')
    plt.scatter(x, y, color='purple', s=0.5)
    plt.plot(x, a*x+b, color='steelblue', linestyle='--', linewidth=2)
    plt.yticks(np.arange(0, 1.1, step = 0.1), size = tick_size)
    plt.xticks(np.arange(0, 1.1, step = 0.1), size = tick_size)
    plt.xlabel(x_label, size = label_size)
    plt.ylabel(y_label, size = label_size)
    plt.ylim([-0.05,1.05])
    plt.grid(True)
    plt.text(0.61, 0.13, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size = text_size)
    plt.show()


def plot_roc_curve(results, plot_gmean = False, model_name = 'Logistic Regression', x_label = 'False Positive Rate', y_label = 'True Positive Rate', tick_size = 14, label_size = 18, legend_size = 18, text_size = 18, figure_size = (8, 8)):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    y_true = np.array(results['case'], dtype='int64')
    y_prob = np.array(results['probability'])

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    df_fpr_tpr = pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Threshold':thresholds})

    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    optimal_threshold_roc = round(thresholds[index], ndigits = 4)
    optimal_gmean = round(gmean[index], ndigits = 4)

    optimal_fpr = round(fpr[index], ndigits = 4)
    optimal_tpr = round(tpr[index], ndigits = 4)

    plt.subplots(1, figsize=figure_size)
    plt.plot(fpr, tpr, color = 'steelblue', label = model_name)
    plt.plot([0, 1],  color = 'orange', ls="--", label = 'baseline')
    if plot_gmean:
        plt.plot(optimal_fpr, optimal_tpr, marker='o', markersize = 10, color = 'red', label = f'threshold = {optimal_threshold_roc}')
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, step = 0.1), size=tick_size)
    plt.xticks(np.arange(0, 1.1, step = 0.1), size=tick_size)
    plt.text(0.7, 0.22, 'AUC = ' + '{:.3f}'.format(roc_auc), size=text_size)
    plt.ylabel(x_label, size=label_size)
    plt.xlabel(y_label, size=label_size)
    plt.ylim([-0.05,1.05])
    plt.legend(prop={'size': legend_size})
    plt.show()


def plot_pr_curve(results, beta = 2, plot_fbeta = False, model_name = 'Logistic Regression', x_label = 'Recall', y_label = 'Precision', tick_size = 14, label_size = 18, legend_size = 18, text_size = 18, figure_size = (8, 8)):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc
    
    y_true = np.array(results['case'], dtype='int64')
    y_prob = np.array(results['probability'])

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision_zeros = np.where(precision == 0)
    recall_zeros = np.where(recall == 0)
    intersect = np.intersect1d(recall_zeros, precision_zeros)
    
    if (len(intersect) != 0):
        min_precision = np_nth_smallest(precision, len(precision_zeros[0])+1)
        min_recall = np_nth_smallest(recall, len(recall_zeros[0])+1)
        
        for item in intersect:
            precision[item] += min_precision
            recall[item] += min_recall
        
    pr_auc = auc(recall, precision)

    df_recall_precision = pd.DataFrame({'Precision':precision[:-1],
                                    'Recall':recall[:-1],
                                    'Threshold':thresholds})

    fb_score = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)
    counts = np.array(np.unique(y_true, return_counts=True)).T
    pos = counts[1][1]
    neg = counts[0][1]
    baseline = pos/(pos+neg)
    index = np.argmax(fb_score)
    optimal_threshold_pr = round(thresholds[index], ndigits = 2)
    optimal_fbeta = round(fb_score[index], ndigits = 4)
    optimal_recall = round(recall[index], ndigits = 4)
    optimal_precision = round(precision[index], ndigits = 2)

    plt.subplots(1, figsize=figure_size)
    plt.plot(recall, precision, color = 'purple', label = model_name)
    plt.plot([baseline, baseline], ls="--", color = 'orange', label = 'baseline')
    if plot_fbeta:
        plt.plot(optimal_recall, optimal_precision, marker='o', markersize = 10, color = 'red', label = f'threshold={optimal_threshold_pr}')
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


def evaluate_operational(results, k, prob_threshold = -1, ouput_random = False, sampling_number = 10):
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

                sorted_df = prediction_df.sort_values(by=['probability'], ascending=False)

                date = f'{day:02}/{month:02}/{year}'

                infected = len(prediction_df.loc[prediction_df['case'] == 1])

                if (prob_threshold <= 0):
                    discovered = len(sorted_df.head(k).loc[(sorted_df['case'] == 1)])
                else:
                    discovered = len(sorted_df.head(k).loc[(sorted_df['case'] == 1) & (sorted_df['probability'] >= prob_threshold)])
                
                if ouput_random:
                    if infected != 0:
                        idx_list = prediction_df.index.to_list()
                        random_discovery_list = []

                        for _ in range(sampling_number):
                            random_sample = random.sample(idx_list, k)
                            random_df = prediction_df[prediction_df.index.isin(random_sample)]
                            random_hits = len(random_df.loc[(random_df['case'] == 1)])
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


def classification_report(results_train, results_test, threshold = 0.5, beta = 1, round_factor = 4):
    
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
        probabilities_train = results_train.loc[results_train['year'] == year]['probability']
        y_train = results_train.loc[results_train['year'] == year]['case'].values.astype(int)
        predictions_train = np.where(probabilities_train > threshold, 1, 0)
        
        bal_acc_train = balanced_accuracy_score(y_train, predictions_train)
        precision_train = precision_score(y_train, predictions_train, labels = [0,1], zero_division = 0)
        recall_train = recall_score(y_train, predictions_train, labels = [0,1], zero_division = 0)
        fb_score_train = fbeta_score(y_train, predictions_train, beta = beta, labels = [0,1], zero_division = 0)
        log_loss_train = log_loss(y_train, probabilities_train, labels = [0,1])
        
        report_train = report_train.append({'Year': f'{year}'.split('.')[0],
                                            'Accuracy': round(bal_acc_train, round_factor),
                                            'Precision': round(precision_train, round_factor),
                                            'Recall': round(recall_train, round_factor), 
                                            fbeta_label : round(fb_score_train, round_factor),
                                            'Loss' : round(log_loss_train, round_factor)}, ignore_index = True)
        
        #metrics on test set
        probabilities_test = results_test.loc[results_test['year'] == year]['probability']
        y_test = results_test.loc[results_test['year'] == year]['case'].values.astype(int)
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