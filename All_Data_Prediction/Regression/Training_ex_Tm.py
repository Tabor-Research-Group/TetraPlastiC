import pandas as pd

import numpy as np

import random

from sklearn import svm

import re



def string_to_pandas_series_ops(string,variable):

    for variable_name in variable.keys():

        variable_ref = f"{variable_name} = variable['{variable_name}']"

        exec(variable_ref)

    # Create a regular expression pattern to match variable names

    pattern = r"[a-zA-Z]+\d+"



    # Replace the original string with a modified string that uses the variable references

    modified_string = re.sub(pattern, lambda match: f"{match.group()} ", string)



    # Evaluate the modified string using the `eval` function

    result = eval(modified_string)



    # Return the result and the variable references

    return result



def load_data(index):

    df_train = pd.read_csv(f'Tm/{index[0]}/{index[1]}/train.dat', sep='\t')

    df_test = pd.read_csv(f'Tm/{index[0]}/{index[1]}/test.dat', sep='\t')

    H_train, H_test = df_train['Properties'], df_test['Properties']

    variable_names = [f"feature{i}" for i in range(1, df_train.shape[1] - 1)]

    variable_ref = {name: df_train[name] for name in variable_names}

    variable_test = {name: df_test[name] for name in variable_names}

    return H_train, H_test, variable_ref, variable_test



def read_model_files(index):

    with open(f'Tm/{index[0]}/{index[1]}/Models/top0100_D002', 'r') as f:

        lines_model = f.readlines()

    with open(f'Tm/{index[0]}/{index[1]}/SIS_subspaces/Uspace.expressions', 'r') as f:

        lines_feature = f.readlines()

    return lines_model, lines_feature



def get_features(lines_model, lines_feature, variable_ref, variable_test, k):

    ID1, ID2 = map(lambda x: int(x.replace(')', '')), lines_model[k + 1].split()[4:6])

    Feature1_string, Feature2_string = lines_feature[ID1 - 1].split()[0], lines_feature[ID2 - 1].split()[0]

    feature1, feature2 = string_to_pandas_series_ops(Feature1_string, variable_ref), string_to_pandas_series_ops(Feature2_string, variable_ref)

    feature1_test, feature2_test = string_to_pandas_series_ops(Feature1_string, variable_test), string_to_pandas_series_ops(Feature2_string, variable_test)

    return np.column_stack((feature1, feature2)), np.column_stack((feature1_test, feature2_test))



def train_best_svr(X, Y, train_idx, val_idx):

    best_params = {'epsilon': 0.001, 'C': 0.001, 'gamma': 0.001, 'score': 0}

    for e in [0.001, 0.1, 1, 10, 100, 1000]:

        for c in [0.001, 0.1, 1, 10, 100, 1000]:

            for g in [0.001, 0.01, 0.1, 1, 10, 100]:

                model = svm.SVR(C=c, gamma=g, epsilon=e)

                model.fit(X[train_idx], Y[train_idx])

                score = model.score(X[val_idx], Y[val_idx])

                if score > best_params['score']:

                    best_params.update({'epsilon': e, 'C': c, 'gamma': g, 'score': score})

    return svm.SVR(C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'])



def evaluate_models():

    all_model_Y_train, all_model_Y_test = [], []

    for i in range(100):

        cv_model_Y_train, cv_model_Y_test = [], []

        for j in range(7):

            H_train, H_test, variable_ref, variable_test = load_data([i,j])

            lines_model, lines_feature = read_model_files([i,j])

            model_Y_train, model_Y_test = [], []



            for k in range(100):

                X, X_test = get_features(lines_model, lines_feature, variable_ref, variable_test, k)

                Y, Y_test = np.array(H_train), np.array(H_test)



                random.seed(0)

                train_idx = sorted(random.sample(range(42), 35))

                val_idx = sorted(set(range(42)) - set(train_idx))



                svr_model = train_best_svr(X, Y, train_idx, val_idx)

                svr_model.fit(X, Y)

                model_Y_train.append([Y, svr_model.predict(X)])

                model_Y_test.append([Y_test, svr_model.predict(X_test)])



            cv_model_Y_train.append(model_Y_train)

            cv_model_Y_test.append(model_Y_test)

        all_model_Y_train.append(cv_model_Y_train)

        all_model_Y_test.append(cv_model_Y_test)

    return all_model_Y_train, all_model_Y_test



all_model_Y_train, all_model_Y_test = evaluate_models()

np.save('all_model_Y_train_Tm.npy',all_model_Y_train)

np.save('all_model_Y_test_Tm.npy',all_model_Y_test)
