import pandas as pd
from sklearn import svm
import numpy as np
import re
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product

def string_to_pandas_series_ops(expression, variable_dict):
    """Evaluate a string expression using provided variable references."""
    pattern = r"[a-zA-Z]+\d+"
    modified_expression = re.sub(pattern, lambda match: f"variable_dict['{match.group()}']", expression)
    return eval(modified_expression, {"variable_dict": variable_dict})

def process_data():
    """Process training and testing data with feature extraction and SVM training."""
    all_train_scores = []
    all_test_scores = []
    
    for i in range(100):
        try:
            df_train = pd.read_csv(f'{i}/train.dat', sep='\t')
            df_test = pd.read_csv(f'./{i}/test.dat', sep='\t')
        except FileNotFoundError as e:
            print(f"Skipping iteration {i}: {e}")
            continue
        
        H_train = df_train['Properties']
        H_test = df_test['Properties']
        variable_names = [f"feature{j}" for j in range(1, df_train.shape[1]-1)]
        variable_ref = {name: df_train[name] for name in variable_names}
        variable_test = {name: df_test[name] for name in variable_names}
        
        try:
            with open(f'{i}/models/top0100_D002', 'r') as model_file:
                lines_model = model_file.readlines()
            with open(f'{i}/SIS_subspaces/Uspace.expressions', 'r') as feature_space:
                lines_feature = feature_space.readlines()
        except FileNotFoundError as e:
            print(f"Skipping iteration {i} due to missing file: {e}")
            continue
        
        model_train_scores = []
        model_test_scores = []
        
        for j in range(100):
            try:
                ID1, ID2 = map(int, [lines_model[j+1].split()[4], lines_model[j+1].split()[5].replace(')', '')])
            except (IndexError, ValueError) as e:
                print(f"Skipping model {j} in iteration {i} due to parsing error: {e}")
                continue
            
            Feature1_string = lines_feature[ID1 - 1].split()[0]
            Feature2_string = lines_feature[ID2 - 1].split()[0]
            
            feature1_train = string_to_pandas_series_ops(Feature1_string, variable_ref)
            feature2_train = string_to_pandas_series_ops(Feature2_string, variable_ref)
            feature1_test = string_to_pandas_series_ops(Feature1_string, variable_test)
            feature2_test = string_to_pandas_series_ops(Feature2_string, variable_test)
            
            X_train = np.column_stack((feature1_train, feature2_train))
            X_test = np.column_stack((feature1_test, feature2_test))
            
            Y_train = H_train.values
            Y_test = H_test.values
            
            random.seed(0)
            train_idx = sorted(random.sample(range(36), 24))
            val_idx = sorted(set(range(36)) - set(train_idx))
            
            best_params = {"C": 0.001, "gamma": 0.001, "epsilon": 0.001}
            best_score = float('-inf')
            
            for e, c, g in product([0.001, 0.1, 1, 10, 100, 1000], [0.001, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100]):
                svr_model = svm.SVR(C=c, gamma=g, epsilon=e)
                svr_model.fit(X_train[train_idx], Y_train[train_idx])
                score = svr_model.score(X_train[val_idx], Y_train[val_idx])
                if score > best_score:
                    best_params = {"C": c, "gamma": g, "epsilon": e}
                    best_score = score
            
            final_svr = svm.SVR(**best_params)
            final_svr.fit(X_train, Y_train)
            
            model_train_scores.append(final_svr.score(X_train, Y_train))
            model_test_scores.append(final_svr.score(X_test, Y_test))
        
        all_train_scores.append(model_train_scores)
        all_test_scores.append(model_test_scores)
    
    np.save('all_train_score.npy', all_train_scores)
    np.save('all_test_score.npy', all_test_scores)

if __name__ == "__main__":
    process_data()


