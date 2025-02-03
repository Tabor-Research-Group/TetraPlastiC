import pandas as pd
from sklearn import svm
import numpy as np
import random

# Function to load training and testing data for a given index
def load_data(index):
    df_train = pd.read_csv(f'H/{index}/train.dat', sep='\t')
    df_test = pd.read_csv(f'H/{index}/test.dat', sep='\t')
    H_train, H_test = df_train['Properties'], df_test['Properties']
    
    # Extract feature names and store references
    variable_names = [f"feature{i}" for i in range(1, df_train.shape[1] - 1)]
    variable_ref = {name: df_train[name] for name in variable_names}
    variable_test = {name: df_test[name] for name in variable_names}
    
    return H_train, H_test, variable_ref, variable_test

# Function to read model and feature space files
def read_model_files(index):
    with open(f'H/{index}/models/top0100_D002', 'r') as f:
        lines_model = f.readlines()
    with open(f'H/{index}/SIS_subspaces/Uspace.expressions', 'r') as f:
        lines_feature = f.readlines()
    return lines_model, lines_feature

# Function to extract features from model and feature space files
def get_features(lines_model, lines_feature, variable_ref, variable_test, j):
    ID1, ID2 = map(lambda x: int(x.replace(')', '')), lines_model[j + 1].split()[4:6])
    Feature1_string, Feature2_string = lines_feature[ID1 - 1].split()[0], lines_feature[ID2 - 1].split()[0]
    
    # Convert feature strings to pandas Series
    feature1, feature2 = string_to_pandas_series_ops(Feature1_string, variable_ref), string_to_pandas_series_ops(Feature2_string, variable_ref)
    feature1_test, feature2_test = string_to_pandas_series_ops(Feature1_string, variable_test), string_to_pandas_series_ops(Feature2_string, variable_test)
    
    return np.column_stack((feature1, feature2)), np.column_stack((feature1_test, feature2_test))

# Function to train an SVR model with the best hyperparameters
def train_best_svr(X, Y, train_idx, val_idx):
    best_params = {'epsilon': 0.001, 'C': 0.001, 'gamma': 0.001, 'score': 0}
    
    # Grid search for best SVR hyperparameters
    for e in [0.001, 0.1, 1, 10, 100, 1000]:
        for c in [0.001, 0.1, 1, 10, 100, 1000]:
            for g in [0.001, 0.01, 0.1, 1, 10, 100]:
                model = svm.SVR(C=c, gamma=g, epsilon=e)
                model.fit(X[train_idx], Y[train_idx])
                score = model.score(X[val_idx], Y[val_idx])
                
                # Update best parameters if current score is better
                if score > best_params['score']:
                    best_params.update({'epsilon': e, 'C': c, 'gamma': g, 'score': score})
    
    return svm.SVR(C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'])

# Function to evaluate models across multiple datasets
def evaluate_models():
    all_train_score, all_test_score = [], []
    
    # Loop over 100 datasets
    for i in range(100):
        H_train, H_test, variable_ref, variable_test = load_data(i)
        lines_model, lines_feature = read_model_files(i)
        model_train_score, model_test_score = [], []
        
        # Train and evaluate 100 models per dataset
        for j in range(100):
            X, X_test = get_features(lines_model, lines_feature, variable_ref, variable_test, j)
            Y, Y_test = np.array(H_train), np.array(H_test)
            
            # Split data into training and validation sets
            random.seed(0)
            train_idx = sorted(random.sample(range(36), 24))
            val_idx = sorted(set(range(36)) - set(train_idx))
            
            # Train the best SVR model
            svr_model = train_best_svr(X, Y, train_idx, val_idx)
            svr_model.fit(X, Y)
            
            # Store model performance scores
            model_train_score.append(svr_model.score(X, Y))
            model_test_score.append(svr_model.score(X_test, Y_test))
        
        all_train_score.append(model_train_score)
        all_test_score.append(model_test_score)
    
    return all_train_score, all_test_score

# Run model evaluation
all_train_score, all_test_score = evaluate_models()
