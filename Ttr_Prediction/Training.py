import pandas as pd
from sklearn import svm
import numpy as np
import re
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, r2_score

def string_to_pandas_series_ops(string, variable):
    """Evaluate a string expression using provided variable references."""
    for variable_name in variable.keys():
        exec(f"{variable_name} = variable['{variable_name}']")
    pattern = r"[a-zA-Z]+\d+"
    modified_string = re.sub(pattern, lambda match: f"{match.group()} ", string)
    return eval(modified_string)

def process_data():
    """Process training and testing data with feature extraction and SVM training."""
    all_train_score = []
    all_test_score = []
    
    for i in range(100):
        df_train = pd.read_csv(f'{i}/train.dat', sep='\t')
        df_test = pd.read_csv(f'./{i}/test.dat', sep='\t')
        
        H_train = df_train['Properties']
        H_test = df_test['Properties']
        variable_names = [f"feature{j}" for j in range(1, df_train.shape[1]-1)]
        variable_ref = {name: df_train[name] for name in variable_names}
        variable_test = {name: df_test[name] for name in variable_names}
        
        with open(f'{i}/models/top0100_D002', 'r') as model_file:
            lines_model = model_file.readlines()
        
        with open(f'{i}/SIS_subspaces/Uspace.expressions', 'r') as feature_space:
            lines_feature = feature_space.readlines()
        
        model_train_score = []
        model_test_score = []
        
        for j in range(100):
            ID1, ID2 = map(int, [lines_model[j+1].split()[4], lines_model[j+1].split()[5].replace(')','')])
            
            Feature1_string = lines_feature[ID1-1].split()[0]
            Feature2_string = lines_feature[ID2-1].split()[0]
            
            feature1 = string_to_pandas_series_ops(Feature1_string, variable_ref)
            feature2 = string_to_pandas_series_ops(Feature2_string, variable_ref)
            feature1_test = string_to_pandas_series_ops(Feature1_string, variable_test)
            feature2_test = string_to_pandas_series_ops(Feature2_string, variable_test)
            
            X = np.array([feature1, feature2]).T
            X_test = np.array([feature1_test, feature2_test]).T
            
            Y = np.array(H_train)
            Y_test = np.array(H_test)
            
            random.seed(0)
            train_idx = sorted(random.sample(range(36), 24))
            val_idx = sorted([x for x in range(36) if x not in train_idx])
            
            best_params = {'epsilon': 0.001, 'C': 0.001, 'gamma': 0.001, 'score': 0}
            
            for e in [0.001, 0.1, 1, 10, 100, 1000]:
                for c in [0.001, 0.1, 1, 10, 100, 1000]:
                    for g in [0.001, 0.01, 0.1, 1, 10, 100]:
                        svr_rbf = svm.SVR(C=c, gamma=g, epsilon=e)
                        svr_rbf.fit(X[train_idx], Y[train_idx])
                        score = svr_rbf.score(X[val_idx], Y[val_idx])
                        if score > best_params['score']:
                            best_params.update({'epsilon': e, 'C': c, 'gamma': g, 'score': score})
            
            svr_rbf = svm.SVR(C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'])
            svr_rbf.fit(X, Y)
            model_train_score.append(svr_rbf.score(X, Y))
            model_test_score.append(svr_rbf.score(X_test, Y_test))
        
        all_train_score.append(model_train_score)
        all_test_score.append(model_test_score)
    
    np.save('all_train_score.npy', all_train_score)
    np.save('all_test_score.npy', all_test_score)

if __name__ == "__main__":
    process_data()

