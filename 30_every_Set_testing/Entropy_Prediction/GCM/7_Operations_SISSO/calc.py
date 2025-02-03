import pandas as pd
import itertools
import numpy as np
from numpy import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import pickle
from joblib import Parallel, delayed
import multiprocessing
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

def RMSE(y_true,y_pred):
    return sqrt(sum((y_true-y_pred)**2)/len(y_pred))

def coeff_determination(y_true, y_pred):
    SS_res =  sum((y_true-y_pred)**2)
    SS_tot = sum((y_true-sum(y_true)/len(y_true))**2)
    return ( 1 - SS_res/(SS_tot) )

def get_loss_and_score(subset,stuff,X,Y,best_C,best_G,best_E):
    X_train_unscaled = []
    Y_train = []
    X_test_unscaled = []
    Y_test = []
    for i in subset:
        X_train_unscaled.append(X[i])
        Y_train.append(Y[i])
    for i in stuff:
        if i not in subset:
            X_test_unscaled.append(X[i])
            Y_test.append(Y[i])
    scaler = MinMaxScaler()
    scaler.fit(X_train_unscaled)
    X_train = scaler.transform(X_train_unscaled)
    X_test = scaler.transform(X_test_unscaled)
    svr_rbf = SVR(kernel="rbf", C = best_C, gamma = best_G, epsilon = best_E)
    svr_rbf.fit(X_train,Y_train)
    Y_pred = svr_rbf.predict(X_test)
    Y_train_pred = svr_rbf.predict(X_train)
    return [RMSE(Y_train,Y_train_pred),RMSE(Y_test,Y_pred),coeff_determination(Y_train,Y_train_pred),coeff_determination(Y_test,Y_pred)]

df_train = pd.read_csv('30_GCM.csv',sep='\t')
variable_names = ["feature"+str(i) for i in range(1,df_train.shape[1]-1,1)]
variable_ref = {variable_name: df_train[variable_name] for variable_name in variable_names}

i=0

model_file = open('./models/top0100_D002','r')
feature_space = open('./SIS_subspaces/Uspace.expressions','r')
lines_model = model_file.readlines()
lines_feature = feature_space.readlines()

ID1 = int(lines_model[1].split()[4])
ID2 = int(lines_model[1].split()[5].replace(')',''))
Feature1_string = lines_feature[ID1-1].split()[0]
Feature2_string = lines_feature[ID2-1].split()[0]
feature1 = string_to_pandas_series_ops(Feature1_string, variable_ref)
feature2 = string_to_pandas_series_ops(Feature2_string, variable_ref)
X = np.array([feature1, feature2])
X = X.transpose()
Y = np.array(df_train.iloc[:,1])
best_C = 10000
best_G = 1
best_E = 0.1
stuff = [x for x in range(30)]
results_list = Parallel(n_jobs=10)(delayed(get_loss_and_score)(subset,stuff,X,Y,best_C,best_G,best_E) for subset in itertools.combinations(stuff, 24))

train_losses = []
train_scores = []
test_losses = []
test_scores = []
for i in range(len(results_list)):
    train_losses.append(results_list[i][0])
    test_losses.append(results_list[i][1])
    train_scores.append(results_list[i][2])
    test_scores.append(results_list[i][3])

with open('train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)
with open('train_scores.pkl', 'wb') as f:
    pickle.dump(train_scores, f)
with open('test_losses.pkl', 'wb') as f:
    pickle.dump(test_losses, f)
with open('test_scores.pkl', 'wb') as f:
    pickle.dump(test_scores, f)
