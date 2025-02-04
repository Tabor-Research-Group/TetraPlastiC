import pandas as pd
from sklearn import svm
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle

plt.rcParams['figure.dpi'] = 300

# Function to convert a string expression into a pandas Series operation
def string_to_pandas_series_ops(string, variable):
    for variable_name, series in variable.items():
        exec(f"{variable_name} = series")
    return eval(re.sub(r"[a-zA-Z]+\d+", lambda match: f"{match.group()} ", string))

Y_pred_score = []
Selected_features = []

# Loop over 100 datasets
for i in range(100):
    # Load training and testing data
    df_train = pd.read_csv(f'{i}/train.dat', sep='\t')
    df_test = pd.read_csv(f'{i}/test.dat', sep='\t')
    
    # Extract class labels from dataset
    Class_test = [int(x.split('group')[-1]) for x in df_test.iloc[:, 0]]
    Class = [int(x.split('group')[-1]) for x in df_train.iloc[:, 0]]
    
    # Extract feature names and store references
    variable_names = [f"feature{l}" for l in range(1, df_train.shape[1])]
    variable_ref = {name: df_train[name] for name in variable_names}
    variable_test = {name: df_test[name] for name in variable_names}
    
    # Read model and feature space files
    with open(f'{i}/models/top0100_D002', 'r') as model_file,          open(f'{i}/SIS_subspaces/Uspace.expressions', 'r') as feature_space,          open(f'./feature_name_{i}', "rb") as fp:
        lines_model = model_file.readlines()
        lines_feature = feature_space.readlines()
        feature_name = pickle.load(fp)
    
    # Process 100 models per dataset
    for j in range(100):
        ID1, ID2 = map(int, [lines_model[j+1].split()[4], lines_model[j+1].split()[5].replace(')', '')])
        Feature1_string, Feature2_string = lines_feature[ID1-1].split()[0], lines_feature[ID2-1].split()[0]
        
        # Convert feature strings to pandas Series
        feature1, feature2 = string_to_pandas_series_ops(Feature1_string, variable_ref), string_to_pandas_series_ops(Feature2_string, variable_ref)
        feature1_test, feature2_test = string_to_pandas_series_ops(Feature1_string, variable_test), string_to_pandas_series_ops(Feature2_string, variable_test)
        
        # Extract and store selected features
        Selected_features.extend([feature_name[int(k)-1] for k in re.findall(r'feature(\d+)', Feature1_string)])
        Selected_features.extend([feature_name[int(k)-1] for k in re.findall(r'feature(\d+)', Feature2_string)])
        
        # Prepare feature matrices
        X, X_test = np.column_stack((feature1, feature2)), np.column_stack((feature1_test, feature2_test))
        Y, Y_test = np.array(Class), np.array(Class_test)
        
        # Hyperparameter tuning for best SVM model
        best_C, best_G, best_score = 0, 0, 0
        for c in [0.1, 1, 10, 100, 1000]:
            for g in [0.001, 0.01, 0.1, 1, 10]:
                svf_rbf = svm.SVC(C=c, gamma=g, probability=True)
                svf_rbf.fit(X, Y)
                if svf_rbf.score(X, Y) > best_score:
                    best_C, best_G, best_score = c, g, svf_rbf.score(X, Y)
        
        # Train best model and record scores
        svf_rbf = svm.SVC(C=best_C, gamma=best_G, probability=True)
        svf_rbf.fit(X, Y)
        Y_pred_score.extend([svf_rbf.score(X_test, Y_test)] * len(re.findall(r'feature(\d+)', Feature1_string) + re.findall(r'feature(\d+)', Feature2_string)))
    
# Analyze selected feature frequency
selected_feature, feature_counts = np.unique(Selected_features, return_counts=True)

# Define modified feature labels for improved visualization
selected_feature_modified = np.array([r'$\sigma_\theta$','$\mu_\theta$' , '$\it{Br}$', '$\it{C_2H_5}$',
        '$\it{C_4H_9}$', '$\it{CH(CH_3)OH}$','$\it{CH_2Br}$','$\it{CH_2Cl}$',
        '$\it{CH_2F}$', '$\it{CH_2NH_2}$', '$\it{CHClOH}$', '$\it{CHFOH}$', '$\it{CHO}$', '$\it{COCl}$',
        '$\it{CONH_2}$', '$\it{COOH}$', 'Center of Mass', '$\it{Cl}$',
       '$E_{CT}$', '$E_{dispersion}$', '$\it{F}$',
       'Interaction Energy', 'Maxacc', '$\it{NH_2}$', '$\it{NO_2}$', '$\it{OH}$', '$\mu_R$',
       '$\sigma_R$', '$\it{SH}$', 'Volume', '$\mu_{dist}$',
       'sphericity'])

# Prepare stacked bar plot
score_stacking = []
for i in selected_feature:
    tmp = np.histogram(np.array(Y_pred_score)[np.where(np.array(Selected_features) == i)[0]], bins=[-np.inf, 0, 0.2, 0.4, 0.6, 0.8, np.inf])[0]
    score_stacking.append(tmp)
    
# Normalize and plot stacked bar chart
all_number = np.sum(score_stacking)
fig, ax = plt.subplots()
labels = ['score < 0.0', '0.2 <= score < 0.0', '0.4 <= score < 0.2', '0.6 <= score < 0.4',
          '0.8 <= score < 0.6', '1.0 <= score < 0.8']
width = 0.5
color_codes = ['#264653', '#2a9d8f', '#8ab17d', '#e9c46a', '#f4a261', '#8c510a']

for i in range(6):
    bottom = np.sum(np.array(score_stacking)[:, :i], axis=1) / all_number
    ax.bar(selected_feature_modified, np.array(score_stacking)[:, i] / all_number, width, label=labels[i], bottom=bottom, color=color_codes[i])

plt.xticks(rotation='90')
plt.xlabel('Descriptors', fontsize=32)
plt.ylabel('Ratio in the SISSO Model Ensemble', fontsize=32)
plt.xticks(fontsize=34)
plt.yticks(fontsize=32)
ax.legend(fontsize=24)
plt.title('Selected Descriptors Ratio from Classification', fontsize=32)

