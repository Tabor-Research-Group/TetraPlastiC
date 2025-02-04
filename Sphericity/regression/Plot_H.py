import pandas as pd
from sklearn import svm
import numpy as np
import re
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['figure.dpi'] = 300

# Function to convert a string expression into a pandas Series operation
def string_to_pandas_series_ops(string, variable):
    for variable_name, series in variable.items():
        exec(f"{variable_name} = series")
    return eval(re.sub(r"[a-zA-Z]+\d+", lambda match: f"{match.group()} ", string))

Selected_feature_space = []
# Loop over 100 datasets
for i in range(100):
    # Load training and testing data
    df_train = pd.read_csv(f'H/{i}/train.dat', sep='\t')
    df_test = pd.read_csv(f'H/{i}/test.dat', sep='\t')
    
    # Extract feature names
    variable_names = [f"feature{j}" for j in range(1, df_train.shape[1] - 1)]
    variable_ref = {name: df_train[name] for name in variable_names}
    variable_test = {name: df_test[name] for name in variable_names}
    
    # Read model and feature space files
    with open(f'H/{i}/models/top0100_D002', 'r') as model_file,          open(f'H/{i}/SIS_subspaces/Uspace.expressions', 'r') as feature_space,          open(f'./feature_name_{i}', "rb") as fp:
        lines_model = model_file.readlines()
        lines_feature = feature_space.readlines()
        feature_name = pickle.load(fp)
    
    tmp2 = []
    # Loop over 100 models
    for j in range(100):
        ID1, ID2 = map(int, [lines_model[j+1].split()[4], lines_model[j+1].split()[5].replace(')', '')])
        Feature1_string, Feature2_string = lines_feature[ID1-1].split()[0], lines_feature[ID2-1].split()[0]
        feature1 = string_to_pandas_series_ops(Feature1_string, variable_ref)
        feature2 = string_to_pandas_series_ops(Feature2_string, variable_ref)
        
        # Extract feature indices
        tmp = [feature_name[int(k)-1] for k in re.findall(r'feature(\d+)', Feature1_string)]
        tmp += [feature_name[int(k)-1] for k in re.findall(r'feature(\d+)', Feature2_string)]
        tmp2.append(tmp)
    Selected_feature_space.append(tmp2)

# Identify frequently selected features
selected_feature, feature_counts = np.unique([z for y in Selected_feature_space for x in y for z in x], return_counts=True)

# Load test scores
all_test_score = np.load('all_H_test_score.npy')
all_score_feature = [all_test_score[i][j] for i in range(len(Selected_feature_space)) for j in range(len(Selected_feature_space[i])) for k in Selected_feature_space[i][j]]

# Define modified feature labels
selected_feature_modified = np.array([r'$\sigma_\theta$', '$\it{Br}$', '$\it{C_4H_9}$', '$\it{CH_2Cl}$', '$\it{CH_2F}$', '$\it{CH_2NH_2}$',
       '$\it{CHClOH}$', '$\it{CHFOH}$', '$\it{CHO}$', '$\it{COCl}$', '$\it{COOH}$', 'Center of Mass', '$\it{Cl}$',
       '$E_{CT}$', '$E_{dispersion}$', '$\it{F}$',
       'Interaction Energy', 'Maxacc', '$\it{NH_2}$', '$\it{NO_2}$', '$\mu_R$',
       '$\sigma_R$', '$\it{SH}$', 'Symmetry', 'Volume', '$\mu_{dist}$',
       'sphericity', '$\sigma_{acc}$'])

# Categorize scores into bins
score_stacking = []
for i in selected_feature:
    tmp = np.histogram([s for j in np.where(selected_feature == i)[0] for s in np.array(all_score_feature)[j]], bins=[-np.inf, 0, 0.2, 0.4, 0.6, 0.8, np.inf])[0]
    score_stacking.append(tmp)

# Normalize and plot stacked bar chart
all_number = np.sum(score_stacking)
fig, ax = plt.subplots()
labels = ['$R^2$ < 0.0', '0.0 <= $R^2$ < 0.2', '0.2 <= $R^2$ < 0.4','0.4 <= $R^2$ < 0.6',
          '0.6 <= $R^2$ < 0.8','0.8 <= $R^2$ < 1.0']
width = 0.5
color_codes = ['#264653', '#2a9d8f', '#8ab17d', '#e9c46a', '#f4a261','#8c510a']

for i in range(6):
    bottom = np.sum(np.array(score_stacking)[:, :i], axis=1) / all_number
    ax.bar(selected_feature_modified, np.array(score_stacking)[:, i] / all_number, width, label=labels[i], bottom=bottom, color=color_codes[i])

plt.xticks(rotation='vertical')
plt.xlabel('Descriptors')
plt.ylabel('Ratio in the SISSO Model Ensemble')
plt.title('Selected Descriptors Ratio from Enthalpy Prediction')
plt.legend()
plt.savefig('Selected Descriptors Ratio from Enthalpy Prediction.png',dpi=300)
