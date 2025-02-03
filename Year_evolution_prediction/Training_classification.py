import pandas as pd
from sklearn import svm
import numpy as np
import re
import matplotlib.pyplot as plt

# Function to evaluate strings as pandas series operations
def string_to_pandas_series_ops(string, variable):
    # Dynamically assign variables from the 'variable' dictionary
    for variable_name in variable.keys():
        variable_ref = f"{variable_name} = variable['{variable_name}']"
        exec(variable_ref)  # Be cautious with eval/exec in general

    # Regex pattern to match variable names like feature1, feature2, etc.
    pattern = r"[a-zA-Z]+\d+"

    # Modify the string to use the variable references
    modified_string = re.sub(pattern, lambda match: f"{match.group()} ", string)

    # Evaluate the modified string and return the result
    result = eval(modified_string)
    return result

# List of years for classification
Class_year = ['1935.0', '1950.0', '1962.0', '1965.0', '1969.0', '1970.0', '1973.0', '1975.0', '1976.0',
              '1982.0', '1985.5', '1986.0', '1988.0', '1991.5', '1994.0', '1994.6666670000002', '1995.0',
              '1995.5', '1996.0', '1997.5', '1998.0', '1999.0', '2001.0', '2005.0', '2006.0', '2007.5', 
              '2008.0', '2009.0', '2011.0', '2014.0', '2017.0', '2018.0', '2019.0', '2023.0']

Train_year = []  # List to store years
Acc_all = []  # List to store accuracies for each year

# Loop over each year in the dataset
for j in Class_year:
    # Read train and test datasets
    df_train = pd.read_csv(f'classification/{j}/train.dat', sep='\t')
    df_test = pd.read_csv(f'classification/{j}/test.dat', sep='\t')

    # Extract class labels for train and test data
    Class_test = [int(df_test.iloc[i, 0].split('group')[-1]) for i in range(len(df_test))]
    Class = [int(df_train.iloc[i, 0].split('group')[-1]) for i in range(len(df_train))]

    # Extract feature names
    variable_names = [f"feature{i}" for i in range(1, df_train.shape[1])]
    variable_ref = {variable_name: df_train[variable_name] for variable_name in variable_names}
    variable_test = {variable_name: df_test[variable_name] for variable_name in variable_names}

    # Read model and feature space files
    with open(f'classification/{j}/models/top0100_D002', 'r') as model_file, open(f'classification/{j}/SIS_subspaces/Uspace.expressions', 'r') as feature_space:
        lines_model = model_file.readlines()
        lines_feature = feature_space.readlines()

    acc_tmp = []  # List to store accuracy for each model

    # Loop through top 100 models
    for k in range(100):
        ID1 = int(lines_model[k + 1].split()[4])
        ID2 = int(lines_model[k + 1].split()[5].replace(')', ''))

        # Get feature expressions for the selected model
        Feature1_string = lines_feature[ID1 - 1].split()[0]
        Feature2_string = lines_feature[ID2 - 1].split()[0]
        
        # Extract features for train and test sets
        feature1 = string_to_pandas_series_ops(Feature1_string, variable_ref)
        feature2 = string_to_pandas_series_ops(Feature2_string, variable_ref)
        feature1_test = string_to_pandas_series_ops(Feature1_string, variable_test)
        feature2_test = string_to_pandas_series_ops(Feature2_string, variable_test)

        # Prepare training and test data matrices
        X = np.array([feature1, feature2]).transpose()
        X_test = np.array([feature1_test, feature2_test]).transpose()
        Y = np.array(Class)
        Y_test = np.array(Class_test)

        # Hyperparameter tuning for the SVM model
        best_C, best_G, best_score = 0, 0, 0

        # Search through different values for C and gamma
        for c in [0.1, 1, 10, 100, 1000]:
            for g in [0.001, 0.01, 0.1, 1, 10]:
                svf_rbf = svm.SVC(C=c, gamma=g, probability=True)
                svf_rbf.fit(X, Y)

                # Choose the best hyperparameters based on model score
                if svf_rbf.score(X, Y) > best_score:
                    best_C, best_G = c, g
                    best_score = svf_rbf.score(X, Y)

        # Train the final SVM model with the best hyperparameters
        svf_rbf = svm.SVC(C=best_C, gamma=best_G, probability=True)
        svf_rbf.fit(X, Y)

        # Compute accuracy on the test set
        acc_tmp.append(svf_rbf.score(X_test, Y_test))

    # Store the average accuracy for this year
    Acc_all.append(np.mean(acc_tmp))
    Train_year.append(float(j))

# Store dataset sizes for train and test data
Train_size = []
Test_size = []
for i in Class_year:
    df_train = pd.read_csv(f'classification/{i}/train.dat', sep='\t')
    df_test = pd.read_csv(f'classification/{i}/test.dat', sep='\t')
    Train_size.append(df_train.shape[0])
    Test_size.append(df_test.shape[0])

# Plotting the results
fig, ax1 = plt.subplots()

# Create a second y-axis for dataset sizes
ax2 = ax1.twinx()

# Plot the accuracy over time
ax1.scatter(Train_year, Acc_all, c='#2a9d8f', edgecolors='black', zorder=2, label='Accuracy')
ax1.plot(Train_year, Acc_all, c='#2a9d8f', zorder=1)

# Plot the dataset sizes (train, test, and total)
ax2.plot(Train_year, Train_size, c='#f4a261', zorder=1, label='Train Size')
ax2.plot(Train_year, Test_size, c='#f4a261', zorder=1)
ax2.plot(Train_year, np.array(Train_size) + np.array(Test_size), c='#f4a261', zorder=1, label='Total Dataset Size')
ax2.scatter(Train_year, Train_size, c='#f4a261', marker='^', edgecolors='black', zorder=2)
ax2.scatter(Train_year, Test_size, c='#f4a261', marker='s', edgecolors='black', zorder=2)

# Set axis labels and title
ax1.set_xlabel('Trained through Year', fontsize=18)
ax1.set_ylabel('Accuracy', color='#2a9d8f', fontsize=18)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax2.set_ylabel('Dataset Size', color='#f4a261', fontsize=18)
ax2.tick_params(axis='y', labelsize=14)

# Add legends
ax1.legend(loc=5)
ax2.legend(loc=1)

# Set plot title and save the figure
plt.title('Classification', fontsize=18)
plt.savefig('Classification.png', dpi=300)
plt.show()

