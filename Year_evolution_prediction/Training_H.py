import pandas as pd
from sklearn import svm
import numpy as np
import re
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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

    # Evaluate the modified string
    result = eval(modified_string)
    return result

# List of regression years
Regression_year = ['1969.0', '1970.0', '1973.0', '1975.0', '1976.0', 
                   '1982.0', '1985.5', '1986.0', '1988.0', '1991.5', 
                   '1994.0', '1994.6666670000002', '1995.0', '1995.5', 
                   '1996.0', '1997.5', '1999.0', '2001.0', '2005.0', 
                   '2006.0', '2007.5', '2014.0', '2017.0']

Train_year = []  # List to store years
all_test_loss = []  # List to store RMSE values for each year

# Loop over each year in the dataset
for year in Regression_year:
    # Read train and test datasets
    df_train = pd.read_csv(f'H/{year}/train.dat', sep='\t')
    df_test = pd.read_csv(f'H/{year}/test.dat', sep='\t')
    
    # Extract the target property values
    H_train = df_train['Properties']
    H_test = df_test['Properties']
    
    # Create variable names for features
    variable_names = [f"feature{i}" for i in range(1, df_train.shape[1] - 1)]
    
    # Create dictionaries for train and test feature data
    variable_ref = {var_name: df_train[var_name] for var_name in variable_names}
    variable_test = {var_name: df_test[var_name] for var_name in variable_names}

    # Read the model and feature space files
    with open(f'H/{year}/models/top0100_D002', 'r') as model_file, open(f'H/{year}/SIS_subspaces/Uspace.expressions', 'r') as feature_space:
        lines_model = model_file.readlines()
        lines_feature = feature_space.readlines()

    model_test_loss = []  # List to store test loss for each model

    # Loop over each model in the top 100
    for j in range(100):
        # Extract feature indices from the model file
        ID1 = int(lines_model[j + 1].split()[4])
        ID2 = int(lines_model[j + 1].split()[5].replace(')', ''))

        # Get feature expressions for the selected model
        Feature1_string = lines_feature[ID1 - 1].split()[0]
        Feature2_string = lines_feature[ID2 - 1].split()[0]

        # Evaluate the features for both train and test datasets
        feature1 = string_to_pandas_series_ops(Feature1_string, variable_ref)
        feature2 = string_to_pandas_series_ops(Feature2_string, variable_ref)
        feature1_test = string_to_pandas_series_ops(Feature1_string, variable_test)
        feature2_test = string_to_pandas_series_ops(Feature2_string, variable_test)

        # Prepare training and test data matrices
        X = np.array([feature1, feature2]).transpose()
        X_test = np.array([feature1_test, feature2_test]).transpose()
        Y = np.array(H_train)
        Y_test = np.array(H_test)

        # Split the data into training and validation sets (80% training, 20% validation)
        random.seed(0)
        train_idx = random.sample(range(X.shape[0]), int(X.shape[0] * 0.8))
        val_idx = [i for i in range(X.shape[0]) if i not in train_idx]
        train_idx.sort()
        val_idx.sort()

        # Hyperparameter tuning for the SVM model
        best_e, best_c, best_g = 0.001, 0.001, 0.001
        best_score = -np.inf  # Start with a very low score to ensure a better one is found

        for e in [0.001, 0.1, 1, 10, 100, 1000]:
            for c in [0.001, 0.1, 1, 10, 100, 1000]:
                for g in [0.001, 0.01, 0.1, 1, 10, 100]:
                    # Create and train the SVR model with the current hyperparameters
                    svf_rbf = svm.SVR(C=c, gamma=g, epsilon=e)
                    svf_rbf.fit(X[train_idx], Y[train_idx])

                    # Evaluate the model on the validation set
                    score = svf_rbf.score(X[val_idx], Y[val_idx])
                    if score > best_score:
                        best_e, best_c, best_g = e, c, g
                        best_score = score

        # Train the final model with the best hyperparameters
        svf_rbf = svm.SVR(C=best_c, gamma=best_g, epsilon=best_e)
        svf_rbf.fit(X, Y)

        # Predict on the test set and calculate RMSE
        Y_pred = svf_rbf.predict(X_test)
        model_test_loss.append(mean_squared_error(Y_test, Y_pred, squared=False))

    # Store the average test loss for this year
    all_test_loss.append(np.mean(model_test_loss))
    Train_year.append(float(year))

# Store dataset sizes for train and test data
Train_size = []
Test_size = []
for year in Regression_year:
    df_train = pd.read_csv(f'H/{year}/train.dat', sep='\t')
    df_test = pd.read_csv(f'H/{year}/test.dat', sep='\t')
    Train_size.append(df_train.shape[0])
    Test_size.append(df_test.shape[0])

# Plotting the results
fig, ax1 = plt.subplots()

# Create a second y-axis for dataset sizes
ax2 = ax1.twinx()

# Plot the RMSE for the test set
ax1.scatter(Train_year, all_test_loss, c='#2a9d8f', edgecolors='black', zorder=2, label='Test Loss')
ax1.plot(Train_year, all_test_loss, c='#2a9d8f', zorder=1)

# Plot the dataset sizes (train, test, and total)
ax2.plot(Train_year, Train_size, c='#f4a261', zorder=1, label='Train Size')
ax2.plot(Train_year, Test_size, c='#f4a261', zorder=1)
ax2.plot(Train_year, np.array(Train_size) + np.array(Test_size), c='#f4a261', zorder=1)
ax2.scatter(Train_year, Train_size, c='#f4a261', marker='^', edgecolors='black', zorder=2)
ax2.scatter(Train_year, Test_size, c='#f4a261', marker='s', edgecolors='black', zorder=2)

# Set axis labels and title
ax1.set_xlabel('Trained through Year', fontsize=18)
ax1.set_ylabel('RMSE for Test Set (cal/mol/K)', color='#2a9d8f', fontsize=18)
ax2.set_ylabel('Dataset Size', color='#f4a261', fontsize=18)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

# Add legends
ax1.legend(loc=5)
ax2.legend(loc=1)

# Set plot title and save the figure
plt.title('Enthalpy Prediction', fontsize=18)
plt.savefig('Enthalpy_Prediction.png', dpi=300)
plt.show()

