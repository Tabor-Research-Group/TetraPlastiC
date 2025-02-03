import pandas as pd
from sklearn import svm
import numpy as np
import re
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Helper function to evaluate expressions with variable references
def string_to_pandas_series_ops(string, variable):
    for variable_name in variable.keys():
        # Dynamically create variables based on the input dictionary 'variable'
        variable_ref = f"{variable_name} = variable['{variable_name}']"
        exec(variable_ref)
    
    # Create a regular expression pattern to match variable names (e.g., feature1, feature2, etc.)
    pattern = r"[a-zA-Z]+\d+"

    # Replace the original string with a modified string that uses the variable references
    modified_string = re.sub(pattern, lambda match: f"{match.group()} ", string)

    # Evaluate the modified string using eval() (use with caution)
    result = eval(modified_string)

    return result

# Initialize lists to store results
Regression_year = ['1969.0', '1970.0', '1973.0', '1975.0', '1976.0', '1982.0', '1985.5', '1986.0', '1988.0', 
                   '1991.5', '1994.0', '1994.6666670000002', '1995.0', '1995.5', '1996.0', '1997.5', '1999.0', 
                   '2001.0', '2005.0', '2006.0', '2007.5', '2014.0', '2017.0']
Train_year = []
all_test_loss = []

# Loop over each year in the regression dataset
for i in Regression_year:
    # Read the training and testing datasets
    df_train = pd.read_csv(f'S/{i}/train.dat', sep='\t')
    df_test = pd.read_csv(f'S/{i}/test.dat', sep='\t')
    H_train = df_train['Properties']
    H_test = df_test['Properties']

    # Extract variable names from training data (assuming the first column is the target variable)
    variable_names = [f"feature{i}" for i in range(1, df_train.shape[1] - 1)]
    variable_ref = {var_name: df_train[var_name] for var_name in variable_names}
    variable_test = {var_name: df_test[var_name] for var_name in variable_names}

    # Read the model and feature space files
    with open(f'S/{i}/models/top0100_D002', 'r') as model_file:
        lines_model = model_file.readlines()
    with open(f'S/{i}/SIS_subspaces/Uspace.expressions', 'r') as feature_space:
        lines_feature = feature_space.readlines()

    model_test_loss = []

    # Loop through the first 100 models
    for j in range(100):
        ID1 = int(lines_model[j+1].split()[4])
        ID2 = int(lines_model[j+1].split()[5].replace(')', ''))

        # Retrieve feature expressions for the selected model
        Feature1_string = lines_feature[ID1-1].split()[0]
        Feature2_string = lines_feature[ID2-1].split()[0]

        # Convert the feature strings to pandas series
        feature1 = string_to_pandas_series_ops(Feature1_string, variable_ref)
        feature2 = string_to_pandas_series_ops(Feature2_string, variable_ref)
        feature1_test = string_to_pandas_series_ops(Feature1_string, variable_test)
        feature2_test = string_to_pandas_series_ops(Feature2_string, variable_test)

        # Prepare the training and testing data matrices
        X = np.array([feature1, feature2]).transpose()
        X_test = np.array([feature1_test, feature2_test]).transpose()
        Y = np.array(H_train)
        Y_test = np.array(H_test)

        # Train-test split (80%-20%)
        random.seed(0)
        train_idx = random.sample(range(X.shape[0]), int(X.shape[0] * 0.8))
        val_idx = [x for x in range(X.shape[0]) if x not in train_idx]
        train_idx.sort()
        val_idx.sort()

        # Hyperparameter tuning for SVR
        best_e, best_c, best_g = 0.001, 0.001, 0.001
        best_score = -np.inf  # Initialize with a low score to compare

        for e in [0.001, 0.1, 1, 10, 100, 1000]:
            for c in [0.001, 0.1, 1, 10, 100, 1000]:
                for g in [0.001, 0.01, 0.1, 1, 10, 100]:
                    svf_rbf = svm.SVR(C=c, gamma=g, epsilon=e)
                    svf_rbf.fit(X[train_idx], Y[train_idx])
                    score = svf_rbf.score(X[val_idx], Y[val_idx])
                    
                    # Update best hyperparameters if current model performs better
                    if score > best_score:
                        best_e, best_c, best_g = e, c, g
                        best_score = score
        
        # Train the final model with the best hyperparameters
        svf_rbf = svm.SVR(C=best_c, gamma=best_g, epsilon=best_e)
        svf_rbf.fit(X, Y)

        # Predict on the test set and calculate the RMSE
        Y_pred = svf_rbf.predict(X_test)
        model_test_loss.append(mean_squared_error(Y_test, Y_pred, squared=False))

    # Store the average test loss for this year
    all_test_loss.append(np.mean(model_test_loss))
    Train_year.append(float(i))

# Store dataset sizes for each year
Train_size = []
Test_size = []
for i in Regression_year:
    df_train = pd.read_csv(f'S/{i}/train.dat', sep='\t')
    df_test = pd.read_csv(f'S/{i}/test.dat', sep='\t')
    Train_size.append(df_train.shape[0])
    Test_size.append(df_test.shape[0])

# Plot results
fig, ax1 = plt.subplots()

# Create a second y-axis for dataset size
ax2 = ax1.twinx()

# Plot the RMSE over time
ax1.scatter(Train_year, all_test_loss, c='#2a9d8f', edgecolors='black', zorder=2)
ax1.plot(Train_year, all_test_loss, c='#2a9d8f', zorder=1)

# Plot the dataset sizes (train, test, and total)
ax2.plot(Train_year, Train_size, c='#f4a261', zorder=1)
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

plt.title('Entropy Prediction', fontsize=18)

# Save the plot as a high-quality image
plt.savefig('Entropy_Prediction.png', dpi=300)
plt.show()

