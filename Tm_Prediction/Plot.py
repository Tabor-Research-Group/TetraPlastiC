import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt

def load_scores():
    """Load precomputed training and test scores from files."""
    return np.load('all_train_score.npy'), np.load('all_test_score.npy')

def load_feature_names(index):
    """Load feature names from a pickle file."""
    with open(f'./feature_name_{index}', "rb") as fp:
        return pickle.load(fp)

def extract_feature_indices(feature_string):
    """Extract feature indices from a feature string."""
    numbers = re.findall(r'feature(\d+)', feature_string)
    return list(map(int, numbers))

def process_feature_selection():
    """Process the feature selection across multiple iterations."""
    Selected_feature_space = []
    
    for i in range(100):
        tmp2 = []
        # Load training and test datasets
        df_train = pd.read_csv(f'{i}/train.dat', sep='\t')
        df_test = pd.read_csv(f'{i}/test.dat', sep='\t')
        
        # Extract properties and variable names
        variable_names = [f"feature{j}" for j in range(1, df_train.shape[1]-1)]
        variable_ref = {name: df_train[name] for name in variable_names}
        variable_test = {name: df_test[name] for name in variable_names}
        
        # Load model and feature space information
        with open(f'{i}/models/top0100_D002', 'r') as model_file:
            lines_model = model_file.readlines()
        
        with open(f'{i}/SIS_subspaces/Uspace.expressions', 'r') as feature_space:
            lines_feature = feature_space.readlines()
        
        feature_name = load_feature_names(i)
        
        for j in range(100):
            tmp = []
            ID1, ID2 = map(int, [lines_model[j+1].split()[4], lines_model[j+1].split()[5].replace(')','')])
            
            Feature1_string = lines_feature[ID1-1].split()[0]
            Feature2_string = lines_feature[ID2-1].split()[0]
            
            # Extract features
            for k in extract_feature_indices(Feature1_string):
                tmp.append(feature_name[k-1])
            for k in extract_feature_indices(Feature2_string):
                tmp.append(feature_name[k-1])
            
            tmp2.append(tmp)
        Selected_feature_space.append(tmp2)
    
    return Selected_feature_space

def compute_feature_scores(Selected_feature_space, all_test_score):
    """Compute the frequency of selected features and their associated test scores."""
    selected_feature, feature_counts = np.unique(
        [z for y in Selected_feature_space for x in y for z in x], return_counts=True
    )
    
    all_score_feature = [all_test_score[i][j] 
                         for i in range(len(Selected_feature_space)) 
                         for j in range(len(Selected_feature_space[i]))
                         for _ in Selected_feature_space[i][j]]
    
    score_stacking = []
    for feature in selected_feature:
        tmp = np.zeros(6)
        indices = np.where(np.array([z for y in Selected_feature_space for x in y for z in x]) == feature)[0]
        scores = np.array(all_score_feature)[indices]
        
        # Categorize scores into bins
        for score in scores:
            if score >= 0.8:
                tmp[5] += 1
            elif score >= 0.6:
                tmp[4] += 1
            elif score >= 0.4:
                tmp[3] += 1
            elif score >= 0.2:
                tmp[2] += 1
            elif score >= 0.0:
                tmp[1] += 1
            else:
                tmp[0] += 1
        score_stacking.append(tmp)
    
    return selected_feature, score_stacking

def plot_feature_scores(selected_feature, score_stacking):
    """Generate a bar plot showing the distribution of selected features."""
    selected_feature_modified = np.array([
        r'$\sigma_\theta$', '$\it{Br}$', '$\it{C_4H_9}$', '$\it{CH_2Cl}$', '$\it{CH_2F}$', '$\it{CH_2NH_2}$',
        '$\it{CHClOH}$', '$\it{CHFOH}$', '$\it{CHO}$', '$\it{COCl}$', '$\it{COOH}$', 'Center of Mass', '$\it{Cl}$',
        '$E_{CT}$', '$E_{dispersion}$', '$\it{F}$', 'Interaction Energy', 'Maxacc', '$\it{NH_2}$', '$\it{NO_2}$', '$\mu_R$',
        '$\sigma_R$', '$\it{SH}$', 'Symmetry', 'Volume', '$\mu_{dist}$', 'sphericity', '$\sigma_{acc}$'
    ])
    
    all_number = sum(sum(score_stacking))
    fig, ax = plt.subplots()
    labels = ['score < 0.0', '0.2 <= score < 0.0', '0.4 <= score < 0.2', '0.6 <= score < 0.4',
              '0.8 <= score < 0.6', '1.0 <= score < 0.8']
    width = 0.5
    color_codes = ['#264653', '#2a9d8f', '#8ab17d', '#e9c46a', '#f4a261', '#8c510a']
    
    for i in range(6):
        bottom = np.sum(np.array(score_stacking)[:, :i], axis=1) / all_number
        ax.bar(selected_feature_modified, np.array(score_stacking)[:, i] / all_number, width,
               label=labels[i], bottom=bottom, color=color_codes[i])
    
    plt.xticks(rotation='vertical')
    plt.xlabel('Descriptors', fontsize=12)
    plt.ylabel('Ratio in the SISSO Model Ensemble', fontsize=12)
    ax.legend()
    plt.title('Selected Feature Ratio from Melting Point Prediction', fontsize=12)
    plt.savefig('Prediction_Tm.png')

if __name__ == "__main__":
    all_train_score, all_test_score = load_scores()
    Selected_feature_space = process_feature_selection()
    selected_feature, score_stacking = compute_feature_scores(Selected_feature_space, all_test_score)
    plot_feature_scores(selected_feature, score_stacking)
