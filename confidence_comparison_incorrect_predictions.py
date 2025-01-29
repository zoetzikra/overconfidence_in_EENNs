import pandas as pd
import numpy as np

# Read the data files
classifier_df = pd.read_csv('./confidence_correctness_classifiers.txt', sep='\t')
probe_df = pd.read_csv('./confidence_correctness_probes.txt', sep='\t')
# Set confidence thresholds
classifier_threshold = 0.85
probe_threshold = 0.70

# Find indices where all exits were incorrect in classifier predictions
all_wrong_mask = (classifier_df['Correct_Exit_1'] == 0) & \
                 (classifier_df['Correct_Exit_2'] == 0) & \
                 (classifier_df['Correct_Exit_3'] == 0) & \
                 (classifier_df['Correct_Exit_4'] == 0)
wrong_indices = classifier_df[all_wrong_mask].index

# Function to determine exit point based on confidence threshold
def get_exit_point(row, threshold):
    if row['Conf_Exit_1'] >= threshold:
        return 1
    elif row['Conf_Exit_2'] >= threshold:
        return 2
    elif row['Conf_Exit_3'] >= threshold:
        return 3
    elif row['Conf_Exit_4'] >= threshold:
        return 4
    return 4  # Default to last exit if no confidence exceeds threshold

# Create output file
with open('exit_analysis_70-85.txt', 'w') as f:
    f.write('Sample\tClassifier_Exit\tProbe_Exit\tMin_Exit\tDecision_Maker\n')
    
    for idx in wrong_indices:
        classifier_row = classifier_df.loc[idx]
        probe_row = probe_df.loc[idx]
        
        # Get exit points
        classifier_exit = get_exit_point(classifier_row, classifier_threshold)
        probe_exit = get_exit_point(probe_row, probe_threshold)
        
        # Determine minimum exit and decision maker
        min_exit = min(classifier_exit, probe_exit)
        if probe_exit < classifier_exit:
            decision_maker = 'probe'
        elif probe_exit == classifier_exit:
            decision_maker = 'same'
        else:
            decision_maker = 'classifier'
        
        # Write results
        f.write(f'{idx}\t{classifier_exit}\t{probe_exit}\t{min_exit}\t{decision_maker}\n')

# Print summary statistics
with open('exit_analysis_summary_70-85.txt', 'w') as f:
    total_samples = len(wrong_indices)
    f.write(f'Total samples analyzed: {total_samples}\n\n')
    
    # Count occurrences of each exit point
    f.write('Classifier Exit Distribution:\n')
    for exit_point in range(1, 5):
        count = sum(1 for idx in wrong_indices if get_exit_point(classifier_df.loc[idx], classifier_threshold) == exit_point)
        percentage = (count / total_samples) * 100
        f.write(f'Exit {exit_point}: {count} ({percentage:.1f}%)\n')
    
    f.write('\nProbe Exit Distribution:\n')
    for exit_point in range(1, 5):
        count = sum(1 for idx in wrong_indices if get_exit_point(probe_df.loc[idx], probe_threshold) == exit_point)
        percentage = (count / total_samples) * 100
        f.write(f'Exit {exit_point}: {count} ({percentage:.1f}%)\n')
    
    # Count decision maker distribution
    classifier_decisions = sum(1 for idx in wrong_indices if 
                             get_exit_point(classifier_df.loc[idx], classifier_threshold) < 
                             get_exit_point(probe_df.loc[idx], probe_threshold))
    probe_decisions = sum(1 for idx in wrong_indices if 
                            get_exit_point(probe_df.loc[idx], probe_threshold) < 
                            get_exit_point(classifier_df.loc[idx], classifier_threshold))
    
    f.write('\nDecision Maker Distribution:\n')
    f.write(f'Classifier: {classifier_decisions} ({(classifier_decisions/total_samples)*100:.1f}%)\n')
    f.write(f'Probe: {probe_decisions} ({(probe_decisions/total_samples)*100:.1f}%)\n')

    # Initialize accumulators
    total_classifier_exit = 0
    total_probe_exit = 0

    for idx in wrong_indices:
        classifier_row = classifier_df.loc[idx]
        probe_row = probe_df.loc[idx]
        
        # Get exit points
        classifier_exit = get_exit_point(classifier_row, classifier_threshold)
        probe_exit = get_exit_point(probe_row, probe_threshold)
        
        # Accumulate exit indices
        total_classifier_exit += classifier_exit
        total_probe_exit += probe_exit
        
    # Calculate averages
    average_classifier_exit = total_classifier_exit / total_samples
    average_probe_exit = total_probe_exit / total_samples

    f.write(f'Total incorrect samples analyzed: {total_samples}\n\n')
    f.write(f'Average Classifier Exit Index: {average_classifier_exit:.2f}\n')
    f.write(f'Average Probe Exit Index: {average_probe_exit:.2f}\n')
