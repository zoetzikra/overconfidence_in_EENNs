import pandas as pd
import numpy as np

# Read the data files
classifier_df = pd.read_csv('./MSDNet/cifar100_4/tested-classifiers/confidence_correctness.txt', sep='\t')
probe_df = pd.read_csv('./MSDNet/cifar100_4/tested-probes/confidence_correctness.txt', sep='\t')

# Set confidence thresholds
classifier_threshold = 0.75
probe_threshold = 0.75

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

# Calculate average exit index and correctness
total_samples = len(classifier_df)

# Initialize accumulators
total_classifier_exit = 0
total_probe_exit = 0
correct_classifier_at_probe_exit = 0
correct_classifier_at_classifier_exit = 0

for idx in range(total_samples):
    classifier_row = classifier_df.loc[idx]
    probe_row = probe_df.loc[idx]
    
    # Get exit points
    classifier_exit = get_exit_point(classifier_row, classifier_threshold)
    probe_exit = get_exit_point(probe_row, probe_threshold)
    
    # Accumulate exit indices
    total_classifier_exit += classifier_exit
    total_probe_exit += probe_exit
    
    # Check correctness at probe exit
    if classifier_row[f'Correct_Exit_{probe_exit}'] == 1:
        correct_classifier_at_probe_exit += 1
    
    # Check correctness at classifier exit
    if classifier_row[f'Correct_Exit_{classifier_exit}'] == 1:
        correct_classifier_at_classifier_exit += 1

# Calculate averages
average_classifier_exit = total_classifier_exit / total_samples
average_probe_exit = total_probe_exit / total_samples
average_correctness_at_probe_exit = correct_classifier_at_probe_exit / total_samples
average_correctness_at_classifier_exit = correct_classifier_at_classifier_exit / total_samples

# Write results to a file
with open('exit_analysis_summary.txt', 'w') as f:
    f.write(f'Total samples analyzed: {total_samples}\n\n')
    f.write(f'Average Classifier Exit Index: {average_classifier_exit:.2f}\n')
    f.write(f'Average Probe Exit Index: {average_probe_exit:.2f}\n')
    f.write(f'Average Correctness at Probe Exit: {average_correctness_at_probe_exit:.2f}\n')
    f.write(f'Average Correctness at Classifier Exit: {average_correctness_at_classifier_exit:.2f}\n')