import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_confidence_trajectories(confidence_file, n_samples=6):
    # Read the data
    df = pd.read_csv(confidence_file, sep='\t')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Randomly sample n_samples from the dataset
    sample_indices = np.random.randint(0, len(df), n_samples)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Get confidence columns
    conf_cols = [f'Conf_Exit_{i}' for i in range(1, 5)]  # Assuming 4 exit points
    
    # Plot confidence trajectories for each sampled point
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))  # Different color for each trajectory
    
    for idx, sample_idx in enumerate(sample_indices):
        confidences = [df.iloc[sample_idx][col] for col in conf_cols]
        plt.plot(range(1, len(conf_cols) + 1), confidences, 'o-', 
                color=colors[idx], 
                linewidth=2, 
                markersize=8,
                label=f'Sample {idx+1}')
    
    # Customize plot
    plt.xlabel('Early Exit', fontsize=12)
    plt.ylabel('y* Probability', fontsize=12)
    plt.title('Confidence Trajectories', fontsize=14, pad=15)
    
    # Set axis properties
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(range(1, len(conf_cols) + 1))
    plt.ylim(0, 1.0)
    
    # Add legend
    plt.legend()
    
    plt.tight_layout()
    return plt

def main():
    confidence_file = 'MSDNet/cifar100_4/tested-classifiers/confidence_correctness.txt'
    
    # Create and save plot
    plt = plot_confidence_trajectories(confidence_file)
    plt.savefig('confidence_trajectories_classifiers.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()