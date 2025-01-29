import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_preprocess_data(file_path):
    """Load and preprocess the confidence_correctness.txt file"""
    # Read the data
    data = pd.read_csv(file_path, delimiter='\t')
    
    # Count number of exits based on confidence columns
    n_exits = len([col for col in data.columns if col.startswith('Conf_Exit_')])
    
    return data, n_exits

def plot_exit_distribution(data, n_exits, save_path='plots_budgeted_task/new_combined_exit_distribution.png'):
    """Create heatmap of exit distributions across budget levels"""
    # Group by budget level and actual exit
    exit_dist = pd.crosstab(data['Budget'], data['Actual_Exit'])
    
    # Convert to percentages
    exit_dist_pct = exit_dist.div(exit_dist.sum(axis=1), axis=0) * 100

    # Create plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(exit_dist_pct.T, cmap='YlOrRd', annot=True, fmt='.1f')
    plt.title('Exit Distribution vs Budget Level (%)')
    plt.xlabel('Budget Level')
    plt.ylabel('Exit Point')
    plt.savefig(save_path)
    plt.close()

def plot_cost_accuracy_tradeoff(data, save_path='plots_budgeted_task/new_combined_cost_accuracy_tradeoff.png'):
    """Create line plot of FLOPs vs accuracy for each budget level"""
    # Calculate metrics for each budget level
    budget_metrics = []
    for budget in data['Budget'].unique():
        budget_data = data[data['Budget'] == budget]
        
        # Calculate accuracy (based on actual exit taken)
        correct_mask = budget_data.apply(
            lambda row: row[f'Correct_Exit_{int(row["Actual_Exit"])}'], axis=1
        )
        accuracy = correct_mask.mean() * 100
        
        # Calculate average FLOPs
        avg_flops = budget_data['Actual_FLOPs'].mean() / 1e6  # Convert to millions
        
        budget_metrics.append({
            'budget': budget,
            'accuracy': accuracy,
            'flops': avg_flops
        })
    
    # Create DataFrame and plot
    metrics_df = pd.DataFrame(budget_metrics)
    plt.figure(figsize=(10, 6))
    
    # Plot only the line, solid (not dotted), no scatter points
    plt.plot(metrics_df['flops'], metrics_df['accuracy'], 'b-', linewidth=2)
    
    plt.xlabel('Average FLOPs (millions)')
    plt.ylabel('Accuracy (%)')
    plt.title('Computational Cost vs Accuracy Trade-off')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_combined_cost_accuracy_tradeoff(save_path='plots_budgeted_task/new_aka_trained_on_val/new_combined_cost_accuracy_tradeoff.png'):
    """Create line plot of FLOPs vs accuracy for all three approaches"""
    plt.figure(figsize=(10, 6))
    
    # Define the data sources and their labels
    sources = [
        {
            # 'path': './RunMethod_9503816_TestCIFAR_dynamic_softmax_combined.out',
            'path': './MSDNet/cifar100_4/new-tested-combined/confidence_correctness.txt',
            'label': 'Combined',
            'color': 'b'
        },
        {
            # 'path': './RunMethod_9420500_TestCIFAR_classifiers_actual_dynamic_softmax.out',
            'path':  './MSDNet/cifar100_4/new-tested-classifiers/confidence_correctness.txt',
            'label': 'Classifiers Only',
            'color': 'r'
        },
        {
            # 'path': './RunMethod_9420699_TestCIFAR_probes_actual_dynamic_softmax.out',
            'path': './MSDNet/cifar100_4/new-tested-probes/confidence_correctness.txt',
            'label': 'Probes Only',
            'color': 'g'
        }
    ]
    
    # Plot each line
    for source in sources:
        # Load and process data
        data, _ = load_and_preprocess_data(source['path'])
        
        # Calculate metrics for each budget level
        budget_metrics = []
        for budget in data['Budget'].unique():
            budget_data = data[data['Budget'] == budget]
            
            # Calculate accuracy (based on actual exit taken)
            correct_mask = budget_data.apply(
                lambda row: row[f'Correct_Exit_{int(row["Actual_Exit"])}'], axis=1
            )
            accuracy = correct_mask.mean() * 100
            
            # Calculate average FLOPs
            avg_flops = budget_data['Actual_FLOPs'].mean() / 1e6  # Convert to millions
            
            budget_metrics.append({
                'budget': budget,
                'accuracy': accuracy,
                'flops': avg_flops
            })
        
        # Create DataFrame and plot
        metrics_df = pd.DataFrame(budget_metrics)
        
        # Plot line with label and color
        plt.plot(metrics_df['flops'], metrics_df['accuracy'], 
                color=source['color'], 
                label=source['label'],
                linewidth=2)
    
    plt.xlabel('Average FLOPs (millions)')
    plt.ylabel('Accuracy (%)')
    plt.title('Computational Cost vs Accuracy Trade-off')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set y-axis limits to start from 52
    plt.ylim(bottom=52)
    
    plt.savefig(save_path)
    plt.close()

def plot_combined_calibration_vs_cost(save_path='plots_budgeted_task/new_aka_trained_on_val/new_combined_calibration_cost.png'):
    """Create line plot of FLOPs vs ECE for all three approaches"""
    plt.figure(figsize=(10, 6))
    
    sources = [
        {
            # 'path': './RunMethod_9503816_TestCIFAR_dynamic_softmax_combined.out',
            'path': './RunMethod_9627870_TestCIFAR_dynamic_softmax_combined.out',
            'label': 'Combined',
            'color': 'b'
        },
        {
            # 'path': './RunMethod_9420500_TestCIFAR_classifiers_actual_dynamic_softmax.out',
            'path':  './RunMethod_9627866_TestCIFAR_dynamic_softmax_classifiers.out',
            'label': 'Classifiers Only',
            'color': 'r'
        },
        {
            # 'path': './RunMethod_9420699_TestCIFAR_probes_actual_dynamic_softmax.out',
            'path': './RunMethod_9612136_TestCIFAR_dynamic_softmax_probes_on_val.out',
            'label': 'Probes Only',
            'color': 'g'
        }
    ]
    
    for source in sources:
        print(f"\nProcessing {source['label']}...")
        budget_metrics = []
        
        try:
            with open(source['path'], 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Look for budget level line with FLOPs
                if 'Budget level' in line and 'FLOPs:' in line:
                    flops = float(line.split('FLOPs:')[1].split('M')[0].strip())
                    
                    # Look ahead for the Accuracy/ECE line (usually 4 lines after)
                    for j in range(i, min(i + 10, len(lines))):
                        if 'Accuracy:' in lines[j] and 'ECE:' in lines[j]:
                            ece = float(lines[j].split('ECE:')[1].split(',')[0].strip())
                            budget_metrics.append({
                                'flops': flops,
                                'ece': ece
                            })
                            break
                i += 1
            
            # Create DataFrame and plot
            metrics_df = pd.DataFrame(budget_metrics)
            print(f"Found {len(metrics_df)} data points")
            print(f"Columns: {metrics_df.columns.tolist()}")
            print(f"First few rows:\n{metrics_df.head()}")
            
            if not metrics_df.empty:
                plt.plot(metrics_df['flops'], metrics_df['ece'], 
                        color=source['color'], 
                        label=source['label'],
                        linewidth=2)
            else:
                print(f"Warning: No data found for {source['label']}")
                
        except FileNotFoundError:
            print(f"Warning: Could not find file {source['path']}")
        except Exception as e:
            print(f"Error processing {source['label']}: {e}")
    
    plt.xlabel('Average FLOPs (millions)')
    plt.ylabel('Expected Calibration Error')
    plt.title('Computational Cost vs Calibration Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_sample_behavior(data, n_exits, save_path='plots_budgeted_task/sample_behavior.png'):
    """Analyze and visualize how samples change exit points across budgets"""
    # Calculate exit point changes for each sample
    sample_metrics = []
    for sample in data['Sample'].unique():
        sample_data = data[data['Sample'] == sample]
        
        # Count exit point changes
        exit_changes = len(sample_data['Actual_Exit'].unique())
        
        # Calculate average confidence across all exits
        conf_cols = [f'Conf_Exit_{i+1}' for i in range(n_exits)]
        avg_conf = sample_data[conf_cols].mean().mean()
        
        sample_metrics.append({
            'sample': sample,
            'exit_changes': exit_changes,
            'avg_confidence': avg_conf
        })
    
    # Create DataFrame and plot
    metrics_df = pd.DataFrame(sample_metrics)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['avg_confidence'], metrics_df['exit_changes'], 
                alpha=0.5, s=20)
    plt.xlabel('Average Confidence')
    plt.ylabel('Number of Exit Point Used')
    plt.title('Sample Behavior Analysis')
    
    # Add density estimation
    sns.kdeplot(data=metrics_df, x='avg_confidence', y='exit_changes', 
                levels=5, color='r', linewidths=1)
    
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_sample_behavior_by_correctness(data, n_exits, save_path_prefix='plots_budgeted_task/new_combined_sample_behavior'):
    """Analyze and visualize sample behavior separately for consistently correct/incorrect predictions"""
    
    # Helper function to check if sample is consistently correct/incorrect across all exits
    def get_sample_consistency(sample_data, n_exits):
        correct_cols = [f'Correct_Exit_{i+1}' for i in range(n_exits)]
        all_correct = sample_data[correct_cols].all(axis=1)
        all_incorrect = ~sample_data[correct_cols].any(axis=1)
        return all_correct, all_incorrect
    
    # Calculate metrics for each sample
    consistently_correct_metrics = []
    consistently_incorrect_metrics = []
    
    for sample in data['Sample'].unique():
        sample_data = data[data['Sample'] == sample]
        all_correct, all_incorrect = get_sample_consistency(sample_data, n_exits)
        
        # Skip samples that aren't consistently correct/incorrect
        if not (all_correct.any() or all_incorrect.any()):
            continue
            
        # Count exit point changes
        exit_changes = len(sample_data['Actual_Exit'].unique())
        
        # Calculate average confidence across all exits
        conf_cols = [f'Conf_Exit_{i+1}' for i in range(n_exits)]
        avg_conf = sample_data[conf_cols].mean().mean()
        
        metrics = {
            'sample': sample,
            'exit_changes': exit_changes,
            'avg_confidence': avg_conf
        }
        
        if all_correct.any():
            consistently_correct_metrics.append(metrics)
        if all_incorrect.any():
            consistently_incorrect_metrics.append(metrics)
    
    # Plot for consistently correct samples
    if consistently_correct_metrics:
        _create_behavior_plot(
            pd.DataFrame(consistently_correct_metrics),
            'Sample Behavior Analysis (Consistently Correct)',
            f'{save_path_prefix}_correct.png'
        )
    
    # Plot for consistently incorrect samples
    if consistently_incorrect_metrics:
        _create_behavior_plot(
            pd.DataFrame(consistently_incorrect_metrics),
            'Sample Behavior Analysis (Consistently Incorrect)',
            f'{save_path_prefix}_incorrect.png'
        )

def _create_behavior_plot(metrics_df, title, save_path):
    """Helper function to create the behavior plot"""
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(metrics_df['avg_confidence'], metrics_df['exit_changes'], 
                alpha=0.5, s=20)
    
    # Add density estimation
    sns.kdeplot(data=metrics_df, x='avg_confidence', y='exit_changes', 
                levels=5, color='r', linewidths=1)
    
    # Add statistics annotation
    stats_text = f"N={len(metrics_df)}\n"
    stats_text += f"Avg Confidence: {metrics_df['avg_confidence'].mean():.3f}\n"
    stats_text += f"Avg Exit Changes: {metrics_df['exit_changes'].mean():.2f}"
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Average Confidence')
    plt.ylabel('Number of Different Exit Points Used')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_fixed_threshold_comparison(results_path, save_dir='plots_budgeted_task'):
    """
    Plot comparison of classifier vs probe thresholds from saved results
    
    Args:
        results_path: Path to the JSON file containing evaluation results
        save_dir: Directory to save the plot
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    plt.figure(figsize=(10, 6))
    
    # Extract data
    probe_thresholds = [r['probe_threshold'] for r in results]
    avg_exits = [r['avg_exit'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    classifier_threshold = results[0]['classifier_threshold']  # Same for all results
    
    # Create scatter plot
    scatter = plt.scatter(avg_exits, accuracies, c=probe_thresholds, cmap='viridis', s=100)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Probe Threshold', fontsize=12)
    
    plt.xlabel('Average Exit Point', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Exit Point for Different Probe Thresholds\n(Classifier threshold = {classifier_threshold:.2f})', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=52)
    
    # Save plot
    save_path = os.path.join(save_dir, f'threshold_comparison_c{classifier_threshold:.2f}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plot saved to {save_path}")


# def plot_exit_distributions_by_correctness(data, budget_levels=[2, 39], save_path_prefix='plots_budgeted_task/exit_dist'):
#     """
#     Plot exit point distributions for specific budget levels, separated by correctness
    
#     Args:
#         data: Dictionary with keys 'combined', 'classifiers', 'probes' containing respective DataFrames
#         budget_levels: List of budget levels to analyze
#         save_path_prefix: Where to save the output plots
#     """
#     for budget_level in budget_levels:
#         for correctness in ['correct', 'incorrect']:
#             plt.figure(figsize=(12, 6))
            
#             # Process each approach
#             for idx, (approach, df) in enumerate(data.items()):
#                 # Filter data for current budget level
#                 level_data = df[df['Budget_Level'] == budget_level]
                
#                 # Get correctness columns and filter
#                 correct_cols = [col for col in df.columns if col.startswith('Correct_Exit_')]
#                 if correctness == 'correct':
#                     samples = level_data[level_data[correct_cols].all(axis=1)]
#                 else:
#                     samples = level_data[~level_data[correct_cols].any(axis=1)]
                
#                 # Get exit distribution
#                 exit_dist = samples['Actual_Exit'].value_counts().sort_index()
                
#                 # Plot
#                 plt.subplot(1, 3, idx+1)
#                 plt.bar(exit_dist.index, exit_dist.values)
#                 plt.title(f'{approach.capitalize()}')
#                 plt.xlabel('Exit Point')
#                 plt.ylabel('Number of Samples')
            
#             plt.suptitle(f'Exit Distribution - Budget Level {budget_level}\n{"Correct" if correctness=="correct" else "Incorrect"} Samples')
#             plt.tight_layout()
#             plt.savefig(f'{save_path_prefix}_budget{budget_level}_{correctness}.png')
#             plt.close()

def main():
    # File path to your confidence_correctness.txt
    file_path = './MSDNet/cifar100_4/new-tested-combined/confidence_correctness.txt'
    
    # Load and process data
    data, n_exits = load_and_preprocess_data(file_path)
    
    # Create all plots
    plot_exit_distribution(data, n_exits)
    plot_cost_accuracy_tradeoff(data)
    plot_sample_behavior_by_correctness(data, n_exits)

    plot_combined_calibration_vs_cost()
    plot_combined_cost_accuracy_tradeoff()


    threshold_results_dir = './MSDNet/cifar100_4/new-tested-combined/fixed_threshold_results'
    for results_file in os.listdir(threshold_results_dir):
        if results_file.endswith('.json'):
            results_path = os.path.join(threshold_results_dir, results_file)
            plot_fixed_threshold_comparison(
                results_path, 
                save_dir='plots_budgeted_task/fixed_threshold_comparisons'
            )
    
    print("Analysis complete! Check the current directory for the generated plots.")

if __name__ == "__main__":
    main()