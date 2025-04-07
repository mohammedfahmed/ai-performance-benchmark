import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_ollama_usage(csv_file='ollama_process_usage.csv', output_dir='results'):
    """
    Analyzes CPU and memory usage of Ollama models over time, generating plots 
    with relative time on the x-axis and saving them to a specified directory.

    Args:
        csv_file (str): Path to the CSV file containing Ollama usage data.
        output_dir (str): Directory to save the generated plots.
    """

    # Load CSV data and convert Timestamp to datetime
    try:
        data = pd.read_csv(csv_file)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate relative time from the first timestamp
    start_time = data['Timestamp'].iloc[0]
    data['Relative Time (s)'] = (data['Timestamp'] - start_time).dt.total_seconds()

    def plot_usage(data, y_column, y_label, title, filename):
        """Helper function to create and save usage plots."""
        plt.figure(figsize=(10, 6))
        for model, group in data.groupby('LLM Model'):
            plt.plot(group['Relative Time (s)'], group[y_column], marker='o', label=model)

        # Calculate and display statistics
        mean_val = data[y_column].mean()
        min_val = data[y_column].min()
        max_val = data[y_column].max()

        plt.title(title)
        plt.xlabel('Relative Time (s)')
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend(title='LLM Models')

        plt.text(0.5, 0.9, f'Mean: {mean_val:.2f} {y_label.split("(")[-1].strip(")")}\nMin: {min_val:.2f} {y_label.split("(")[-1].strip(")")}\nMax: {max_val:.2f} {y_label.split("(")[-1].strip(")")}',
                 transform=plt.gca().transAxes, fontsize=12, va='top', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # Generate CPU usage plot
    plot_usage(data, 'CPU (%)', 'CPU Usage (%)', 'CPU Usage for All LLM Models', 'all_models_cpu_usage_relative_time.png')

    # Generate memory usage plot
    plot_usage(data, 'Memory (GB)', 'Memory Usage (GB)', 'Memory Usage for All LLM Models', 'all_models_memory_usage_relative_time.png')

    print(f"Combined plots with relative time axes have been saved in the '{output_dir}/' folder.")

# Example usage:
analyze_ollama_usage() # Uses default filename and output directory.
# analyze_ollama_usage('my_ollama_data.csv', 'custom_results') # Example with custom file and dir.
