# AI Performance Benchmarking

This repository provides a benchmarking tool to evaluate your PC's performance for neural network training and large language model (LLM) inference. It uses PyTorch to test both CPU/GPU utilization and model performance for various deep learning tasks.

## Features

- **Neural Network Training Benchmark:** 
  - Measures the time and resource consumption for training a simple neural network model on a random dataset.
  
- **LLM Inference Benchmark:**
  - Measures the time and resource consumption for performing inference using a pre-trained large language model (GPT-2).
  
- **System Information:**
  - Provides real-time insights into CPU, GPU (if available), and memory usage during benchmarking.

## Requirements

- Python 3.x
- PyTorch (with CUDA support for GPU benchmarking)
- Hugging Face `transformers` library for LLM inference
- `psutil` for monitoring system resource usage

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ai-performance-benchmark.git
cd ai-performance-benchmark
```

### Step 2: Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate  # For Windows
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can manually install the dependencies:

```bash
pip install torch psutil transformers
```

### Step 3: Set up CUDA (optional)

If you want to use GPU for benchmarking, ensure that you have CUDA and cuDNN installed on your machine. You can follow the instructions from the official PyTorch website: https://pytorch.org/get-started/locally/

## Usage

### Step 1: Run the Benchmark Script

To start the benchmark, simply run the `benchmark.py` script:

```bash
python benchmark.py
```

The script will output the following:

- **CPU Usage:** The percentage of CPU being utilized during the benchmark.
- **GPU Memory Usage:** If you have a CUDA-enabled GPU, it will display memory usage and GPU load.
- **Training Time for Neural Network:** The time taken to train a simple neural network for a specified number of epochs.
- **Inference Time for LLM:** The time taken to perform inference using a pre-trained GPT-2 model.

### Step 2: Modify the Script

You can modify the script for more advanced benchmarking, such as testing with different models, datasets, or system configurations. Simply edit the `benchmark.py` file to change the training or inference configurations.

## Example Output

```bash
CPU Usage: 25%
Memory Usage: 60%
GPU Memory Usage: 1024.0 MB
GPU Usage: 60.0 MB

Starting Neural Network Benchmark...
Neural Network Training Time: 15.23 seconds

Starting LLM Inference Benchmark...
Generated Text: The future of AI is exciting, with advances in deep learning and natural language processing.
LLM Inference Time: 2.32 seconds
```

## Contributing

Feel free to open issues or submit pull requests if you have improvements or bug fixes! Contributions are always welcome.

### Example Contributions

- Add support for more complex neural network models.
- Support additional language models for inference.
- Improve the system resource monitoring and visualization.

## License

This project is licensed under the MIT License.

