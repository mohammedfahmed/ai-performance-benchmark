import ollama
import time

# List of popular LLMs to evaluate
models = ["llama2", "gpt-4", "gpt-3.5", "bloom"]

# Sample evaluation prompt
evaluation_prompt = """
Please respond with a brief summary of the following text:
'Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.'
"""

# Function to pull models if not available
def pull_model(model_name):
    try:
        print(f"Attempting to pull model: {model_name}")
        ollama.pull(model_name)
        print(f"Model {model_name} pulled successfully.")
    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")

# Function to evaluate a model's performance
def evaluate_model(model_name, prompt):
    try:
        start_time = time.time()  # Track the time taken for the response

        # Check if the model is available, if not, try pulling it
        try:
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            print(f"Model {model_name} not found locally. Attempting to pull it...")
            pull_model(model_name)  # Try pulling the model if it's not found
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

        response_time = time.time() - start_time  # Calculate response time

        # Extract the model's answer from the response
        model_response = response['message']
        print(f"Model: {model_name}")
        print(f"Response: {model_response}")
        print(f"Response Time: {response_time:.2f} seconds\n")
    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")

# Evaluate each model in the list
for model in models:
    evaluate_model(model, evaluation_prompt)
