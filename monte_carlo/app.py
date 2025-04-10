import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="LLM Model Usage Dashboard", page_icon="📊")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f7f7;
        }
        .stApp {
            background-color: #fafafa;
        }
        .css-1d391kg {
            background-color: #FFFFFF;
        }
        .stSelectbox, .stDataFrame {
            font-size: 14px;
        }
        h1, h2 {
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv('ollama_process_usage.csv')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data[~((data['CPU (%)'] < 10) & (data['Memory (GB)'] < 0.5) & (data['Name'] != 'ollama.exe'))]
    data = data.dropna(subset=['LLM Model'])
    return data

data = load_data()

# Layout structure using columns
col1, col2 = st.columns(2)

with col1:
    llm_models = data['LLM Model'].unique()
    selected_model = st.selectbox('Select LLM Model', llm_models)

    # Dropdown to select the prompt index or another column (you can change 'Prompt Index' to any column you want to filter by)
    prompt_indexes = data['Prompt Index'].unique()  # Assuming 'Prompt Index' is a column
    selected_prompt_index = st.selectbox('Select Prompt Index', prompt_indexes)

# Filter data based on both model and prompt index selection
filtered_data = data[(data['LLM Model'] == selected_model) & (data['Prompt Index'] == selected_prompt_index)]

# Creating a Time column in seconds
filtered_data['Time (s)'] = 0
group = filtered_data.sort_values(by='Timestamp')
start_time = group['Timestamp'].min()
filtered_data['Time (s)'] = (filtered_data['Timestamp'] - start_time).dt.total_seconds()

# Creating results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Plotting section layout
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.subheader('CPU Usage vs. Time')
    fig1, ax = plt.subplots(figsize=(14, 8))
    for pid in filtered_data['PID'].unique():
        pid_data = filtered_data[filtered_data['PID'] == pid]
        ax.plot(pid_data['Time (s)'], pid_data['CPU (%)'], label=f'PID: {pid}', linewidth=2, marker='o')
    ax.set_title(f'CPU Usage vs. Time (LLM Model: {selected_model}, Prompt Index: {selected_prompt_index})', fontsize=18, fontweight='bold', color='#4B4B4B')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('CPU Usage (%)', fontsize=14)
    ax.set_ylim(-10, 600)
    ax.set_facecolor('#fafafa')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    st.pyplot(fig1)

with plot_col2:
    st.subheader('Memory Usage vs. Time')
    fig2, ax = plt.subplots(figsize=(14, 8))
    for pid in filtered_data['PID'].unique():
        pid_data = filtered_data[filtered_data['PID'] == pid]
        ax.plot(pid_data['Time (s)'], pid_data['Memory (GB)'], label=f'PID: {pid}', linewidth=2, marker='x')
    ax.set_title(f'Memory Usage vs. Time (LLM Model: {selected_model}, Prompt Index: {selected_prompt_index})', fontsize=18, fontweight='bold', color='#4B4B4B')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Memory Usage (GB)', fontsize=14)
    ax.set_ylim(-1, 30)
    ax.set_facecolor('#fafafa')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', fontsize=12)
    st.pyplot(fig2)

# Display the dataframe in a styled format
st.markdown(f"### Data Overview for {selected_model} and Prompt Index: {selected_prompt_index}")
st.dataframe(filtered_data.style.set_properties(**{
    'background-color': 'white',
    'color': 'black',
    'border-color': 'lightgray',
    'text-align': 'center'
}))

# Save the plots to the results folder
cpu_plot_path = f'results/cpu_usage_{selected_model}_{selected_prompt_index}.png'
memory_plot_path = f'results/memory_usage_{selected_model}_{selected_prompt_index}.png'

fig1.savefig(cpu_plot_path, dpi=300, bbox_inches='tight')
fig2.savefig(memory_plot_path, dpi=300, bbox_inches='tight')
