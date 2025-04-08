import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv('ollama_process_usage.csv')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data[~((data['CPU (%)'] < 10) & (data['Memory (GB)'] < 0.5))]
    data = data.dropna(subset=['LLM Model'])

    return data

data = load_data()

col1, col2, col3 = st.columns(3)

with col1:
    name_options = data['Name'].unique()
    selected_name = st.selectbox('Select Name', name_options)

with col2:
    llm_models = data['LLM Model'].unique()
    selected_model = st.selectbox('Select LLM Model', llm_models)

with col3:
    pids = data[data['LLM Model'] == selected_model]['PID'].unique()
    selected_pid = st.selectbox('Select PID', pids)

filtered_data = data[(data['LLM Model'] == selected_model) & (data['PID'] == selected_pid) & (data['Name'] == selected_name)]

filtered_data.loc[:, 'Time (s)'] = 0

group = filtered_data.sort_values(by='Timestamp')
start_time = group['Timestamp'].min()

filtered_data.loc[:, 'Time (s)'] = (filtered_data['Timestamp'] - start_time).dt.total_seconds()

if not os.path.exists('results'):
    os.makedirs('results')

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.subheader('CPU Usage vs. Time')
    fig1, ax = plt.subplots(figsize=(14, 8))
    ax.plot(filtered_data['Time (s)'], filtered_data['CPU (%)'], marker='o')
    ax.set_title(f'CPU Usage vs. Time (PID: {selected_pid}, LLM Model: {selected_model})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CPU (%)')
    ax.set_ylim(-10, 600)
    st.pyplot(fig1)

with plot_col2:
    st.subheader('Memory Usage vs. Time')
    fig2, ax = plt.subplots(figsize=(14, 8))
    ax.plot(filtered_data['Time (s)'], filtered_data['Memory (GB)'], marker='o')
    ax.set_title(f'Memory Usage vs. Time (PID: {selected_pid}, LLM Model: {selected_model})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Memory (GB)')
    ax.set_ylim(-1, 30)
    st.pyplot(fig2)

st.dataframe(filtered_data)

cpu_plot_path = f'results/cpu_usage_{selected_pid}_{selected_model}.png'
memory_plot_path = f'results/memory_usage_{selected_pid}_{selected_model}.png'


fig1.savefig(cpu_plot_path)
fig2.savefig(memory_plot_path)
