import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model as keras_load_model
import os
import re

def plot_accuracy(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].legend()
    ax[0].set_title("Loss")
    ax[1].plot(history.history.get("mean_absolute_error", []), label="MAE")
    ax[1].set_title("Mean Absolute Error")
    ax[1].legend()
    st.pyplot(fig)

def load_excel_files(uploaded_files):
    data_frames = []
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Parse time from filenames like 'Data02k0.7.csv' -> 0.7
        time_val = 0.0
        match = re.search(r'k([0-9]+\.?[0-9]*)', file.name)
        if match:
            time_val = float(match.group(1).replace(',', '.'))

        if 'time' not in df.columns:
            df['time'] = time_val

        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)

def preprocess_data(df, input_steps=3, output_steps=1):
    df = df.sort_values(by='time')
    features = ['Points:0', 'Points:1', 'U:0', 'U:1', 'U:2', 'p']
    grouped = df.groupby('time')
    times = sorted(df['time'].unique())
    X, y = [], []
    for i in range(len(times) - input_steps - output_steps + 1):
        x_seq = [grouped.get_group(times[i + j])[features].values for j in range(input_steps)]
        y_seq = [grouped.get_group(times[i + input_steps + j])[features].values for j in range(output_steps)]
        if all(x.shape == x_seq[0].shape for x in x_seq + y_seq):
            x_arr = np.stack(x_seq)
            y_arr = np.stack(y_seq)
            X.append(x_arr.reshape((input_steps, -1)))
            y.append(y_arr.reshape((output_steps, -1)))
    return np.array(X), np.array(y), times

def visualize_flow_field(df, time_step, show_vectors=True, color_param='p'):
    timestep_df = df[df['time'] == time_step] if 'time' in df.columns else df
    x = timestep_df['Points:0']
    y = timestep_df['Points:1']
    u = timestep_df['U:0']
    v = timestep_df['U:1']
    color_data = timestep_df[color_param] if color_param in timestep_df else timestep_df['p']

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.tricontourf(x, y, color_data, cmap='viridis', levels=14)
    plt.colorbar(sc, ax=ax, label=color_param)
    if show_vectors:
        ax.quiver(x, y, u, v, color='white', scale=30)
    ax.set_title(f'Flow Field at time {time_step}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)

def save_model(model, model_name, save_dir='saved_models'):
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, model_name))

def load_model(model_name, save_dir='saved_models'):
    return keras_load_model(os.path.join(save_dir, model_name), compile=False)

def prepare_data(df, simulation_type="Transient", input_steps=3, output_steps=1):
    X, y, _ = preprocess_data(df, input_steps=input_steps, output_steps=output_steps)
    return train_test_split(X, y, test_size=0.2, random_state=42)