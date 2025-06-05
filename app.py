import streamlit as st
st.set_page_config(page_title="CFD Surrogate Model Platform", layout="wide")

st.title("🔬 Multi-Method Surrogate Modeling Platform")

# Sidebar for method selection
method = st.sidebar.radio("Choose Modeling Approach", ["FVM Surrogates", "SPH Surrogates", "DEM Surrogates"])

# Route to different app modules
if method == "FVM Surrogates":
    st.header("🌀 Flow Field Prediction (FVM-based)")
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import load_excel_files, visualize_flow_field, preprocess_data, plot_accuracy, save_model, load_model, prepare_data
    from models import create_cnn_lstm, create_convlstm, create_fno
    
    # PyVista support
    try:
        import pyvista as pv
        from streamlit.components.v1 import html
        pv.set_plot_theme("document")
        # pv.start_xvfb()  # Removed for Windows compatibility
        HAS_PYVISTA = True
    except ImportError:
        HAS_PYVISTA = False
    
    st.sidebar.title("⚙️ Configuration")
    
    st.title("🌪️ CFD-AI Predictive Platform")
    st.markdown("Upload your CFD or experimental data and build surrogate models for fast prediction.")
    
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = []
    if "data_type" not in st.session_state:
        st.session_state.data_type = None
    if "simulation_type" not in st.session_state:
        st.session_state.simulation_type = None
    if "model_type" not in st.session_state:
        st.session_state.model_type = None
    
    simulation_type = st.sidebar.radio("Simulation Type", ["Steady-State", "Transient"])
    st.session_state.simulation_type = simulation_type
    
    if simulation_type == "Transient":
        uploaded_files = st.file_uploader("Upload Excel files for each time step", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
    else:
        uploaded_files = st.file_uploader("Upload a single Excel file", type=["xlsx", "xls", "csv"], accept_multiple_files=False)
    
    if uploaded_files:
        if simulation_type == "Transient":
            df = load_excel_files(uploaded_files)
        else:
            df = load_excel_files([uploaded_files] if uploaded_files else [])
        if df.empty:
            st.error("Uploaded files contain no data.")
        else:
            st.session_state.uploaded_data = df
    else:
        df = pd.DataFrame()
    
    if not df.empty:
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())
    
        all_columns = df.columns.tolist()
        input_cols = st.multiselect("Select input features", options=all_columns)
        output_cols = st.multiselect("Select output target(s)", options=all_columns)
    
        if simulation_type == "Transient":
            time_col = st.selectbox("Select time column", options=all_columns)
        else:
            time_col = None
    
        # includes velocity magnitude
        vis_param = st.selectbox("Select parameter to visualize", options=["|U|", "p", "U:0", "U:1", "U:2"])
        show_vectors = st.checkbox("Overlay Velocity Vectors", value=True)
    
        st.session_state.input_cols = input_cols
        st.session_state.output_cols = output_cols
        st.session_state.time_col = time_col
        input_steps = st.sidebar.number_input("Input time steps", min_value=1, max_value=10, value=3)
        output_steps = st.sidebar.number_input("Output time steps", min_value=1, max_value=5, value=1)
        st.session_state.input_steps = input_steps
        st.session_state.output_steps = output_steps
    
        data_type = st.sidebar.selectbox("Data Format", ["Tabular (FVM-like)", "Grid-based", "Meshless (SPH/DEM)"])
        st.session_state.data_type = data_type
    
        if simulation_type == "Transient" and time_col:
            st.subheader("🔍 Flow Field Visualization")
            available_times = sorted(df[time_col].unique())
            time_selected = st.selectbox("Select time step", options=available_times)
            visualize_flow_field(df, time_selected, show_vectors, color_param=vis_param)
        elif not df.empty:
            visualize_flow_field(df, df['time'].iloc[0] if 'time' in df.columns else None, True)
    
        st.subheader("🧠 Surrogate Model Selection")
        auto_suggested = {
            ("Transient", "Tabular (FVM-like)"): ["CNN-LSTM", "ConvLSTM", "FNO"],
            ("Steady-State", "Tabular (FVM-like)"): ["CNN", "FNO"],
            ("Transient", "Meshless (SPH/DEM)"): ["PointNet", "GraphNet"],
            ("Steady-State", "Meshless (SPH/DEM)"): ["MLP", "GraphNet"]
        }
        suggested_models = auto_suggested.get((simulation_type, data_type), ["CNN-LSTM", "FNO"])
        model_choice = st.sidebar.selectbox("Surrogate Model", suggested_models)
        st.session_state.model_type = model_choice
    
        st.subheader("🎯 Ground Truth Time Selection")
        if simulation_type == "Transient" and time_col:
            available_times = sorted(df[time_col].unique())
            ground_truth_time = st.selectbox("Select ground truth time step for prediction evaluation", options=available_times)
            st.session_state.ground_truth_time = ground_truth_time
    
        num_epochs = st.sidebar.slider("Epochs", 1, 100, 10)
        batch_size = st.sidebar.slider("Batch Size", 4, 128, 32)
    
        if st.button("🚀 Train Model"):
            try:
                X_train, X_test, y_train, y_test = prepare_data(df, simulation_type, input_steps=input_steps, output_steps=output_steps)
                input_shape = X_train.shape[1:]
                output_shape = y_train.shape[1:]
    
                if model_choice == "CNN-LSTM":
                    model = create_cnn_lstm(input_shape, output_shape)
                elif model_choice == "ConvLSTM":
                    model = create_convlstm(input_shape, output_shape)
                elif model_choice == "FNO":
                    model = create_fno(input_shape, output_shape)
                else:
                    st.error("Model not implemented.")
                    st.stop()
    
                history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
                st.success("✅ Model trained successfully!")
                st.subheader("📈 Accuracy Plot")
                st.pyplot(plot_accuracy(history))
    
                save_model(model, f"{model_choice}.h5")
                st.success("Model saved for future use.")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
    
        if st.button("📊 Predict Flow Field Using Saved Model"):
            try:
                model = load_model(f"{model_choice}.h5")
                X_train, X_test, y_train, y_test = prepare_data(df, simulation_type, input_steps=input_steps, output_steps=output_steps)
                predictions = model.predict(X_test)
    
                st.subheader("🔍 Ground Truth vs Prediction")
                idx = 0 if len(predictions) == 1 else st.slider("Select test sample", 0, len(predictions)-1, 0)
    
                selected_time = st.session_state.ground_truth_time
                df_selected = df[df['time'] == selected_time]
                n_points = len(df_selected)
                n_features = 6
    
                if vis_param == "|U|":
                    ground_u = y_test[idx].reshape(-1, n_points, n_features)[0][:, 0:3]
                    pred_u = predictions[idx].reshape(-1, n_points, n_features)[0][:, 0:3]
                    ground = np.linalg.norm(ground_u, axis=1)
                    pred = np.linalg.norm(pred_u, axis=1)
                else:
                    param_index = {"U:0": 0, "U:1": 1, "U:2": 2, "p": 3}.get(vis_param, 3)
                    ground = y_test[idx].reshape(-1, n_points, n_features)[0][:, param_index]
                    pred = predictions[idx].reshape(-1, n_points, n_features)[0][:, param_index]
    
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].tricontourf(df_selected['Points:0'], df_selected['Points:1'], ground, cmap='viridis')
                ax[0].set_title(f"Ground Truth ({vis_param})")
    
                ax[1].tricontourf(df_selected['Points:0'], df_selected['Points:1'], pred, cmap='viridis')
                ax[1].set_title(f"Prediction ({vis_param})")
                st.pyplot(fig)
    
                # PyVista 3D plot (optional)
                if HAS_PYVISTA and st.checkbox("🔄 Show Interactive PyVista Plot"):
                    try:
                        mesh = pv.PolyData(np.column_stack([df_selected["Points:0"], df_selected["Points:1"], np.zeros(n_points)]))
                        mesh.point_data["Ground"] = ground
                        mesh.point_data["Prediction"] = pred
                        plotter = pv.Plotter(window_size=[800, 500], notebook=False)
                        plotter.add_mesh(mesh, scalars="Ground", cmap="viridis", show_scalar_bar=True)
                        plotter.view_xy()
                        plotter.set_background("white")
                        plotter.show(jupyter_backend="none", auto_close=False)
                    except Exception as e:
                        st.warning(f"⚠️ PyVista interactive plotting failed: {e}")
    
                if st.button("📤 Download Ground Truth vs Prediction as CSV"):
                    out_df = pd.DataFrame({
                        "x": df_selected['Points:0'].values,
                        "y": df_selected['Points:1'].values,
                        f"GT_{vis_param}": ground,
                        f"Pred_{vis_param}": pred
                    })
                    out_df.to_csv("comparison_output.csv", index=False)
                    with open("comparison_output.csv", "rb") as f:
                        st.download_button("Download CSV", f, file_name="comparison_output.csv")
    
                if st.button("🖼️ Download Plot as Image"):
                    fig.savefig("comparison_plot.png")
                    with open("comparison_plot.png", "rb") as f:
                        st.download_button("Download Plot", f, file_name="comparison_plot.png")
    
                # Time-lapse animation
                st.subheader("📽️ Time-lapse Flow Field Prediction")
                num_steps = st.slider("Number of time steps to animate", min_value=2, max_value=min(10, len(predictions)), value=3)
    
                for i in range(num_steps):
                    if vis_param == "|U|":
                        ground_u = y_test[i].reshape(-1, n_points, n_features)[0][:, 0:3]
                        pred_u = predictions[i].reshape(-1, n_points, n_features)[0][:, 0:3]
                        ground_frame = np.linalg.norm(ground_u, axis=1)
                        pred_frame = np.linalg.norm(pred_u, axis=1)
                    else:
                        param_index = {"U:0": 0, "U:1": 1, "U:2": 2, "p": 3}.get(vis_param, 3)
                        ground_frame = y_test[i].reshape(-1, n_points, n_features)[0][:, param_index]
                        pred_frame = predictions[i].reshape(-1, n_points, n_features)[0][:, param_index]
    
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                    ax[0].tricontourf(df_selected['Points:0'], df_selected['Points:1'], ground_frame, cmap='viridis')
                    ax[0].set_title(f"GT {vis_param} @ step {i}")
                    ax[1].tricontourf(df_selected['Points:0'], df_selected['Points:1'], pred_frame, cmap='viridis')
                    ax[1].set_title(f"Pred {vis_param} @ step {i}")
                    st.pyplot(fig)
    
                # Side-by-side multi-parameter
                st.subheader("📊 Side-by-side Multi-Parameter Views")
                if st.button("Compare p vs |U|"):
                    param_fields = {"p": 3, "|U|": None}
                    u_fields = [0, 1, 2]
                    for param in param_fields:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        if param == "|U|":
                            u_vec = y_test[idx].reshape(-1, n_points, n_features)[0][:, u_fields]
                            z_vals = np.linalg.norm(u_vec, axis=1)
                        else:
                            param_index = param_fields[param]
                            z_vals = y_test[idx].reshape(-1, n_points, n_features)[0][:, param_index]
                        contour = ax.tricontourf(df_selected['Points:0'], df_selected['Points:1'], z_vals, cmap='plasma')
                        plt.colorbar(contour, ax=ax)
                        ax.set_title(f"{param} Flow Field")
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

elif method == "SPH Surrogates":
    st.header("💧 SPH Surrogates")
    st.info("Coming soon: Surrogate models for Smoothed Particle Hydrodynamics (SPH).")

elif method == "DEM Surrogates":
    st.header("⚙️ DEM Surrogates")
    st.info("Coming soon: Surrogate models for Discrete Element Method (DEM).")