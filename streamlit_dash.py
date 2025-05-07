import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout, Input, MaxPooling1D
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="Taal Lake Water Quality Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-image: linear-gradient(135deg, #ffcce6, #ccf2ff);
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(to bottom, #ffcce6, #ccf2ff);
    }
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #00cccc;
        font-weight: bold !important;
    }
    .st-bq {
        border-left: 3px solid #00cccc;
    }
    .stAlert {
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.7) !important;
    }
    .css-1aumxhk {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        color: #00cccc;
        font-weight: bold;
        border: 2px solid #00cccc;
        background-color: rgba(255, 255, 255, 0.7);
    }
    .stTextInput>div>div>input {
        color: #00cccc;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\AstiAd\Downloads\Water Quality-Elective - Final Dataset.csv")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.results = None

# Main dashboard
st.title("🌊 Taal Lake Water Quality Dashboard")
st.markdown("""
Interactive dashboard for monitoring and predicting water quality parameters in Taal Lake.
""")

# Sidebar controls
with st.sidebar:
    st.header("📊 Dashboard Controls")
    
    # Date range selector
    date_range = st.date_input(
        "Select Date Range",
        value=(pd.to_datetime('2020-01-01'), pd.to_datetime('2023-12-31')),
        min_value=pd.to_datetime('2010-01-01'),
        max_value=pd.to_datetime('2023-12-31')
    )
    
    # Site selector
    sites = st.multiselect(
        "Select Monitoring Sites",
        options=["All"] + sorted(load_data()['Site'].unique()),
        default=["All"]
    )
    
    # Parameter groups
    parameter_groups = {
        "🌡️ Temperature": ['Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature'],
        "🧪 Chemical": ['pH', 'Ammonia', 'Nitrate', 'Phosphate'],
        "💨 Gas": ['Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide', 'Air Temperature']
    }
    selected_group = st.radio(
        "Parameter Group",
        options=list(parameter_groups.keys()),
        index=0
    )
    
    # Weather data selector
    weather_col = st.selectbox(
        "Select Weather Data to Clean",
        options=['Weather Condition', 'Wind Direction'],
        index=0
    )

# Load and preprocess data
with st.spinner("Loading and preprocessing data..."):
    df = load_data()
    
    # Data cleaning
    df = df.dropna()
    df.columns = df.columns.str.strip()
    
    numerical_columns = [
        'Surface temp', 'Middle temp', 'Bottom temp', 'Water Temperature',
        'pH', 'Ammonia', 'Nitrate', 'Phosphate',
        'Dissolved Oxygen', 'Sulfide', 'Carbon Dioxide', 'Air Temperature'
    ]
    
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        non_zero_values = df[df[col] != 0][col]
        col_mean = non_zero_values.mean()
        df[col] = df[col].replace(0, col_mean).fillna(col_mean)
    
    # Weather data cleaning
    df[weather_col] = df[weather_col].astype(str).str.strip()
    non_zero = df[~df[weather_col].isin(['0', '0.0', '0.00'])]
    if not non_zero.empty:
        col_mode = non_zero[weather_col].mode()[0]
        df[weather_col] = df[weather_col].replace(['0', '0.0', '0.00'], col_mode)
    
    df = df.drop_duplicates()
    
    # Date handling
    if all(col in df.columns for col in ['Year', 'Month']):
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
        df = df.drop(['Year', 'Month'], axis=1)
    
    # Normalization
    scaler = MinMaxScaler()
    numerical_cols = [col for col in numerical_columns if col in df.columns]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Filter data
    if "All" not in sites:
        df = df[df['Site'].isin(sites)]

# Dashboard tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🌡️ Time Series", "🔥 Correlations", "📉 Scatter Relationships", "🤖 AI Predictions"])

with tab1:
    st.header("Data Overview")
    
    # Summary statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", len(df))
        st.metric("Monitoring Sites", len(df['Site'].unique()))
    with col2:
        st.metric("Parameters Tracked", len(numerical_cols))
        st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Interactive data explorer
    st.subheader("Data Explorer")
    st.dataframe(df.head(100), use_container_width=True)

with tab2:
    st.header("Time Series Analysis")
    
    # Parameter selection
    selected_params = st.multiselect(
        "Select Parameters",
        options=parameter_groups[selected_group],
        default=parameter_groups[selected_group][:2]
    )
    
    if selected_params:
        # Create interactive plot
        melted_df = df.melt(
            id_vars=['Date', 'Site'], 
            value_vars=selected_params,
            var_name='Parameter',  # This creates the 'Parameter' column
            value_name='Value'
        )
        fig = px.line(
            melted_df,
        x='Date',
        y='Value',
        color='Site',
        facet_col='Parameter' if len(selected_params) > 1 else None,
        facet_col_wrap=2,
        height=600,
        title=f"Time Series of {selected_group} Parameters",
        template='plotly_white'
        )

        fig.update_xaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one parameter")

with tab3:
    st.header("Correlation Analysis")
    
    # Heatmap controls
    col1, col2 = st.columns(2)
    with col1:
        heatmap_type = st.radio(
            "View Mode",
            options=["All Sites", "By Site"],
            horizontal=True
        )
    with col2:
        if heatmap_type == "By Site":
            selected_site = st.selectbox(
                "Select Site",
                options=df['Site'].unique()
            )
    
    # Generate heatmap
    if heatmap_type == "By Site":
        heatmap_data = df[df['Site'] == selected_site][numerical_cols]
        title = f"Correlation Heatmap for {selected_site}"
    else:
        heatmap_data = df[numerical_cols]
        title = "Correlation Heatmap (All Sites)"
    
    fig = px.imshow(
        heatmap_data.corr(),
        text_auto=".2f",
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        title=title,
        width=1000,
        height=800,
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Scatter Plot: Weather & Wind vs Water Quality Parameters")

    # Define selectable relationships
    scatter_options = {
        "Weather Condition vs Nitrate": ("Weather Condition", "Nitrate"),
        "Weather Condition vs Nitrite": ("Weather Condition", "Nitrite"),
        "Weather Condition vs Ammonia": ("Weather Condition", "Ammonia"),
        "Weather Condition vs Phosphate": ("Weather Condition", "Phosphate"),
        "Wind Direction vs Ammonia": ("Wind Direction", "Ammonia"),
        "Wind Direction vs Nitrate": ("Wind Direction", "Nitrate"),
        "Wind Direction vs Phosphate": ("Wind Direction", "Phosphate"),
        "Wind Direction vs Nitrite": ("Wind Direction", "Nitrite"),
        "Wind Direction vs Dissolved Oxygen": ("Wind Direction", "Dissolved Oxygen"),
    }

    # Select relationship
    selected_relation = st.selectbox("Select parameter relationship to plot", list(scatter_options.keys()))

    x_col, y_col = scatter_options[selected_relation]

    if x_col in df.columns and y_col in df.columns:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Site",
            title=f"{x_col} vs {y_col}",
            labels={x_col: x_col, y_col: y_col},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Cannot plot: {x_col} or {y_col} not found in dataset.")

with tab5:
    st.header("AI-Powered Water Quality Predictions")
    
    # Model configuration
    with st.expander("⚙️ Model Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox(
                "Target Parameter",
                options=numerical_cols,
                index=numerical_cols.index('pH')
            )
            n_steps = st.slider(
                "Sequence Length",
                min_value=1,
                max_value=10,
                value=3
            )
        with col2:
            epochs = st.slider(
                "Training Epochs",
                min_value=10,
                max_value=200,
                value=50
            )
            batch_size = st.slider(
                "Batch Size",
                min_value=16,
                max_value=128,
                value=32
            )
    
    if st.button("🚀 Train Prediction Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes"):
            # Prepare data
            target_idx = numerical_cols.index(target)
            data_array = df[numerical_cols].values
            
            def create_sequences(data, target_idx, n_steps=3):
                X, y = [], []
                for i in range(len(data)-n_steps):
                    X.append(data[i:(i+n_steps), :])
                    y.append(data[i+n_steps, target_idx])
                return np.array(X), np.array(y)
                
            X, y = create_sequences(data_array, target_idx, n_steps)
            
            # Split data
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Model definitions
            def create_cnn_model():
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    Conv1D(32, 2, activation='relu'),
                    MaxPooling1D(1),
                    Flatten(),
                    Dense(50, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mae')
                return model
                
            def create_lstm_model():
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    LSTM(50, activation='tanh'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mae')
                return model
                
            def create_hybrid_model():
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    Conv1D(32, 2, activation='relu'),
                    MaxPooling1D(1),
                    LSTM(50, return_sequences=False),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mae')
                return model
                
            # Train models
            models = {
                'CNN': create_cnn_model(),
                'LSTM': create_lstm_model(),
                'Hybrid CNN-LSTM': create_hybrid_model()
            }
            
            results = {}
            for name, model in models.items():
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    verbose=0
                )
                
                y_pred = model.predict(X_test).flatten()
                
                results[name] = {
                    'model': model,
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'history': history
                }
            
            st.session_state.results = results
            st.session_state.model_trained = True
            st.success("Model training completed!")
    
    if st.session_state.model_trained:
        # Display results
        st.subheader("Model Performance")
        
        # Metrics cards
        cols = st.columns(3)
        for i, (name, res) in enumerate(st.session_state.results.items()):
            with cols[i]:
                st.metric(
                    label=f"{name} Model",
                    value=f"MAE: {res['mae']:.4f}",
                    help=f"RMSE: {res['rmse']:.4f}"
                )
        
        # Training curves
        st.subheader("Training Progress")
        fig = go.Figure()
        for name, res in st.session_state.results.items():
            fig.add_trace(go.Scatter(
                x=list(range(epochs)),
                y=res['history'].history['loss'],
                name=f"{name} Training",
                line=dict(width=2)
            ))
            fig.add_trace(go.Scatter(
                x=list(range(epochs)),
                y=res['history'].history['val_loss'],
                name=f"{name} Validation",
                line=dict(width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss (MAE)",
            hovermode="x unified",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Water Quality Assessment
        st.subheader("Water Quality Index Assessment")
        
        # Use LSTM model for predictions
        y_pred_lstm = st.session_state.results['LSTM']['model'].predict(X_test).flatten()
        
        def calculate_wqi(preds, feature_names, sequences):
            denorm = scaler.inverse_transform(sequences.reshape(-1,len(feature_names))).reshape(sequences.shape)
            for i,p in enumerate(preds): 
                denorm[i,-1,feature_names.index(target)] = p*(scaler.data_max_[feature_names.index(target)]-scaler.data_min_[feature_names.index(target)])+scaler.data_min_[feature_names.index(target)]
            wqi, violations = [], []
            for seq in denorm:
                val = seq[-1]
                score=100; vio=[]
                if not 6.5<=val[feature_names.index('pH')]<=8.5: score-=25; vio.append('pH')
                if val[feature_names.index('Dissolved Oxygen')]<5: score-=20; vio.append('Low Oxygen')
                if val[feature_names.index('Ammonia')]>1: score-=15; vio.append('High Ammonia')
                if val[feature_names.index('Phosphate')]>0.4: score-=10; vio.append('High Phosphate')
                wqi.append(max(score,0)); violations.append(vio)
            return np.array(wqi), violations
        
        wqi_vals, vio_list = calculate_wqi(
            y_pred_lstm,
            numerical_cols,
            X_test
        )
        
        # WQI Distribution
        wqi_bins = {
            'Excellent (90-100)': (90, 100),
            'Good (70-89)': (70, 89),
            'Fair (50-69)': (50, 69),
            'Poor (<50)': (0, 49)
        }
        
        wqi_counts = {cat: 0 for cat in wqi_bins}
        for val in wqi_vals:
            for cat, (low, high) in wqi_bins.items():
                if low <= val <= high:
                    wqi_counts[cat] += 1
                    break
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                names=list(wqi_counts.keys()),
                values=list(wqi_counts.values()),
                title="WQI Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Water Quality Categories")
            for cat, count in wqi_counts.items():
                st.progress(
                    count/len(wqi_vals),
                    text=f"{cat}: {count} samples ({count/len(wqi_vals):.1%})"
                )
                
        st.subheader("📌 Actual vs Predicted Values")
        # Compute predictions again (optional, or reuse stored ones)
        scatter_cols = st.columns(3)
        for i, (name, res) in enumerate(st.session_state.results.items()):
                model = res['model']
                y_pred = model.predict(X_test).flatten()
              
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    marker=dict(color='skyblue', size=6, opacity=0.6),
                    name='Predicted vs Actual'
                ))
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_test,
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='Ideal Line (y=x)'
                ))
                fig.update_layout(
                    title=f"{name}: Actual vs Predicted ({target})",
                    xaxis_title="Actual",
                    yaxis_title="Predicted",
                    width=400,
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    template="plotly_white"
                )
                with scatter_cols[i]:
                    st.plotly_chart(fig, use_container_width=True)
              
        # Recommendations
        st.subheader("Recommendations")
        
        mapping = {
            'pH': 'Adjust pH levels through aeration or chemical treatment',
            'Low Oxygen': 'Increase oxygenation through waterfall aeration or surface agitators',
            'High Ammonia': 'Implement biological filtration or water exchange',
            'High Phosphate': 'Use phosphate-removing media or limit nutrient runoff'
        }
        
        bad_vios = {
            v
            for val, vlist in zip(wqi_vals, vio_list)
            if val < 70
            for v in vlist
        }
        
        if bad_vios:
            st.warning("⚠️ Water quality issues detected:")
            for v in sorted(bad_vios):
                with st.expander(f"**{v}** - Recommended Action"):
                    st.info(mapping[v])
        else:
            st.success("✅ No significant water quality issues detected")


# Footer
st.markdown("---")
st.markdown("""
**Taal Lake Water Quality Dashboard**  
Developed for environmental monitoring and predictive analysis  
Groupings:
Antivo, Bongalon, Capacia, Diaz, Maghirang 
""")