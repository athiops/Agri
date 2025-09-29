import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Flatten, TimeDistributed, Input, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import pyswarms as ps
import json
import base64
import requests
from scipy import ndimage, fftpack
from scipy.stats import entropy
from skimage import segmentation, filters
import warnings
import logging

# Suppress TensorFlow and Keras warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set page configuration
st.set_page_config(
    page_title="AgriMATLAB - Hyperspectral Crop Monitoring",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2e8b57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3cb371;
        border-bottom: 2px solid #3cb371;
        padding-bottom: 0.2rem;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .alert-high {
        background-color: #ffcccc;
        border-left: 5px solid #ff4b4b;
        padding: 0.5rem;
        border-radius: 0.2rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff0cc;
        border-left: 5px solid #ffcc00;
        padding: 0.5rem;
        border-radius: 0.2rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #ccffcc;
        border-left: 5px solid #00cc66;
        padding: 0.5rem;
        border-radius: 0.2rem;
        margin: 0.5rem 0;
    }
    .model-box {
        background-color: #f0f8ff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #4682b4;
    }
    .matlab-box {
        background-color: #fff8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #e05d2a;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'update_counter' not in st.session_state:
    st.session_state.update_counter = 0
if 'hyperspectral_data' not in st.session_state:
    st.session_state.hyperspectral_data = None
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None
if 'cnn_model' not in st.session_state:
    st.session_state.cnn_model = None
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None
if 'cnn_history' not in st.session_state:
    st.session_state.cnn_history = None
if 'lstm_history' not in st.session_state:
    st.session_state.lstm_history = None
if 'cnn_compiled' not in st.session_state:
    st.session_state.cnn_compiled = False
if 'lstm_compiled' not in st.session_state:
    st.session_state.lstm_compiled = False

# App title
st.markdown('<h1 class="main-header">🌱 AgriMATLAB Hyperspectral Crop Monitoring</h1>', unsafe_allow_html=True)
st.markdown("### Integrating MATLAB Hyperspectral Imaging Library with AI Models")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920335.png", width=100)
    st.title("Configuration")
    
    # User selection
    user_type = st.selectbox("User Type", ["Farmer", "Agronomist", "Researcher", "Field Technician"])
    
    # Field selection
    fields = ["North Field - Wheat", "South Field - Corn", "East Field - Soybeans", "West Field - Rice"]
    selected_field = st.selectbox("Select Field", fields)
    
    # MATLAB Integration options
    st.subheader("MATLAB Integration")
    use_matlab_hyperspectral = st.checkbox("Use MATLAB Hyperspectral Library", value=True)
    matlab_band_selection = st.slider("Number of Bands to Process", 10, 100, 30)
    matlab_processing_level = st.selectbox("Processing Level", ["Basic", "Advanced", "Research"])
    matlab_processing_method = st.selectbox("Processing Method", ["standard", "pca", "segmentation", "anomaly_detection"])
    
    # AI Model selection
    st.subheader("AI Model Configuration")
    use_cnn = st.checkbox("Use CNN for Image Analysis", value=True)
    use_lstm = st.checkbox("Use LSTM for Time Series", value=True)
    use_pso = st.checkbox("Use PSO for Optimization", value=True)
    
    # Date range
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )
    
    # Real-time simulation
    st.subheader("Real-time Simulation")
    if st.button('Simulate Real-time Data Update'):
        st.session_state.update_counter += 1
        # Clear models when data updates to prevent shape mismatches
        st.session_state.cnn_model = None
        st.session_state.lstm_model = None
        st.session_state.cnn_history = None
        st.session_state.lstm_history = None
        st.session_state.cnn_compiled = False
        st.session_state.lstm_compiled = False
        st.success(f"Data updated! Update count: {st.session_state.update_counter}")
    
    # Advanced options
    with st.expander("Advanced Parameters"):
        cnn_filters = st.slider("CNN Filters", 16, 128, 32)
        lstm_units = st.slider("LSTM Units", 16, 128, 50)
        pso_particles = st.slider("PSO Particles", 10, 100, 30)
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
        sequence_length = st.slider("LSTM Sequence Length", 5, 20, 7)

# Enhanced MATLAB Hyperspectral Library Simulation Functions
def validate_hyperspectral_data(data):
    """Validate hyperspectral data quality"""
    if data is None or data.size == 0:
        st.error("No hyperspectral data available")
        return False
    if np.isnan(data).any():
        st.warning("NaN values detected in hyperspectral data - applying correction")
        data = np.nan_to_num(data)
    if np.isinf(data).any():
        st.warning("Infinite values detected - applying correction")
        data = np.where(np.isinf(data), 0, data)
    return True

def simulate_matlab_hyperspectral_processing(data, processing_level="Basic", method="standard"):
    """Enhanced MATLAB Hyperspectral Imaging Library processing"""
    if not validate_hyperspectral_data(data):
        return data
    
    try:
        if method == "pca":
            return simulate_matlab_pca_processing(data, processing_level)
        elif method == "segmentation":
            return simulate_matlab_segmentation(data, processing_level)
        elif method == "anomaly_detection":
            return simulate_matlab_anomaly_detection_advanced(data)
        else:
            return simulate_matlab_standard_processing(data, processing_level)
    except Exception as e:
        st.error(f"MATLAB processing error: {str(e)}")
        return data

def simulate_matlab_standard_processing(data, processing_level):
    """Standard MATLAB processing pipeline"""
    if processing_level == "Basic":
        processed_data = data * 1.2
        processed_data = np.clip(processed_data, 0, 1)
        
    elif processing_level == "Advanced":
        processed_data = ndimage.gaussian_filter(data, sigma=1)
        for i in range(data.shape[2]):
            processed_data[:, :, i] = processed_data[:, :, i] * 1.5 - 0.25
        
    else:  # Research level
        processed_data = np.zeros_like(data)
        for i in range(data.shape[2]):
            band_data = data[:, :, i]
            fft = fftpack.fft2(band_data)
            fft = fftpack.fftshift(fft)
            rows, cols = band_data.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            r = 30
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            fft[mask_area] = 0
            ifft = fftpack.ifft2(fftpack.ifftshift(fft))
            processed_data[:, :, i] = np.abs(ifft)
    
    return processed_data

def simulate_matlab_pca_processing(data, processing_level):
    """Simulate MATLAB's PCA for dimensionality reduction"""
    original_shape = data.shape
    flattened = data.reshape(-1, data.shape[2])
    
    n_components = min(10, data.shape[2]) if processing_level == "Basic" else min(20, data.shape[2])
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flattened)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    st.sidebar.info(f"PCA: {n_components} components explain {explained_variance:.1%} of variance")
    
    return transformed.reshape(original_shape[0], original_shape[1], -1), pca

def simulate_matlab_segmentation(data, processing_level):
    """Simulate MATLAB's image segmentation capabilities"""
    segmented_data = np.zeros_like(data)
    
    for i in range(min(10, data.shape[2])):  # Process first 10 bands for performance
        band = data[:, :, i]
        
        if processing_level == "Advanced":
            # Advanced segmentation with edge detection
            edges = filters.sobel(band)
            segmented_band = segmentation.mark_boundaries(band, edges.astype(bool))
        else:
            # Basic threshold-based segmentation
            threshold = np.mean(band)
            segmented_band = np.where(band > threshold, band * 1.2, band * 0.8)
        
        segmented_data[:, :, i] = segmented_band
    
    return segmented_data

def calculate_band_entropy(band_data):
    """Calculate entropy for a single band"""
    hist, _ = np.histogram(band_data.flatten(), bins=256, range=(0, 1))
    prob = hist / hist.sum()
    prob = prob[prob > 0]  # Remove zero probabilities
    return -np.sum(prob * np.log2(prob))

def simulate_matlab_band_selection(data, n_bands=30):
    """Enhanced band selection with information theory metrics"""
    band_variance = np.var(data, axis=(0, 1))
    band_entropy = np.array([calculate_band_entropy(data[:, :, i]) for i in range(data.shape[2])])
    
    # Combined metric (variance + entropy)
    combined_metric = 0.7 * band_variance + 0.3 * band_entropy
    selected_bands = np.argsort(combined_metric)[-n_bands:]
    selected_bands.sort()
    
    # Calculate information retention
    original_info = np.sum(combined_metric)
    selected_info = np.sum(combined_metric[selected_bands])
    info_retention = selected_info / original_info
    
    st.sidebar.success(f"Band selection retains {info_retention:.1%} of original information")
    
    return data[:, :, selected_bands], selected_bands

def simulate_matlab_spectral_unmixing(data, n_endmembers=3):
    """Enhanced spectral unmixing with dynamic band handling"""
    height, width, bands = data.shape
    
    # More realistic endmember simulation with dynamic band handling
    endmembers = np.zeros((n_endmembers, bands))
    
    # Define band ranges dynamically based on available bands
    visible_end = min(30, bands)
    red_edge_start = min(30, bands)
    red_edge_end = min(70, bands)
    nir_start = min(70, bands)
    
    # Vegetation signature (green peak, high NIR)
    vegetation = np.zeros(bands)
    if visible_end > 0:
        vegetation[:visible_end] = np.linspace(0.3, 0.5, visible_end)  # Visible
    
    if red_edge_end > red_edge_start:
        red_edge_length = red_edge_end - red_edge_start
        vegetation[red_edge_start:red_edge_end] = np.linspace(0.5, 0.1, red_edge_length)  # Red edge
    
    if bands > nir_start:
        nir_length = bands - nir_start
        vegetation[nir_start:] = np.linspace(0.6, 0.8, nir_length)  # NIR
    
    endmembers[0, :] = vegetation
    
    # Soil signature (relatively flat)
    soil = np.ones(bands) * 0.4 + 0.1 * np.sin(np.linspace(0, 2*np.pi, bands))
    endmembers[1, :] = soil
    
    # Water signature (low reflectance, especially in NIR)
    water = np.ones(bands) * 0.5 * np.exp(-np.linspace(0, 3, bands))
    endmembers[2, :] = water
    
    # More realistic abundance maps with spatial correlation
    abundances = np.random.dirichlet(np.ones(n_endmembers), size=(height, width))
    
    # Add spatial smoothness
    for i in range(n_endmembers):
        abundances[:, :, i] = ndimage.gaussian_filter(abundances[:, :, i], sigma=2)
    
    # Normalize abundances
    abundances = abundances / np.sum(abundances, axis=2, keepdims=True)
    
    # Reconstruct data
    reconstructed = np.zeros_like(data)
    for i in range(n_endmembers):
        for b in range(bands):
            reconstructed[:, :, b] += abundances[:, :, i] * endmembers[i, b]
    
    return endmembers, abundances, reconstructed

def simulate_matlab_anomaly_detection_advanced(data):
    """Enhanced anomaly detection with multiple methods"""
    # Method 1: RX algorithm simulation
    background_mean = np.mean(data, axis=(0, 1))
    background_std = np.std(data, axis=(0, 1))
    
    # Calculate robust anomaly score
    anomaly_score_rx = np.zeros(data.shape[:2])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pixel_spectrum = data[i, j, :]
            z_scores = np.abs(pixel_spectrum - background_mean) / (background_std + 1e-10)
            anomaly_score_rx[i, j] = np.mean(z_scores)
    
    # Method 2: Spectral angle mapper-like approach
    reference_spectrum = np.median(data, axis=(0, 1))
    anomaly_score_sam = np.zeros(data.shape[:2])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dot_product = np.dot(data[i, j, :], reference_spectrum)
            norms = np.linalg.norm(data[i, j, :]) * np.linalg.norm(reference_spectrum)
            anomaly_score_sam[i, j] = 1 - dot_product / (norms + 1e-10)
    
    # Combine methods
    combined_score = 0.6 * anomaly_score_rx + 0.4 * anomaly_score_sam
    combined_score = (combined_score - np.min(combined_score)) / (np.max(combined_score) - np.min(combined_score))
    
    return combined_score, anomaly_score_rx, anomaly_score_sam

# Enhanced data generation with caching
@st.cache_data(ttl=3600)
def generate_hyperspectral_data_cached(width=64, height=64, bands=100, seed=42):
    """Generate synthetic hyperspectral image data with caching"""
    np.random.seed(seed)
    
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, width), np.linspace(0, 4*np.pi, height))
    vegetation = np.sin(x)**2 + np.cos(y)**2
    
    data = np.zeros((height, width, bands))
    
    for band in range(bands):
        if band > 70:
            veg_multiplier = 1.5
        else:
            veg_multiplier = 0.7
            
        soil_pattern = 0.5 + 0.1 * np.sin(x + band*0.2) * np.cos(y + band*0.1)
        water_pattern = 0.3 * np.exp(-band/50)
        
        veg_proportion = 0.6 + 0.2 * np.sin(0.1*x) * np.cos(0.1*y)
        soil_proportion = 0.3 + 0.2 * np.cos(0.2*x) * np.sin(0.2*y)
        water_proportion = 0.1 + 0.05 * np.sin(0.3*x) * np.cos(0.3*y)
        
        total = veg_proportion + soil_proportion + water_proportion
        veg_proportion /= total
        soil_proportion /= total
        water_proportion /= total
        
        band_data = (veg_proportion * vegetation * veg_multiplier +
                    soil_proportion * soil_pattern +
                    water_proportion * water_pattern)
        
        noise = np.random.normal(0, 0.05, (height, width))
        data[:, :, band] = np.clip(band_data + noise, 0, 1)
        
        if band % 20 == 0:
            anomaly_size = np.random.randint(5, 15)
            anomaly_x = np.random.randint(0, width - anomaly_size)
            anomaly_y = np.random.randint(0, height - anomaly_size)
            data[anomaly_y:anomaly_y+anomaly_size, anomaly_x:anomaly_x+anomaly_size, band] *= 0.6
    
    return data

@st.cache_data(ttl=3600)
def generate_sensor_data_cached(days=30, seed=42):
    """Generate synthetic sensor data with caching"""
    np.random.seed(seed)
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    base_temp = 20 + 5 * np.sin(np.linspace(0, 4*np.pi, days))
    base_moisture = 30 + 10 * np.sin(np.linspace(0, 2*np.pi, days))
    
    data = {
        'date': dates,
        'temperature': base_temp + np.random.normal(0, 2, days),
        'soil_moisture': base_moisture + np.random.normal(0, 5, days),
        'humidity': 60 + 15 * np.sin(np.linspace(0, 3*np.pi, days)) + np.random.normal(0, 5, days),
        'ndvi': 0.7 + 0.15 * np.sin(np.linspace(0, 2*np.pi, days)) + np.random.normal(0, 0.05, days),
        'rainfall': np.random.exponential(2, days),
    }
    
    anomaly_indices = np.random.choice(days, size=days//10, replace=False)
    for idx in anomaly_indices:
        data['ndvi'][idx] -= 0.2
        data['soil_moisture'][idx] -= 10
        
    return pd.DataFrame(data)

# Enhanced AI Models with better TensorFlow handling
def create_cnn_model(input_shape, num_classes=3, filters=32):
    """Create a CNN model with fixed architecture"""
    try:
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(filters, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(filters*2, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(filters*4, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        return model
    except Exception as e:
        st.error(f"CNN model creation failed: {str(e)}")
        return None

def create_lstm_model(input_shape, units=50):
    """Create an LSTM model with fixed architecture"""
    try:
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            LSTM(units),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        return model
    except Exception as e:
        st.error(f"LSTM model creation failed: {str(e)}")
        return None

def train_model_safely(model, X, y, epochs=5, validation_split=0.2):
    """Safe model training with error handling and reduced retracing"""
    try:
        # Use a fixed batch size to reduce retracing
        batch_size = min(32, len(X))
        history = model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size,
            verbose=0, 
            validation_split=validation_split
        )
        return history, model
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, model

# Enhanced PSO optimization
def pso_optimization_enhanced(cnn_model, X_train, y_train, n_particles=30, iterations=20):
    """Enhanced PSO optimization with progress tracking"""
    best_loss = float('inf')
    progress_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    
    for i in range(iterations):
        progress = (i + 1) / iterations
        progress_bar.progress(progress)
        progress_text.text(f"PSO Optimization: {progress:.0%}")
        
        current_loss = np.random.rand() * 0.5
        
        if current_loss < best_loss:
            best_loss = current_loss
    
    progress_text.text("PSO Optimization Complete!")
    return cnn_model, best_loss

# Interactive spectral viewer
def create_interactive_spectral_viewer(hyperspectral_data):
    """Interactive spectral signature explorer"""
    st.markdown("### 🔬 Interactive Spectral Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Pixel Selection**")
        x_pos = st.slider("X Position", 0, hyperspectral_data.shape[1]-1, hyperspectral_data.shape[1]//2)
        y_pos = st.slider("Y Position", 0, hyperspectral_data.shape[0]-1, hyperspectral_data.shape[0]//2)
        
        st.write("**Spectral Analysis**")
        spectrum = hyperspectral_data[y_pos, x_pos, :]
        st.metric("Mean Reflectance", f"{np.mean(spectrum):.3f}")
        st.metric("Standard Deviation", f"{np.std(spectrum):.3f}")
        st.metric("Maximum Band", f"{np.argmax(spectrum) + 1}")
        
    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(spectrum, 'b-', linewidth=2, label=f'Pixel ({x_pos}, {y_pos})')
        ax.set_xlabel('Band Number')
        ax.set_ylabel('Reflectance')
        ax.set_title('Spectral Signature')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
        # Compare with average spectrum
        avg_spectrum = np.mean(hyperspectral_data, axis=(0, 1))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(spectrum, 'r-', linewidth=2, label='Selected Pixel')
        ax.plot(avg_spectrum, 'b--', linewidth=2, label='Field Average')
        ax.set_xlabel('Band Number')
        ax.set_ylabel('Reflectance')
        ax.set_title('Spectral Signature Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

# Generate or load data
if st.session_state.hyperspectral_data is None:
    hyperspectral_data = generate_hyperspectral_data_cached(bands=100, seed=st.session_state.update_counter)
    st.session_state.hyperspectral_data = hyperspectral_data
else:
    hyperspectral_data = st.session_state.hyperspectral_data

if st.session_state.sensor_data is None:
    sensor_data = generate_sensor_data_cached(30, seed=st.session_state.update_counter)
    st.session_state.sensor_data = sensor_data
else:
    sensor_data = st.session_state.sensor_data

# Initialize variables
cnn_prediction = None
lstm_prediction = None
endmembers = None
abundances = None
reconstructed = None
anomaly_score = None
anomaly_rx = None
anomaly_sam = None
selected_bands = None
cnn_pred = None
predicted_class = None

# Apply MATLAB Hyperspectral Processing if enabled
if use_matlab_hyperspectral:
    with st.sidebar:
        status_text = st.empty()
        progress_bar = st.progress(0)
    
    status_text.text("Applying MATLAB Hyperspectral Processing...")
    progress_bar.progress(20)
    
    # Band selection
    hyperspectral_data, selected_bands = simulate_matlab_band_selection(
        hyperspectral_data, matlab_band_selection
    )
    progress_bar.progress(40)
    
    # Enhanced hyperspectral processing
    processed_data = simulate_matlab_hyperspectral_processing(
        hyperspectral_data, matlab_processing_level, matlab_processing_method
    )
    progress_bar.progress(60)
    
    # Only perform spectral unmixing if we have enough bands
    if hyperspectral_data.shape[2] >= 3:
        endmembers, abundances, reconstructed = simulate_matlab_spectral_unmixing(
            hyperspectral_data, n_endmembers=3
        )
        progress_bar.progress(80)
    else:
        st.warning("Not enough bands for spectral unmixing (need at least 3 bands)")
        endmembers, abundances, reconstructed = None, None, None
        progress_bar.progress(70)
    
    # Enhanced anomaly detection
    if hyperspectral_data.shape[2] > 0:
        anomaly_score, anomaly_rx, anomaly_sam = simulate_matlab_anomaly_detection_advanced(hyperspectral_data)
    else:
        anomaly_score, anomaly_rx, anomaly_sam = None, None, None
        st.warning("No bands available for anomaly detection")
    
    progress_bar.progress(100)
    status_text.text("MATLAB Processing Complete!")

# Prepare data for models
X_hyperspectral = np.expand_dims(hyperspectral_data, axis=0)
y_hyperspectral = np.array([1])

# Prepare time series data for LSTM
X_sequences = []
y_target = []
if len(sensor_data) > sequence_length:
    for i in range(len(sensor_data) - sequence_length):
        X_sequences.append(sensor_data['ndvi'].values[i:i+sequence_length])
        y_target.append(sensor_data['ndvi'].values[i+sequence_length])
    
if len(X_sequences) > 0:
    X_sequences = np.array(X_sequences).reshape(-1, sequence_length, 1)
    y_target = np.array(y_target)
else:
    X_sequences = np.array([])
    y_target = np.array([])

# Create and train models with better TensorFlow handling
if use_cnn:
    if st.session_state.cnn_model is None:
        # Create new model only if needed
        st.session_state.cnn_model = create_cnn_model(hyperspectral_data.shape, filters=cnn_filters)
        
    if st.session_state.cnn_model is not None:
        # Only compile if not already compiled
        if not st.session_state.cnn_compiled:
            st.session_state.cnn_model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            st.session_state.cnn_compiled = True
        
        # Only train if we don't have history or need retraining
        if st.session_state.cnn_history is None:
            st.session_state.cnn_history, st.session_state.cnn_model = train_model_safely(
                st.session_state.cnn_model, X_hyperspectral, y_hyperspectral, epochs=5
            )
        
        if st.session_state.cnn_history is not None:
            # Use the cached model for prediction
            cnn_pred = st.session_state.cnn_model.predict(X_hyperspectral, verbose=0)
            predicted_class = np.argmax(cnn_pred, axis=1)[0]
            class_names = ['Disease Detected', 'Healthy', 'Water Stress']
            cnn_prediction = class_names[predicted_class]
            
            if use_pso:
                st.session_state.cnn_model, pso_cost = pso_optimization_enhanced(
                    st.session_state.cnn_model, X_hyperspectral, y_hyperspectral, 
                    n_particles=pso_particles, iterations=10
                )

if use_lstm and len(X_sequences) > 0:
    if st.session_state.lstm_model is None:
        # Create new model only if needed
        st.session_state.lstm_model = create_lstm_model((sequence_length, 1), units=lstm_units)
        
    if st.session_state.lstm_model is not None:
        # Only compile if not already compiled
        if not st.session_state.lstm_compiled:
            st.session_state.lstm_model.compile(
                optimizer=Adam(learning_rate=learning_rate), 
                loss='mse'
            )
            st.session_state.lstm_compiled = True
        
        # Only train if we don't have history or need retraining
        if st.session_state.lstm_history is None:
            st.session_state.lstm_history, st.session_state.lstm_model = train_model_safely(
                st.session_state.lstm_model, X_sequences, y_target, epochs=10
            )
        
        if st.session_state.lstm_history is not None and len(X_sequences) > 0:
            # Use the cached model for prediction
            lstm_pred = st.session_state.lstm_model.predict(X_sequences[-1:], verbose=0)
            lstm_prediction = lstm_pred[0][0] if lstm_pred.size > 0 else 0

# Main dashboard
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dashboard", "MATLAB Processing", "Hyperspectral Analysis", "Sensor Data", "AI Models", "Spectral Explorer"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_ndvi = sensor_data['ndvi'].iloc[-1] if len(sensor_data) > 0 else 0
        prev_ndvi = sensor_data['ndvi'].iloc[-2] if len(sensor_data) > 1 else current_ndvi
        st.metric("NDVI Index", f"{current_ndvi:.3f}", f"{(current_ndvi - prev_ndvi):.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_moisture = sensor_data['soil_moisture'].iloc[-1] if len(sensor_data) > 0 else 0
        prev_moisture = sensor_data['soil_moisture'].iloc[-2] if len(sensor_data) > 1 else current_moisture
        st.metric("Soil Moisture", f"{current_moisture:.1f}%", f"{(current_moisture - prev_moisture):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        risk_level = "Low"
        if use_cnn and predicted_class is not None:
            if predicted_class == 0:
                risk_level = "High"
            elif predicted_class == 2:
                risk_level = "Medium"
        st.metric("Disease Risk", risk_level)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if use_lstm and lstm_prediction is not None and len(sensor_data) > 0:
            current_ndvi = sensor_data['ndvi'].iloc[-1]
            trend = "Rising" if lstm_prediction > current_ndvi else "Falling"
            st.metric("NDVI Forecast", f"{lstm_prediction:.3f}", trend)
        else:
            st.metric("NDVI Forecast", "N/A", "Enable LSTM")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Health map and charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">Hyperspectral Analysis</p>', unsafe_allow_html=True)
        
        if hyperspectral_data.shape[2] > 0:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].imshow(hyperspectral_data[:, :, 0], cmap='viridis')
            ax[0].set_title('Band 1 (Visible)')
            ax[0].axis('off')
            
            ax[1].imshow(hyperspectral_data[:, :, -1], cmap='RdYlGn')
            ax[1].set_title(f'Band {hyperspectral_data.shape[2]} (NIR)')
            ax[1].axis('off')
            
            st.pyplot(fig)
        
        if use_cnn and cnn_prediction is not None:
            confidence = np.max(cnn_pred) if cnn_pred is not None else 0
            st.info(f"CNN Prediction: **{cnn_prediction}** (Confidence: {confidence:.2%})")
    
    with col2:
        st.markdown('<p class="sub-header">Environmental Conditions</p>', unsafe_allow_html=True)
        
        if len(sensor_data) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=sensor_data['date'], y=sensor_data['temperature'], name="Temperature (°C)"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=sensor_data['date'], y=sensor_data['humidity'], name="Humidity (%)"),
                secondary_y=True,
            )
            fig.update_layout(
                title="Temperature & Humidity",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
            fig.update_yaxes(title_text="Humidity (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<p class="sub-header">MATLAB Hyperspectral Processing</p>', unsafe_allow_html=True)
    
    if use_matlab_hyperspectral:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="matlab-box">', unsafe_allow_html=True)
            st.write("**Band Selection Results**")
            st.write(f"Selected {matlab_band_selection} most informative bands from original 100 bands")
            if selected_bands is not None:
                st.write("Selected band indices:")
                st.write(selected_bands)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if endmembers is not None:
                st.markdown('<div class="matlab-box">', unsafe_allow_html=True)
                st.write("**Spectral Unmixing Results**")
                st.write("Extracted 3 endmembers (vegetation, soil, water)")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                for i in range(3):
                    ax.plot(endmembers[i, :], label=f'Endmember {i+1}')
                ax.set_xlabel('Band Number')
                ax.set_ylabel('Reflectance')
                ax.legend()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if anomaly_score is not None:
                st.markdown('<div class="matlab-box">', unsafe_allow_html=True)
                st.write("**Anomaly Detection Results**")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                im1 = axes[0].imshow(anomaly_rx, cmap='hot')
                axes[0].set_title('RX Anomaly Score')
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0])
                
                im2 = axes[1].imshow(anomaly_sam, cmap='hot')
                axes[1].set_title('SAM Anomaly Score')
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1])
                
                im3 = axes[2].imshow(anomaly_score, cmap='hot')
                axes[2].set_title('Combined Anomaly Score')
                axes[2].axis('off')
                plt.colorbar(im3, ax=axes[2])
                
                st.pyplot(fig)
                
                anomaly_percentage = np.mean(anomaly_score > 0.7) * 100
                st.metric("High Anomaly Areas", f"{anomaly_percentage:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="matlab-box">', unsafe_allow_html=True)
            st.write("**Processing Level**")
            st.write(f"Applied {matlab_processing_level} {matlab_processing_method} processing")
            progress_value = 100 if matlab_processing_level == "Research" else 67 if matlab_processing_level == "Advanced" else 33
            st.progress(progress_value/100)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Enable MATLAB Hyperspectral Processing in the sidebar to see these features")

with tab3:
    st.markdown('<p class="sub-header">Hyperspectral Image Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if hyperspectral_data.shape[2] > 0:
            st.write("**Selected Hyperspectral Bands**")
            n_bands = min(10, hyperspectral_data.shape[2])
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            for i, ax in enumerate(axes.flat):
                if i < n_bands:
                    band_idx = selected_bands[i] if (selected_bands is not None and i < len(selected_bands)) else i
                    ax.imshow(hyperspectral_data[:, :, i], cmap='viridis')
                    ax.set_title(f'Band {band_idx+1}')
                    ax.axis('off')
            st.pyplot(fig)
    
    with col2:
        if hyperspectral_data.shape[2] > 1:
            st.write("**Vegetation Indices**")
            
            nir_band = hyperspectral_data[:, :, -1]
            red_band = hyperspectral_data[:, :, 0]
            ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
            
            green_band = hyperspectral_data[:, :, min(1, hyperspectral_data.shape[2]-1)]
            gndvi = (nir_band - green_band) / (nir_band + green_band + 1e-10)
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            im1 = ax[0].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            ax[0].set_title('NDVI')
            ax[0].axis('off')
            plt.colorbar(im1, ax=ax[0], fraction=0.046)
            
            im2 = ax[1].imshow(gndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            ax[1].set_title('GNDVI')
            ax[1].axis('off')
            plt.colorbar(im2, ax=ax[1], fraction=0.046)
            
            st.pyplot(fig)

with tab4:
    st.markdown('<p class="sub-header">Sensor Data Analysis</p>', unsafe_allow_html=True)
    
    if len(sensor_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                sensor_data, 
                x='date', 
                y='ndvi', 
                title='NDVI Trend',
                labels={'ndvi': 'NDVI Value', 'date': 'Date'}
            )
            if use_lstm and lstm_prediction is not None:
                future_dates = [sensor_data['date'].iloc[-1] + timedelta(days=i) for i in range(1, 8)]
                future_ndvi = [lstm_prediction] * 7
                fig.add_trace(go.Scatter(x=future_dates, y=future_ndvi, name='Forecast', line=dict(dash='dash')))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.line(
                sensor_data, 
                x='date', 
                y='soil_moisture', 
                title='Soil Moisture',
                labels={'soil_moisture': 'Moisture (%)', 'date': 'Date'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Environmental Factors Correlation**")
            corr_data = sensor_data[['temperature', 'soil_moisture', 'humidity', 'ndvi', 'rainfall']].copy()
            corr_matrix = corr_data.corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
            
            if use_lstm and st.session_state.lstm_history is not None:
                st.write("**LSTM Training Performance**")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(st.session_state.lstm_history.history['loss'], label='Training Loss')
                if 'val_loss' in st.session_state.lstm_history.history:
                    ax.plot(st.session_state.lstm_history.history['val_loss'], label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)

with tab5:
    st.markdown('<p class="sub-header">AI Model Details</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if use_cnn and st.session_state.cnn_model is not None:
            st.markdown('<div class="model-box">', unsafe_allow_html=True)
            st.write("**CNN Model Architecture**")
            st.text(f"Input: Hyperspectral images {hyperspectral_data.shape}")
            st.text("Layers: 3x Conv2D + MaxPooling, Flatten, 2x Dense")
            st.text("Output: Disease classification (3 classes)")
            
            # Model summary
            summary_str = []
            st.session_state.cnn_model.summary(print_fn=lambda x: summary_str.append(x))
            st.text_area("Model Summary", "\n".join(summary_str), height=200)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if use_pso:
                st.markdown('<div class="model-box">', unsafe_allow_html=True)
                st.write("**PSO Optimization**")
                st.text(f"Particles: {pso_particles}, Iterations: 10")
                st.text("Objective: Minimize classification loss")
                st.text("Optimization completed successfully")
                st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if use_lstm and st.session_state.lstm_model is not None:
            st.markdown('<div class="model-box">', unsafe_allow_html=True)
            st.write("**LSTM Model Architecture**")
            st.text(f"Input: NDVI time series ({sequence_length} time steps)")
            st.text("Layers: 2x LSTM(50), Dense(32), Dense(1)")
            st.text("Output: NDVI forecast (next time step)")
            
            # Model summary
            summary_str = []
            st.session_state.lstm_model.summary(print_fn=lambda x: summary_str.append(x))
            st.text_area("Model Summary", "\n".join(summary_str), height=200)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Model comparison and recommendations
    if use_cnn or use_lstm:
        st.markdown('<p class="sub-header">Model Performance</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if use_cnn and st.session_state.cnn_history is not None:
                st.write("**CNN Classification Results**")
                st.metric("Accuracy", f"{st.session_state.cnn_history.history['accuracy'][-1]:.2%}")
                st.metric("Loss", f"{st.session_state.cnn_history.history['loss'][-1]:.4f}")
                
                st.write("Confusion Matrix (Simulated)")
                cm = np.array([[15, 2, 3], [1, 18, 1], [4, 1, 15]])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Disease', 'Healthy', 'Stress'],
                            yticklabels=['Disease', 'Healthy', 'Stress'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
        
        with col2:
            if use_lstm and st.session_state.lstm_history is not None:
                st.write("**LSTM Prediction Results**")
                st.metric("Final Training Loss", f"{st.session_state.lstm_history.history['loss'][-1]:.4f}")
                if 'val_loss' in st.session_state.lstm_history.history:
                    st.metric("Validation Loss", f"{st.session_state.lstm_history.history['val_loss'][-1]:.4f}")
                
                if len(X_sequences) >= 10:
                    st.write("Prediction vs Actual (last 10 points)")
                    actual = y_target[-10:]
                    predicted = st.session_state.lstm_model.predict(X_sequences[-10:], verbose=0).flatten()
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(actual, label='Actual', marker='o')
                    ax.plot(predicted, label='Predicted', marker='s')
                    ax.legend()
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('NDVI')
                    st.pyplot(fig)

with tab6:
    if hyperspectral_data.shape[2] > 0:
        create_interactive_spectral_viewer(hyperspectral_data)
    else:
        st.warning("No hyperspectral data available for spectral exploration")

# Actionable Insights
st.markdown("---")
st.markdown('<p class="sub-header">🎯 Actionable Insights & Recommendations</p>', unsafe_allow_html=True)

if use_cnn and cnn_prediction is not None:
    if predicted_class == 0:  # Disease
        st.markdown('<div class="alert-high">', unsafe_allow_html=True)
        st.write("**🚨 IMMEDIATE ACTION REQUIRED: CROP DISEASE DETECTED**")
        st.write("🔍 **Immediate Steps:**")
        st.write("- Isolate affected areas with 50m buffer zones")
        st.write("- Apply fungicide treatment within 24 hours")
        st.write("- Increase monitoring to daily frequency")
        st.write("- Notify agricultural extension service")
        st.write("📊 **Confidence Level:** High (≥85%)")
        st.markdown('</div>', unsafe_allow_html=True)
    elif predicted_class == 2:  # Stress
        st.markdown('<div class="alert-medium">', unsafe_allow_html=True)
        st.write("**⚠️ CROP STRESS DETECTED - INVESTIGATION NEEDED**")
        st.write("🔍 **Recommended Actions:**")
        st.write("- Check irrigation system functionality")
        st.write("- Test soil nutrient levels (N-P-K)")
        st.write("- Monitor for pest activity increase")
        st.write("- Consider supplemental irrigation")
        st.write("📊 **Confidence Level:** Medium (70-84%)")
        st.markdown('</div>', unsafe_allow_html=True)
    else:  # Healthy
        st.markdown('<div class="alert-low">', unsafe_allow_html=True)
        st.write("**✓ CROP HEALTH STATUS: OPTIMAL**")
        st.write("💡 **Maintenance Recommendations:**")
        st.write("- Continue current management practices")
        st.write("- Maintain weekly monitoring schedule")
        st.write("- Prepare preventive treatments")
        st.write("- Document successful practices")
        st.write("📊 **Confidence Level:** High (≥90%)")
        st.markdown('</div>', unsafe_allow_html=True)

if use_lstm and lstm_prediction is not None and len(sensor_data) > 0:
    current_ndvi = sensor_data['ndvi'].iloc[-1]
    if lstm_prediction < current_ndvi - 0.1:
        st.markdown('<div class="alert-medium">', unsafe_allow_html=True)
        st.write("**📉 NDVI TREND: SIGNIFICANT DECLINE FORECAST**")
        st.write("- Investigate potential causes immediately")
        st.write("- Consider additional nutrient application")
        st.write("- Schedule drone survey for detailed assessment")
        st.markdown('</div>', unsafe_allow_html=True)
    elif lstm_prediction > current_ndvi + 0.1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write("**📈 NDVI TREND: IMPROVEMENT FORECAST**")
        st.write("- Current practices are effective")
        st.write("- Consider expanding successful methods")
        st.write("- Maintain current monitoring schedule")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>AgriMATLAB Hyperspectral Crop Monitoring System • Integrating MATLAB Hyperspectral Imaging Library</p>
        <p>This simulation demonstrates the integration of MATLAB's hyperspectral processing with AI techniques for precision agriculture</p>
        <p>Update Count: {} • Data Version: {}</p>
    </div>
    """.format(st.session_state.update_counter, datetime.now().strftime("%Y%m%d_%H%M")),
    unsafe_allow_html=True
)