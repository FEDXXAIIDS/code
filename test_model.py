import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# ==========================================
# 1. Load the Pre-trained Model & Preprocessors
# ==========================================
model_path = "fedxaiids_CNNLSTM_TEST.h5"
scaler_path = "scaler_TEST.pkl"
encoder_path = "label_encoder_TEST.pkl"

print("Loading saved artifacts...")
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

# Inspect the architecture (Expected input dim: (None, 46))
model.summary()

# ==========================================
# 2. Setup Meta-parameters from Model Input
# ==========================================
# Dynamic retrieval of expected feature size from the model layer
input_dim = model.input_shape[1]
print(f"\nModel expects {input_dim} numerical features per sample.")

# ==========================================
# 3. Choose your Input Data Source
# ==========================================
USE_REAL_DATA = True  # Flip to True if you want to test an actual CSV snippet

if USE_REAL_DATA:
    # Example: Reading a real line from your CICIoT2023 files
    csv_file_path = "test_data.csv"

    # Read just a tiny snippet for prediction
    df_chunk = pd.read_csv(csv_file_path, nrows=5, low_memory=False)

    # Replicate your script's exact cleaning routine
    for col in df_chunk.select_dtypes(include=["object"]).columns:
        if col not in ["Label", "label", "attack", "class"]:
            df_chunk[col] = pd.to_numeric(df_chunk[col], errors="coerce")
    df_chunk = df_chunk.replace([np.inf, -np.inf], np.nan).dropna()

    # Identify and drop target labels & timestamps
    timestamp_cols = [
        c for c in df_chunk.columns if "time" in c.lower() or "timestamp" in c.lower()
    ]
    if timestamp_cols:
        df_chunk = df_chunk.drop(timestamp_cols, axis=1)

    target_col = next(
        (
            c
            for c in ["label", "Label", "attack", "Attack", "class", "Class"]
            if c in df_chunk.columns
        ),
        df_chunk.columns[-1],
    )
    X_raw = df_chunk.drop(target_col, axis=1).values

    print(f"\nLoaded {len(X_raw)} real sample(s) from CSV.")
else:
    # Generate mock unscaled data mirroring your tabular features
    print("\nGenerating mock unscaled feature data...")
    X_raw = np.random.rand(1, input_dim).astype(np.float32)

# ==========================================
# 4. Preprocess Data & Run Inference
# ==========================================
# Step A: Standardize features using the training pipeline's scaler
X_scaled = scaler.transform(X_raw)

# Step B: Pass into the model (it handles its own internal reshaping to (46, 1) inside the first layer)
raw_predictions = model.predict(X_scaled)

# Step C: Parse Softmax distributions into human-readable text classes
predicted_classes_idx = np.argmax(raw_predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes_idx)

# ==========================================
# 5. Output Results
# ==========================================
print("\n" + "=" * 40)
print("INFERENCE RESULTS")
print("=" * 40)
for i, (prob, lbl) in enumerate(zip(raw_predictions, predicted_labels)):
    print(f"Sample {i + 1}:")
    print(f"  -> Predicted Class: {lbl}")
    print(f"  -> Confidence Distribution: {prob}")
