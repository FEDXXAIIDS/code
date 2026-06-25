##### Config #####
CONFIG_ROUNDS: int = 7
CONFIG_CLIENTS: int = 5
CONFIG_SHAP_SAMPLE: int = 25
##################

import os
# -------------------------
# Determinism / Reproducibility
# -------------------------
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

# -------------------------
# Imports
# -------------------------
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import matplotlib.pyplot as plt
import warnings
import glob
import pandas as pd
warnings.filterwarnings('ignore')

# =====================================================
# Federated XAI IDS (CNN + Transformer Hybrid Version)
# =====================================================
class FederatedXAIIDS:
    def __init__(self, num_clients=4, input_dim=46, num_classes=6, batch_size=500000):
        self.num_clients = num_clients
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.client_models = []
        self.global_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    # -----------------------------------------------------
    # CNN + Transformer Model (Hybrid)
    # -----------------------------------------------------
    def create_model(self):
        inputs = keras.Input(shape=(self.input_dim,))

        # Reshape to 3D tensor for Conv1D
        x = layers.Reshape((self.input_dim, 1))(inputs)

        # CNN feature extractor
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.Dropout(0.3)(x)

        # --- Transformer Block ---
        # Multi-Head Attention Layer (Self-Attention)
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
        attention_output = layers.Dropout(0.3)(attention_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output) # Residual Connection

        # Feed Forward Network
        ffn_output = layers.Dense(128, activation='relu')(x)
        ffn_output = layers.Dropout(0.3)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output) # Residual Connection
        # -------------------------

        # Fully connected layers
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # -----------------------------------------------------
    # Process data files in batches
    # -----------------------------------------------------
    def process_files_in_batches(self, folder_path, file_pattern="*.csv", max_files=8, max_samples=None):
        print("Processing LIMITED files for testing...")

        csv_files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
        if not csv_files:
            raise ValueError(f"No CSV files found in {folder_path}")

        csv_files = csv_files[:max_files]
        print(f"Using {len(csv_files)} out of {len(glob.glob(os.path.join(folder_path, file_pattern)))} files for testing")

        print("Fitting preprocessing on sample data...")
        sample_data = self._get_sample_for_preprocessing(csv_files, sample_size=10000)
        self._fit_preprocessors(sample_data)

        all_client_data = {f'client_{i}': {'X_train': [], 'X_test': [], 'y_train': [], 'y_test': []}
                          for i in range(self.num_clients)}

        total_processed = 0

        for file_idx, csv_file in enumerate(csv_files):
            print(f"Processing file {file_idx+1}/{len(csv_files)}: {os.path.basename(csv_file)}")

            try:
                for chunk in pd.read_csv(csv_file, chunksize=self.batch_size, low_memory=False):
                    if max_samples and total_processed >= max_samples:
                        print(f"Reached maximum sample limit of {max_samples}")
                        break

                    X_processed, y_processed = self._process_chunk(chunk)
                    if X_processed is not None:
                        chunk_client_data = self._split_chunk_among_clients(X_processed, y_processed)
                        for client_id, data in chunk_client_data.items():
                            all_client_data[client_id]['X_train'].append(data['X_train'])
                            all_client_data[client_id]['X_test'].append(data['X_test'])
                            all_client_data[client_id]['y_train'].append(data['y_train'])
                            all_client_data[client_id]['y_test'].append(data['y_test'])
                        total_processed += len(X_processed)
                        print(f"   Processed {total_processed} samples so far...")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

            if max_samples and total_processed >= max_samples:
                break

        # Combine chunks for each client
        final_client_data = []
        for client_id in range(self.num_clients):
            client_key = f'client_{client_id}'
            if (all_client_data[client_key]['X_train'] and
                len(all_client_data[client_key]['X_train'][0]) > 0):

                X_train = np.vstack(all_client_data[client_key]['X_train'])
                X_test = np.vstack(all_client_data[client_key]['X_test'])
                y_train = np.vstack(all_client_data[client_key]['y_train'])
                y_test = np.vstack(all_client_data[client_key]['y_test'])

                final_client_data.append({
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                })
                print(f"Client {client_id+1}: {len(X_train)} training, {len(X_test)} test samples")
            else:
                final_client_data.append({
                    'X_train': np.array([]).reshape(0, self.input_dim),
                    'X_test': np.array([]).reshape(0, self.input_dim),
                    'y_train': np.array([]).reshape(0, self.num_classes),
                    'y_test': np.array([]).reshape(0, self.num_classes)
                })
        print(f"Total processed samples: {total_processed}")
        return final_client_data

    # -----------------------------------------------------
    # Helper functions
    # -----------------------------------------------------
    def _get_sample_for_preprocessing(self, csv_files, sample_size=10000):
        sample_data, total_samples = [], 0
        for csv_file in csv_files:
            if total_samples >= sample_size:
                break
            try:
                df_sample = pd.read_csv(csv_file, nrows=min(2000, max(1, sample_size//len(csv_files))), low_memory=False)
                sample_data.append(df_sample)
                total_samples += len(df_sample)
            except Exception:
                continue
        if sample_data:
            return pd.concat(sample_data, ignore_index=True)
        else:
            raise ValueError("Could not load sample data for preprocessing")

    def _fit_preprocessors(self, sample_data):
        print("Fitting preprocessing objects...")
        sample_data = self._clean_data_chunk(sample_data)
        timestamp_cols = [c for c in sample_data.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
        if timestamp_cols:
            sample_data = sample_data.drop(timestamp_cols, axis=1)
        target_col = self._identify_target_column(sample_data)
        X_sample = sample_data.drop(target_col, axis=1)
        y_sample = sample_data[target_col]
        self.input_dim = X_sample.shape[1]
        print(f"Input dimension set to: {self.input_dim}")
        y_sample = self._convert_to_six_classes(y_sample)
        self.scaler.fit(X_sample)
        self.label_encoder.fit(y_sample)
        print("Preprocessing objects fitted successfully")
        print(f"Classes: {list(self.label_encoder.classes_)}")

    def _process_chunk(self, chunk):
        chunk_clean = self._clean_data_chunk(chunk)
        if chunk_clean.empty:
            return None, None
        timestamp_cols = [c for c in chunk_clean.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
        if timestamp_cols:
            chunk_clean = chunk_clean.drop(timestamp_cols, axis=1)
        target_col = self._identify_target_column(chunk_clean)
        X = chunk_clean.drop(target_col, axis=1)
        y = chunk_clean[target_col]
        y = self._convert_to_six_classes(y)
        X_processed = self.scaler.transform(X)
        y_encoded = self.label_encoder.transform(y)
        y_processed = keras.utils.to_categorical(y_encoded, self.num_classes)
        return X_processed, y_processed

    def _clean_data_chunk(self, df_chunk):
        for col in df_chunk.select_dtypes(include=['object']).columns:
            if col not in ['Label', 'label', 'attack', 'class']:
                try:
                    df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
                except:
                    pass
        df_chunk = df_chunk.replace([np.inf, -np.inf], np.nan)
        df_chunk = df_chunk.dropna()
        return df_chunk

    def _identify_target_column(self, df):
        for c in ['label', 'Label', 'attack', 'Attack', 'class', 'Class']:
            if c in df.columns:
                return c
        return df.columns[-1]

    def _split_chunk_among_clients(self, X_chunk, y_chunk):
        chunk_size = len(X_chunk)
        if chunk_size == 0:
            return {}
        samples_per_client = chunk_size // self.num_clients
        client_data = {}
        for i in range(self.num_clients):
            start_idx, end_idx = i * samples_per_client, (i + 1) * samples_per_client
            if i == self.num_clients - 1:
                end_idx = chunk_size
            if start_idx < end_idx:
                X_client, y_client = X_chunk[start_idx:end_idx], y_chunk[start_idx:end_idx]
                if len(X_client) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_client, y_client, test_size=0.2, random_state=seed_value
                    )
                    client_data[f'client_{i}'] = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test
                    }
                else:
                    client_data[f'client_{i}'] = {
                        'X_train': X_client,
                        'X_test': np.array([]).reshape(0, self.input_dim),
                        'y_train': y_client,
                        'y_test': np.array([]).reshape(0, self.num_classes)
                    }
        return client_data

    def _convert_to_six_classes(self, y):
        y_str = y.astype(str).str.lower()
        class_mapping = {}
        for lbl in y_str.unique():
            lbl_low = lbl.lower()
            if any(k in lbl_low for k in ['benign', 'normal', 'legitimate']):
                class_mapping[lbl] = 'Benign'
            elif any(k in lbl_low for k in ['ddos', 'distributed denial']):
                class_mapping[lbl] = 'DDoS'
            elif any(k in lbl_low for k in ['dos', 'denial of service']):
                class_mapping[lbl] = 'DoS'
            elif 'mirai' in lbl_low:
                class_mapping[lbl] = 'Mirai'
            elif any(k in lbl_low for k in ['recon', 'scan', 'portscan', 'hostscan']):
                class_mapping[lbl] = 'Recon'
            else:
                class_mapping[lbl] = 'Other'
        return y_str.map(class_mapping)

    # -----------------------------------------------------
    # Federated Training
    # -----------------------------------------------------
    def federated_training(self, client_data, rounds=5, epochs=10):
        print("\nStarting Federated Training (TEST MODE)...")
        self.global_model = self.create_model()
        global_weights = self.global_model.get_weights()
        accuracy_trends = {f'client_{i}': [] for i in range(self.num_clients)}

        for r in range(rounds):
            print(f"\n--- Federated Round {r+1} ---")
            client_weights, client_samples = [], []
            for i, data in enumerate(client_data):
                if len(data['X_train']) == 0:
                    print(f"Client {i+1}: No data, skipping...")
                    continue
                print(f"Training Client {i+1}...")
                local_model = self.create_model()
                local_model.set_weights(global_weights)
                history = local_model.fit(
                    data['X_train'], data['y_train'],
                    epochs=epochs, batch_size=256,
                    validation_data=(data['X_test'], data['y_test']),
                    verbose=0
                )
                acc = history.history['val_accuracy'][-1]
                accuracy_trends[f'client_{i}'].append(acc)
                client_weights.append(local_model.get_weights())
                client_samples.append(len(data['X_train']))
                print(f"Client {i+1} accuracy: {acc:.4f}")

            if client_weights:
                print("Aggregating weights with FedAvg...")
                new_global_weights = []
                total_samples = sum(client_samples)
                for weights in zip(*client_weights):
                    agg = np.zeros_like(weights[0])
                    for w, n in zip(weights, client_samples):
                        agg += w * (n / total_samples)
                    new_global_weights.append(agg)
                global_weights = new_global_weights
                self.global_model.set_weights(global_weights)

            global_acc = self.evaluate_global_model(client_data)
            print(f"Global model accuracy after round {r+1}: {global_acc:.4f}")
        return accuracy_trends

    # -----------------------------------------------------
    # Evaluate Global Model
    # -----------------------------------------------------
    def evaluate_global_model(self, client_data):
        X_test_all, y_test_all = [], []
        for d in client_data:
            if len(d['X_test']) > 0:
                X_test_all.append(d['X_test'])
                y_test_all.append(d['y_test'])
        if not X_test_all:
            return 0.0
        X_test, y_test = np.vstack(X_test_all), np.vstack(y_test_all)
        loss, acc = self.global_model.evaluate(X_test, y_test, verbose=0)
        return acc

    # -----------------------------------------------------
    # Train and Evaluate
    # -----------------------------------------------------
    def train_and_evaluate(self, data_folder, file_pattern="*.csv"):
        client_data = self.process_files_in_batches(data_folder, file_pattern)
        accuracy_trends = self.federated_training(client_data, rounds=CONFIG_ROUNDS, epochs=10)

        print("\n" + "="*50)
        print("TEST MODE - FINAL EVALUATION")
        print("="*50)
        final_acc = self.evaluate_global_model(client_data)
        print(f"Final Global Model Test Accuracy: {final_acc:.4f}")

        X_test_all, y_test_all = [], []
        for d in client_data:
            if len(d['X_test']) > 0:
                X_test_all.append(d['X_test'])
                y_test_all.append(d['y_test'])
        if X_test_all:
            X_test, y_test = np.vstack(X_test_all), np.vstack(y_test_all)
            y_pred = self.global_model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            print("\nClassification Report:")
            print(classification_report(y_true_classes, y_pred_classes, target_names=self.label_encoder.classes_))
            print(f"Overall Accuracy: {accuracy_score(y_true_classes, y_pred_classes):.4f}")

        # SHAP Explainability
        print("\n" + "="*60)
        print("🔍 Applying SHAP (Kernel SHAP) for Global Explainability")
        print("="*60)
        try:
            sample_size = min(CONFIG_SHAP_SAMPLE, X_test.shape[0])
            rng = np.random.default_rng(seed_value)
            shap_idx = rng.choice(X_test.shape[0], sample_size, replace=False)
            X_shap = X_test[shap_idx]
            background_size = min(100, X_test.shape[0])
            background_idx = rng.choice(X_test.shape[0], background_size, replace=False)
            X_bg = X_test[background_idx]
            explainer = shap.KernelExplainer(self.global_model.predict, X_bg)
            shap_vals = explainer.shap_values(X_shap, nsamples=100)
            shap.summary_plot(shap_vals, X_shap, show=False, plot_type="bar")
            plt.title("Global Feature Importance (SHAP - Kernel Explainer)")
            plt.tight_layout()
            plt.savefig("shap_global_feature_importance_bar.png")
            plt.close()
            shap.summary_plot(shap_vals, X_shap, show=False)
            plt.title("Global SHAP Summary (Feature Impact on Predictions)")
            plt.tight_layout()
            plt.savefig("shap_global_summary_dot.png")
            plt.close()
            print("📊 SHAP summary plots saved successfully.")
        except Exception as e:
            print(f"⚠️ SHAP explainability failed: {e}")

        return self.global_model, accuracy_trends

# =====================================================
# Run Federated CNN + Transformer IDS
# =====================================================
if __name__ == "__main__":
    fed_xai_ids = FederatedXAIIDS(num_clients=CONFIG_CLIENTS, input_dim=46, num_classes=6, batch_size=500000)
    data_folder = r"CICIoT2023"
    file_pattern = "*.csv"

    try:
        print("=== TEST MODE: Federated CNN + Transformer XAI IDS Training with LIMITED DATASET ===")
        print("Using only a deterministic subset of files for quick testing")
        print("=" * 70)
        global_model, accuracy_trends = fed_xai_ids.train_and_evaluate(data_folder, file_pattern)

        global_model.save("fedxaiids_Transformer_TEST.h5")
        print("Model saved as 'fedxaiids_Transformer_TEST.h5'")

        import joblib
        joblib.dump(fed_xai_ids.scaler, 'scaler_TEST.pkl')
        joblib.dump(fed_xai_ids.label_encoder, 'label_encoder_TEST.pkl')
        print("Preprocessing objects saved")

        print("\n" + "="*50)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*50)
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
