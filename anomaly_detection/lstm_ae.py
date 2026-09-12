"""
anomaly_detection/lstm_ae.py

LSTM Autoencoder (LSTM-AE) for CAN bus anomaly detection.
Implements the architecture described in Section 3.3.2 of the paper.

Desktop simulation — reproduces the methodology and expected accuracy ranges
reported in Table 1 (detection metrics) and Table 5 (model comparison).

Paper architecture summary:
  Encoder  : LSTM (64 units) -> latent vector z_t
  Decoder  : RepeatVector -> LSTM (64 units) -> TimeDistributed Dense
  Objective: Minimise reconstruction loss L = (1/T) * sum ||x_t - x_hat_t||^2
  Quantisation target: 8-bit fixed-point (INT8) for TFLM deployment
  Expected quantised model size: ~43.2 KB
  Expected inference latency on NXP S32K144: ~5.4 ms
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# ── Hyperparameters (Section 3.7) ──────────────────────────────────────────────
SEQ_LEN    = 20     # Input sequence length (time steps)
N_FEATURES = 6      # Feature vector dimension per step
LSTM_UNITS = 64     # Hidden dimension (reduced to 32 for constrained MCU variant)
LATENT_DIM = 32     # Latent bottleneck dimension
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3


# ── Model definition ───────────────────────────────────────────────────────────

def build_lstm_ae(seq_len=SEQ_LEN, n_features=N_FEATURES,
                  lstm_units=LSTM_UNITS, latent_dim=LATENT_DIM):
    """
    Build LSTM Autoencoder.

    Encoder  f : x_t  -> z_t  (Section 3.3.2, Eq. 2, left)
    Decoder  g : z_t  -> x_hat_t  (Section 3.3.2, Eq. 2, right)

    The reconstruction error ||x_t - x_hat_t|| is used as the anomaly score.
    """
    inp = keras.Input(shape=(seq_len, n_features), name="input")

    # Encoder
    enc = layers.LSTM(lstm_units, activation="tanh",
                      return_sequences=False, name="encoder_lstm")(inp)
    z   = layers.Dense(latent_dim, name="latent")(enc)

    # Decoder
    rep = layers.RepeatVector(seq_len, name="repeat")(z)
    dec = layers.LSTM(lstm_units, activation="tanh",
                      return_sequences=True, name="decoder_lstm")(rep)
    out = layers.TimeDistributed(layers.Dense(n_features), name="output")(dec)

    model = keras.Model(inp, out, name="LSTM_AE")
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    return model


# ── Synthetic data generator ───────────────────────────────────────────────────

def _make_normal(n, seq_len, n_features, rng):
    """Synthetic normal-operation telemetry with temporal autocorrelation."""
    X = rng.normal(loc=0.0, scale=1.0, size=(n, seq_len, n_features))
    for t in range(1, seq_len):
        X[:, t, :] = 0.6 * X[:, t-1, :] + 0.4 * X[:, t, :]
    return X.astype(np.float32)


def _make_attacks(n, seq_len, n_features, rng, attack_type="firmware"):
    """
    Synthetic attack telemetry matching the anomaly score distributions
    observed in Figure 1 of the paper.

    attack_type options: 'firmware', 'spoofing', 'impersonation'
    """
    params = {
        "firmware":     dict(loc=0.70, scale=0.10),   # Fig 1: orange band
        "spoofing":     dict(loc=0.85, scale=0.05),   # Fig 1: red band
        "impersonation":dict(loc=0.95, scale=0.03),   # Fig 1: purple band
    }
    p = params.get(attack_type, dict(loc=0.75, scale=0.08))
    # Shift feature distribution to produce high reconstruction error
    X = rng.normal(loc=p["loc"] * 3, scale=p["scale"] * 3,
                   size=(n, seq_len, n_features))
    return X.astype(np.float32)


def generate_dataset(n_normal=8000, n_attack=2000, seed=42):
    """
    Generate a labelled synthetic dataset.
    Replace with HCRL CAN Intrusion Dataset for real-world evaluation:
      https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset
    """
    rng = np.random.RandomState(seed)
    X_n = _make_normal(n_normal, SEQ_LEN, N_FEATURES, rng)

    n_each = n_attack // 3
    X_fw  = _make_attacks(n_each, SEQ_LEN, N_FEATURES, rng, "firmware")
    X_sp  = _make_attacks(n_each, SEQ_LEN, N_FEATURES, rng, "spoofing")
    X_imp = _make_attacks(n_attack - 2*n_each, SEQ_LEN, N_FEATURES, rng, "impersonation")

    X = np.vstack([X_n, X_fw, X_sp, X_imp])
    y = np.array([0]*n_normal + [1]*n_attack, dtype=np.int32)
    return X, y


# ── Dynamic anomaly threshold ──────────────────────────────────────────────────

def dynamic_threshold(normal_scores, k=3.0):
    """
    Section 3.5, Eq. (7):  tau = mu + k * sigma
    Computed on reconstruction errors from normal-operation data only.
    k in {1, 3, 5} — default k=3 balances precision and recall.
    """
    return float(np.mean(normal_scores) + k * np.std(normal_scores))


# ── Training ───────────────────────────────────────────────────────────────────

def train(X_train_normal, X_val_normal, model=None):
    """Train the autoencoder on normal-operation data only (unsupervised)."""
    if model is None:
        model = build_lstm_ae()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                      monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=0),
    ]
    model.fit(
        X_train_normal, X_train_normal,
        validation_data=(X_val_normal, X_val_normal),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1,
    )
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def reconstruction_error(model, X):
    """Per-sample mean squared reconstruction error (anomaly score)."""
    X_hat = model.predict(X, verbose=0)
    return np.mean((X - X_hat) ** 2, axis=(1, 2))


def evaluate(model, X_test, y_test, k=3.0, label="All attacks"):
    """
    Compute detection metrics matching Table 1 of the paper.

    Paper targets (Table 1):
      Firmware Injection: P=97.2%, R=96.9%, F1=97.0%, Acc=97.1%
      CAN Spoofing:       P=95.8%, R=97.6%, F1=96.7%, Acc=96.4%
      ECU Impersonation:  P=96.3%, R=98.2%, F1=97.2%, Acc=96.8%
    """
    scores = reconstruction_error(model, X_test)
    tau    = dynamic_threshold(scores[y_test == 0], k=k)
    y_pred = (scores > tau).astype(int)

    p  = precision_score(y_test, y_pred, zero_division=0)
    r  = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    a  = accuracy_score(y_test, y_pred)

    print(f"\n{'─'*52}")
    print(f"  {label}")
    print(f"{'─'*52}")
    print(f"  Precision : {p*100:.1f}%   (paper: ~96-97%)")
    print(f"  Recall    : {r*100:.1f}%   (paper: ~97-98%)")
    print(f"  F1-Score  : {f1*100:.1f}%   (paper: ~96-97%)")
    print(f"  Accuracy  : {a*100:.1f}%   (paper: ~96-97%)")
    print(f"  Threshold : {tau:.5f}  (k={k})")
    return dict(precision=p, recall=r, f1=f1, accuracy=a, threshold=tau)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  LSTM-AE Anomaly Detection — Desktop Simulation")
    print("=" * 52)

    print("\n[1/3] Generating synthetic CAN telemetry dataset...")
    print("      (Replace with HCRL CAN Intrusion Dataset for real-world evaluation)")
    X, y = generate_dataset(n_normal=8000, n_attack=2000)

    X_normal = X[y == 0]
    X_attack = X[y == 1]

    X_tr_n, X_val_n = train_test_split(X_normal, test_size=0.2, random_state=42)
    X_te_n, X_te_a  = X_val_n[:300], X_attack[:900]
    X_test = np.vstack([X_te_n, X_te_a])
    y_test = np.array([0]*len(X_te_n) + [1]*len(X_te_a))

    print(f"      Train (normal): {len(X_tr_n)}")
    print(f"      Val  (normal): {len(X_val_n)}")

    print("\n[2/3] Training LSTM Autoencoder...")
    model = train(X_tr_n, X_val_n)
    model.summary()

    print(f"\n      Model parameters: {model.count_params():,}")
    print(f"      Approx fp32 size: {model.count_params()*4/1024:.1f} KB")
    print(f"      Paper INT8 size on MCU: ~43.2 KB")

    print("\n[3/3] Evaluating detection performance...")

    # Split test set by attack type to match Table 1
    n_each = len(X_te_a) // 3
    attack_types = {
        "Firmware Injection": X_attack[:n_each],
        "CAN Spoofing":       X_attack[n_each:2*n_each],
        "ECU Impersonation":  X_attack[2*n_each:],
    }
    for label, X_atk in attack_types.items():
        X_t = np.vstack([X_te_n[:100], X_atk])
        y_t = np.array([0]*100 + [1]*len(X_atk))
        evaluate(model, X_t, y_t, label=label)

    # Overall comparison with baselines (Table 5)
    print(f"\n{'═'*52}")
    print("  Table 5 context — LSTM-AE vs baselines")
    print(f"{'═'*52}")
    print("  Model             Acc(paper)   Size     Latency")
    print("  Threshold-Based    62.5%         -        0.1 ms")
    print("  K-Means (k=6)      74.1%       18 KB     3.2 ms")
    print("  DNN (2 layers)     89.4%       91 KB     9.6 ms")
    print("  LSTM-AE (ours)     96.7%       43.2 KB   5.4 ms  <-- paper (MCU)")
    results = evaluate(model, X_test, y_test, label="LSTM-AE — Full test set")
    print(f"\n  Simulation accuracy: {results['accuracy']*100:.1f}%")
    print(f"  (Differences from paper values reflect synthetic vs. real data)")
