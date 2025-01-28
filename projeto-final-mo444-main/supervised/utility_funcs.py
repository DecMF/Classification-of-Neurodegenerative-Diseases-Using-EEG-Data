
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def evaluate_thresholds(y_true, y_pred_probs, thresholds):

    results = []

    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_probs)
        
        results.append({
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": auc
        })
    return pd.DataFrame(results)

def build_model(param_grid, n_timesteps, n_features):
    model = Sequential([
        Input(shape=(n_timesteps, n_features)),
        Conv1D(filters=param_grid["filters_1"], kernel_size=3, activation='relu', kernel_regularizer=l2(param_grid["l2_reg"])),
        MaxPooling1D(pool_size=2),
        Dropout(param_grid["dropout_rate"]),
        
        Conv1D(filters=param_grid["filters_2"], kernel_size=3, activation='relu', kernel_regularizer=l2(param_grid["l2_reg"])),
        MaxPooling1D(pool_size=2),
        Dropout(param_grid["dropout_rate"]),
        
        LSTM(param_grid["lstm_units"], activation='tanh', return_sequences=False, kernel_regularizer=l2(param_grid["l2_reg"])),
        Dropout(param_grid["dropout_rate"]),
        
        Dense(param_grid["dense_units"], activation='relu', kernel_regularizer=l2(param_grid["l2_reg"])),
        Dropout(param_grid["dropout_rate"] / 2),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_model(model, param_grid, x_train, y_train, x_val, y_val):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=param_grid['learning_rate']),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='AUC'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    callbacks = [
        EarlyStopping(monitor='val_AUC', patience=2, restore_best_weights=True, mode='max')
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    return history

def objective(trial, x_train, y_train, x_val, y_val, n_timesteps, n_features):
    param_grid = {
        "filters_1": trial.suggest_categorical("filters_1", [16, 32, 64]),
        "filters_2": trial.suggest_categorical("filters_2", [32, 64, 128]),
        "lstm_units": trial.suggest_categorical("lstm_units", [50, 100, 150]),
        "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "l2_reg": trial.suggest_float("l2_reg", 1e-4, 1e-3, log=True),
    }

    model = build_model(param_grid, n_timesteps, n_features)
    history = train_model(model, param_grid, x_train, y_train, x_val, y_val)

    val_auc = max(history.history['val_AUC'])
    return val_auc


def plot_training_curves(history):
    """Plota as curvas de treinamento e validação (loss)."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Treino - Loss', linewidth=2.5, color="#A1D490")  # Verde pastel
    plt.plot(history.history['val_loss'], label='Validação - Loss', linewidth=2.5, color="#91BFF8")  # Azul pastel
    
    plt.title('Curva de Treinamento e Validação - Loss', fontsize=16, weight='bold')
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.gca().set_facecolor('#F9F9F9')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_pred_probs):
    """Plota a curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc_score = roc_auc_score(y_true, y_pred_probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve - Validação (AUC: {auc_score:.3f})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    
    plt.title('Curva ROC - Validação', fontsize=16, weight='bold')
    plt.xlabel('Taxa de Falsos Positivos', fontsize=14)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=14)
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_probs):
    """Plota a curva Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    
    plt.figure(figsize=(12, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve - Validação', linewidth=2.5, color="#FF9F89")
    
    plt.title('Curva Precision-Recall - Validação', fontsize=16, weight='bold')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(fontsize=12, loc="lower left")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()



def plot_precision_recall_by_threshold(metrics_table, title="Precision vs Recall por Threshold"):
    """
    Plota as curvas de Precision e Recall em função do Threshold.

    Parameters:
    - metrics_table: DataFrame contendo as colunas 'Threshold', 'Precision' e 'Recall'.
    - title: Título do gráfico (opcional, padrão: "Precision vs Recall por Threshold").
    """
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_table.Threshold, metrics_table.Precision, 
             label="Precision", marker='o', linestyle='-', linewidth=2.5, markersize=8, color="#A1D490")  # Verde pastel
    plt.plot(metrics_table.Threshold, metrics_table.Recall, 
             label="Recall", marker='o', linestyle='-', linewidth=2.5, markersize=8, color="#91BFF8")  # Azul pastel
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Métricas", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    plt.gca().set_facecolor('#F9F9F9')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

