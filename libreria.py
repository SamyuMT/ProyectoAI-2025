# Standard library
import json
import os
from pathlib import Path
from typing import List, Tuple

# Third-party libraries
import cloudpickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import randint, uniform
from tqdm import tqdm

# scikit-learn core & utilities
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.model_selection import (
    HalvingRandomSearchCV,
    KFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    explained_variance_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Gradient-boosting libraries
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def load_sample(path: Path, n: int | None) -> pl.DataFrame:
    """Carga CSV con dtypes correctos y muestreo aleatorio, conservando nulos."""
    df = pl.read_csv(
        path,
        infer_schema_length=None,
        low_memory=True
    )
    if n is not None:
        df = df.sample(n, with_replacement=False, shuffle=True, seed=42)
    return df

def concatenate_dfs(dfs: list[Path], sample: int | None = None) -> pl.DataFrame:
    frames = []
    for trimestre, csv_path in tqdm(dfs.items(), desc="Cargando"):
        df = load_sample(csv_path, sample).with_columns(
            pl.lit(trimestre).alias("TRIMESTRE_REF")
        )
        frames.append(df)
    """Concatena una lista de DataFrames de Polars."""
    return pl.concat(frames)



def encode_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Codifica una columna categórica usando LabelEncoder y agrega una nueva columna con sufijo '_ENC'.
    Guarda el mapeo en un archivo JSON dentro de la carpeta 'logs'.
    """
    # Convertir la columna a pandas Series para LabelEncoder
    values = df[column].to_pandas()
    le = LabelEncoder()
    encoded = le.fit_transform(values)
    # Nombre de la nueva columna
    new_col = f"{column}_ENC"
    # Agregar la columna codificada al DataFrame
    df = df.with_columns(
        pl.Series(new_col, encoded)
    )
    # Guardar el mapeo en un archivo JSON en la carpeta 'logs'
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    os.makedirs('logs', exist_ok=True)
    json_path = os.path.join('logs', f'{column.lower()}_label_mapping.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        mapping_serializable = {str(k): int(v) for k, v in mapping.items()}
        json.dump(mapping_serializable, f, ensure_ascii=False, indent=2)
    return df

def delate_column(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Elimina una columnas del DataFrame.
    """
    df = df.drop(columns)
    return df

def val_nulos(df: pl.DataFrame) -> pl.DataFrame:
    valores_nulos = df.select([
        pl.col(col).is_null().sum().alias(col) for col in df.columns
    ]).to_dict(as_series=False)
    return valores_nulos


def drop_high_nulls(
    df: pl.DataFrame,
    threshold: float = 0.70,
    verbose: bool = True
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Elimina columnas cuyo porcentaje de nulos supere 'threshold'.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame original.
    threshold : float, default 0.70
        Umbral máximo de nulos permitido (0 – 1).
    verbose : bool
        Imprime o no la lista de columnas eliminadas.

    Returns
    -------
    cleaned : pl.DataFrame
        DataFrame sin las columnas con exceso de nulos.
    dropped : list[str]
        Nombres de las columnas eliminadas.
    """
    n_rows = df.height
    # ratio de nulos por columna
    null_ratios = (
        df
        .select([
            pl.col(c).is_null().sum().alias(c) for c in df.columns
        ])
        .row(0)  # devuelve lista con los totales
    )
    dropped = [
        (col, str(df.schema[col])) for col, nulls in zip(df.columns, null_ratios)
        if nulls / n_rows > threshold
    ]
    if verbose and dropped:
        cols_str = ', '.join([f"{col} ({dtype})" for col, dtype in dropped])
        print(f"🗑️  Columnas eliminadas (+{threshold*100:.0f}% nulos): {cols_str}")

    cleaned = df.drop([col for col, _ in dropped])
    return cleaned, [col for col, _ in dropped]


def corr_with_target(df: pl.DataFrame, num_cols: list, target: str) -> pl.DataFrame:
    corrs = []
    for col in num_cols:
        # Solo calcula si la columna no es el target
        if col != target:
            # Selecciona solo la columna y el target para ahorrar memoria
            sub_df = df.select([col, target]).drop_nulls()
            if sub_df.height > 0:
                corr = sub_df.corr()[0, 1]
                corrs.append((col, corr))
            else:
                corrs.append((col, None))
    return pl.DataFrame({"variable": [c[0] for c in corrs], "pearson_corr": [c[1] for c in corrs]})


def plot_correlation_matrix(df: pl.DataFrame, num_cols: list, target: str) -> None:
    """
    Plotea la matriz de correlación entre las columnas numéricas y el target.
    """
    corrs = corr_with_target(df, num_cols, target)
    corrs = corrs.sort("pearson_corr", descending=True).to_pandas()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x="variable", y="pearson_corr", data=corrs)
    plt.xticks(rotation=90)
    plt.axvline(0, color='gray', linestyle='--')
    plt.title(f"Correlación con {target}")
    plt.xlabel("Coeficiente de correlación de Pearson")
    plt.ylabel("Variables")
    plt.show()

def fill_null_column(df, column, value):
    # Polars maneja internamente el procesamiento en lotes y es eficiente con grandes volúmenes de datos.
    # No es necesario dividir manualmente el DataFrame en lotes para operaciones como fill_null.
    # Solo define la función normalmente:
    return df.with_columns(
        pl.col(column).fill_null(value)
    )


def df_full_clean(df: pl.DataFrame) -> pl.DataFrame:
    print(df.shape)
    df = encode_column(df, "MOVIMIENTO")
    df = df.with_columns(
    pl.col("MESINICIO").str.slice(-4).cast(pl.Int32).alias("ANIO_MESINICIO")
    )
    ID_COLS = [
    "NOFORMULAR",    # id único de la obra/formulario
    "CONS_ID",       # id asociado a mano de obra
    "CENSO_NUMERO",   # (si también es secuencial)
    "MOVIMIENTO",    
    "MESINICIO", 
    ]
    df = delate_column(df, ID_COLS)
    df_combined, cols_out = drop_high_nulls(df, 0.70)
    del df
    print(df_combined.shape)
    ID_COLS = [
    "MEZ_OBRA",
    "CONCRETO"      
    ]
    df_combined_new = delate_column(df_combined, ID_COLS)
    del df_combined
    # Ejemplo de uso:
    valor_definido = 0  # Cambia este valor por el que desees
    df_combined_new = fill_null_column(df_combined_new, "AREA_NOVIS", valor_definido)
    df_combined_new = fill_null_column(df_combined_new, "UNI_DEC_NOVIS", valor_definido)
    # Lista de columnas objetivo y sus valores de nulos (solo usamos los nombres)
    cols_a_rellenar = [
        "C1_EXCAVACION", "C1_CIMENTACION", "C1_DESAGUES", "C2_ESTRUCTURA", "C2_INST_HIDELEC",
        "C2_CUBIERTA", "C3_MAMPOSTERIA", "C3_PANETE", "C4_PISO_ENCHAPE", "C4_CARP_METALICA",
        "C4_CARP_MADERA", "C4_CIELO_RASO", "C5_VID_CERRAJERIA", "C5_PINTURA",
        "C6_REM_EXTERIORES", "C6_REM_ACABADOS", "C6_ASEO"
    ]

    for col in cols_a_rellenar:
        valor_definido = df_combined_new[col].mean()  # Obtener el valor mediano de la columna
        df_combined_new = fill_null_column(df_combined_new, col, valor_definido)
    df_combined_new = df_combined_new.filter(~pl.col("AMPLIACION").is_null()) # Elimina filas porque no hay tantos invalidos
    df_combined_new = df_combined_new.filter(~pl.col("ANIO_MESINICIO").is_null()) # Elimina filas porque no hay tantos invalidos
    valor_definido = df_combined_new["MANO_OBRAF"].mean()  # Obtener el valor mediano de la columna
    df_combined_new = fill_null_column(df_combined_new, "MANO_OBRAF", valor_definido)
    valor_definido = 0  # Cambia este valor por el que desees
    df_combined_new = fill_null_column(df_combined_new, "LIC_RADICADO_SN", valor_definido)
    valor_definido = 0  # Cambia este valor por el que desees
    df_combined_new = fill_null_column(df_combined_new, "USO_DOS", valor_definido)
    valor_definido = 0  # Cambia este valor por el que desees
    df_combined_new = fill_null_column(df_combined_new, "SIS_CONSTR", valor_definido)
    return df_combined_new

def corr_target(Target, df):
    """
    Calcula la correlación de las variables con la variable objetivo.
    """
    num_cols = [
        col for col, dtype in zip(df.columns, df.dtypes)
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)
    ]
    corr_target = corr_with_target(df, num_cols, Target).sort("pearson_corr", descending=True)
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(corr_target)
    plot_correlation_matrix(
    df,
    num_cols,
    target=Target
    )



def make_search(model, param_dist):
    return HalvingRandomSearchCV(
        model,
        param_dist,
        resource="n_estimators",
        max_resources=1200,   # valor del constructor
        min_resources=300,
        factor=2,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

# --- Scorers que devuelven métricas EN ESCALA ORIGINAL -------------
def _mae_unscaled(y_true_s, y_pred_s, scaler):
    y_t = scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
    y_p = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    return mean_absolute_error(y_t, y_p)

def _rmse_unscaled(y_true_s, y_pred_s, scaler):
    y_t = scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
    y_p = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    return np.sqrt(mean_squared_error(y_t, y_p))

def _r2_unscaled(y_true_s, y_pred_s, scaler):
    y_t = scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
    y_p = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    return r2_score(y_t, y_p)


def train_evaluate_regression_models(models, X_train_trans, y_train,
                                     X_val_trans, y_val_raw, y_scaler,clave,
                                     artifacts_dir=None, save_models=True,
                                     cv_folds: int = 0, random_state: int = 42):
    """
    Entrena y evalúa múltiples modelos de regresión con comparación hold-out
    y validación cruzada opcional.

    Parameters
    ----------
    ...
    cv_folds : int, default=0
        Si >1 se realiza K-Fold con ese número de particiones sobre el set
        de entrenamiento y se reportan las métricas medias ± std.
    random_state : int, default=42
        Semilla para la división K-Fold reproducible.

    Returns
    -------
    pd.DataFrame
        Métricas comparativas por modelo (hold-out y, opcionalmente, CV).
    """
    # -----------------------------------------------------------------
    # Preparativos
    # -----------------------------------------------------------------
    if save_models and artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_metrics = {}          # resultados finales
    all_predictions = {}        # para gráficos hold-out

    # Para residual plots y boxplot de errores (igual que tu versión)
    plt.figure(figsize=(16, 8))
    fig_resid, ax_resid = plt.subplots(1, len(models), figsize=(18, 6))
    fig_err, ax_err = plt.subplots(figsize=(12, 8))

    # -----------------------------------------------------------------
    # Loop por modelo
    # -----------------------------------------------------------------
    for i, (name, model) in enumerate(models.items()):
        print(f"\nEntrenando y evaluando {name}...")

        # ---------------------- FIT FINAL (todo el training) ----------
        model.fit(X_train_trans, y_train)

        if save_models and artifacts_dir is not None:
            joblib.dump(model, artifacts_dir / f"{name}_{clave}_model.joblib")

        # ---------------------- PREDICCIONES HOLD-OUT -----------------
        y_pred_scaled = model.predict(X_val_trans)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        all_predictions[name] = y_pred

        # Métricas hold-out
        mae = mean_absolute_error(y_val_raw, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_raw, y_pred))
        r2 = r2_score(y_val_raw, y_pred)
        ev = explained_variance_score(y_val_raw, y_pred)

        # ---------------------- VALIDACIÓN CRUZADA --------------------
        if cv_folds and cv_folds > 1:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

            scoring = {
                'MAE':  make_scorer(_mae_unscaled, greater_is_better=False,
                                    scaler=y_scaler),
                'RMSE': make_scorer(_rmse_unscaled, greater_is_better=False,
                                    scaler=y_scaler),
                'R2':   make_scorer(_r2_unscaled, scaler=y_scaler)
            }

            cv_results = cross_validate(
                model, X_train_trans, y_train,
                cv=kf, scoring=scoring, n_jobs=-1, return_train_score=False
            )

            mae_cv  = -cv_results['test_MAE'].mean()
            rmse_cv = -cv_results['test_RMSE'].mean()
            r2_cv   =  cv_results['test_R2'].mean()

            mae_cv_std  = cv_results['test_MAE'].std()
            rmse_cv_std = cv_results['test_RMSE'].std()
            r2_cv_std   = cv_results['test_R2'].std()
        else:
            mae_cv = rmse_cv = r2_cv = mae_cv_std = rmse_cv_std = r2_cv_std = np.nan

        # ---------------------- ALMACENAR -----------------------------
        model_metrics[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Explained Variance': ev,
            'CV_MAE': mae_cv,
            'CV_RMSE': rmse_cv,
            'CV_R²': r2_cv,
            'CV_MAE_STD': mae_cv_std,
            'CV_RMSE_STD': rmse_cv_std,
            'CV_R²_STD': r2_cv_std
        }

        # ---------------------- RESIDUALES ----------------------------
        residuals = y_val_raw.ravel() - y_pred
        ax = ax_resid[i] if len(models) > 1 else ax_resid
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.set_title(f'Residuales - {name}')
        ax.set_xlabel('Predicciones')
        ax.set_ylabel('Residuales')
        ax.axhline(y=0, color='r', linestyle='-')
        ax.grid(True, alpha=0.3)

    # -----------------------------------------------------------------
    # BOX-PLOT de errores absolutos (hold-out) y gráficos comparativos
    # -----------------------------------------------------------------
    error_data, model_names = [], []
    for name, preds in all_predictions.items():
        error_data.append(np.abs(y_val_raw.ravel() - preds))
        model_names.append(name)
    ax_err.boxplot(error_data, labels=model_names)
    ax_err.set_title('Comparación de Errores Absolutos')
    ax_err.set_ylabel('Error Absoluto')
    ax_err.grid(True, alpha=0.3)

    fig_resid.tight_layout(); fig_err.tight_layout(); plt.show()

    # -----------------------------------------------------------------
    # DataFrame de resultados
    # -----------------------------------------------------------------
    metrics_df = pd.DataFrame(model_metrics).T

    # Puedes graficar métricas como antes (omitido aquí para brevedad)
    return metrics_df

def save_model_artifacts(preproc, y_scaler, model, artifacts_path="./model_artifacts", model_name="lgbm_reg"):
    """
    Guarda el preprocesador, el escalador y el modelo entrenado en la ruta especificada.
    
    Parameters:
    -----------
    preproc : ColumnTransformer
        El preprocesador de características
    y_scaler : StandardScaler
        El escalador para la variable objetivo
    model : trained model
        El modelo entrenado
    artifacts_path : str or Path
        La ruta donde se guardarán los artefactos
    model_name : str
        El nombre base para el archivo del modelo
    
    Returns:
    --------
    Path
        La ruta donde se guardaron los artefactos
    """
    # Convertir a Path y crear directorio si no existe
    artifacts_dir = Path(artifacts_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar preproc y y_scaler usando cloudpickle para evitar problemas de pickling con funciones locales
    with open(artifacts_dir/"feature_pipeline.joblib", "wb") as f:
        cloudpickle.dump(preproc, f)
    
    with open(artifacts_dir/"y_scaler.joblib", "wb") as f:
        cloudpickle.dump(y_scaler, f)
    
    # Guardar el modelo con joblib
    joblib.dump(model, artifacts_dir/f"{model_name}.joblib")
    
    print(f"✅ Pipeline, scaler y modelo guardados en {artifacts_dir}")
    
    return artifacts_dir

def evaluate_regression_model(model, y_scaler, X_val_trans, y_val_raw, print_results=True):
    """
    Evaluate a regression model using scaled validation data and return metrics.
    
    Parameters:
    -----------
    model : trained model instance
        The trained model with a predict method
    y_scaler : scaler instance
        Scaler used to transform target variable
    X_val_trans : array-like
        Transformed validation features
    y_val_raw : array-like
        Raw (unscaled) validation target values
    print_results : bool, default=True
        Whether to print the evaluation metrics
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Get predictions and inverse transform
    y_pred_scaled = model.predict(X_val_trans)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    
    # Calculate metrics
    mae = mean_absolute_error(y_val_raw, y_pred)
    mse = mean_squared_error(y_val_raw, y_pred)
    r2 = r2_score(y_val_raw, y_pred)
    
    # Store metrics
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'y_pred': y_pred  # Optional: return predictions for further analysis
    }
    
    # Print results if requested
    if print_results:
        print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.3f}")
    
    return metrics

def inferencia_dataset(path_csv, preproc_path, y_scaler_path):
    preproc = joblib.load(preproc_path)
    y_scaler = joblib.load(y_scaler_path)

    df = pl.read_csv(path_csv)
    X = df.drop("PRECIOVTAX")
    y = df.select("PRECIOVTAX").to_numpy().reshape(-1, 1)

    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["TRIMESTRE"]
    )

    X_train_trans = preproc.transform(X_train)
    X_val_trans = preproc.transform(X_val)
    y_train = y_scaler.transform(y_train_raw).ravel()

    return X_train_trans, X_val_trans, y_train, y_val_raw