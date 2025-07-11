#!/usr/bin/env python
"""
Pipeline de procesamiento de datos del Titanic

Este módulo implementa un pipeline de procesamiento de datos para el dataset del Titanic,
incluyendo ingeniería de características, preprocesamiento y entrenamiento de modelo.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PipelineConfig:
    """Parámetros de configuración para el pipeline de datos."""
    
    datasets_dir: Path = Path("datasets")
    raw_data_file: str = "raw-data.csv"
    train_data_file: str = "train.csv"
    test_data_file: str = "test.csv"
    test_size: float = 0.2
    rare_label_threshold: float = 0.05
    random_seed: int = 404


class DataPreprocessor:
    """Maneja tareas de preprocesamiento e ingeniería de características."""

    def __init__(self, config: PipelineConfig):
        """Inicializa el preprocesador con parámetros de configuración."""
        self.config = config
        self.num_vars: List[str] = []
        self.cat_vars: List[str] = []
        self.rare_labels: Dict[str, Set[str]] = {}
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()

    @staticmethod
    def _get_first_cabin(row: str) -> Optional[str]:
        """Extrae la primera cabina de una cadena de cabinas."""
        try:
            return str(row).split()[0]
        except (AttributeError, IndexError):
            return None

    @staticmethod
    def _get_title(name: str) -> str:
        """Extrae el título de un nombre de pasajero."""
        title_map = {
            "Mrs": "Mrs",
            "Mr": "Mr",
            "Miss": "Miss",
            "Master": "Master"
        }
        
        for title, mapped_title in title_map.items():
            if title in str(name):
                return mapped_title
        return "Other"

    @staticmethod
    def _extract_letter_from_cabin(cabin: Any) -> Optional[str]:
        """Extrae la parte de letra de un código de cabina."""
        if isinstance(cabin, str):
            import re
            letters = re.findall("[a-zA-Z]+", cabin)
            return "".join(letters) if letters else None
        return None

    def _find_rare_labels(self, data: pd.DataFrame, column: str) -> Set[str]:
        """Encuentra etiquetas de categoría raras en una columna."""
        value_counts = data[column].value_counts(normalize=True)
        rare_mask = value_counts < self.config.rare_label_threshold
        return set(value_counts.loc[rare_mask].index.astype(str))

    def prepare_raw_data(self, url: str) -> None:
        """Descarga y prepara el dataset crudo."""
        # Crear directorio de datasets si no existe
        self.config.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Descargar y procesar datos
        data = pd.read_csv(url)
        
        # Manejar valores faltantes y conversiones de tipo
        data.replace("?", np.nan, inplace=True)
        data["age"] = pd.to_numeric(data["age"], errors="coerce")
        data["fare"] = pd.to_numeric(data["fare"], errors="coerce")
        
        # Ingeniería de características
        data["cabin"] = data["cabin"].apply(self._get_first_cabin)
        data["title"] = data["name"].apply(self._get_title)
        
        # Eliminar columnas innecesarias
        drop_cols = ["boat", "body", "home.dest", "ticket", "name"]
        data = data.drop(columns=drop_cols)
        
        # Guardar datos procesados
        data.to_csv(self.config.datasets_dir / self.config.raw_data_file, index=False)

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide los datos en conjuntos de entrenamiento y prueba."""
        X = data.drop("survived", axis=1)
        y = data["survived"]
        
        split_result = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_seed
        )
        X_train, X_test, y_train, y_test = split_result
        return cast(Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], (X_train, X_test, y_train, y_test))

    def identify_variable_types(self, data: pd.DataFrame) -> None:
        """Identifica variables numéricas y categóricas."""
        self.num_vars = [
            col for col in data.columns
            if data[col].dtype != object and col != "survived"
        ]
        self.cat_vars = [
            col for col in data.columns
            if data[col].dtype == object
        ]

    def process_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Procesa características para entrenamiento del modelo."""
        # Procesar características numéricas
        for var in self.num_vars:
            X_train[f"{var}_nan"] = X_train[var].isnull().astype(int)
            X_test[f"{var}_nan"] = X_test[var].isnull().astype(int)

        # Procesar características categóricas
        X_train["cabin"] = X_train["cabin"].apply(self._extract_letter_from_cabin)
        X_test["cabin"] = X_test["cabin"].apply(self._extract_letter_from_cabin)

        # Llenar valores faltantes
        X_train[self.cat_vars] = X_train[self.cat_vars].fillna("missing")
        X_test[self.cat_vars] = X_test[self.cat_vars].fillna("missing")

        # Manejar valores faltantes numéricos
        self.imputer.fit(X_train[self.num_vars])
        X_train[self.num_vars] = self.imputer.transform(X_train[self.num_vars])
        X_test[self.num_vars] = self.imputer.transform(X_test[self.num_vars])

        # Manejar categorías raras
        for col in self.cat_vars:
            self.rare_labels[col] = self._find_rare_labels(X_train, col)
            X_train[col] = np.where(
                X_train[col].isin(list(self.rare_labels[col])),
                "Rare",
                X_train[col]
            )
            X_test[col] = np.where(
                X_test[col].isin(list(self.rare_labels[col])),
                "Rare",
                X_test[col]
            )

        # Codificación one-hot
        X_train_encoded = pd.get_dummies(X_train, columns=self.cat_vars, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, columns=self.cat_vars, drop_first=True)

        # Asegurar mismas columnas en train y test
        missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
        for col in missing_cols:
            X_test_encoded[col] = 0

        # Alinear columnas
        X_test_encoded = X_test_encoded[X_train_encoded.columns]

        # Escalar características
        self.scaler.fit(X_train_encoded)
        return (
            self.scaler.transform(X_train_encoded),
            self.scaler.transform(X_test_encoded)
        )


class ModelTrainer:
    """Maneja entrenamiento y evaluación del modelo."""

    def __init__(self, config: PipelineConfig):
        """Inicializa el entrenador del modelo."""
        self.config = config
        self.model: Optional[BaseEstimator] = None

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Entrena el modelo y evalúa el rendimiento."""
        self.model = LogisticRegression(
            C=0.0005,
            class_weight="balanced",
            random_state=self.config.random_seed
        )
        self.model.fit(X_train, y_train)

        metrics = {}
        for name, (X, y) in {"train": (X_train, y_train), "test": (X_test, y_test)}.items():
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1]
            
            metrics[name] = {
                "roc_auc": roc_auc_score(y, y_prob),
                "accuracy": accuracy_score(y, y_pred)
            }

        return metrics


def main() -> None:
    """Función principal para ejecutar el pipeline."""
    # Inicializar configuración
    config = PipelineConfig()

    # Inicializar componentes del pipeline
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config)

    # Descargar y preparar datos
    url = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
    preprocessor.prepare_raw_data(url)

    # Cargar datos preparados
    data = pd.read_csv(config.datasets_dir / config.raw_data_file)

    # Dividir datos
    X_train, X_test, y_train, y_test = preprocessor.split_data(data)

    # Guardar datos divididos
    X_train.to_csv(config.datasets_dir / config.train_data_file, index=False)
    X_test.to_csv(config.datasets_dir / config.test_data_file, index=False)

    # Identificar tipos de variables
    preprocessor.identify_variable_types(X_train)

    # Procesar características
    X_train_processed, X_test_processed = preprocessor.process_features(X_train, X_test)

    # Entrenar y evaluar modelo
    metrics = trainer.train_model(X_train_processed, y_train, X_test_processed, y_test)

    # Imprimir resultados
    for dataset, dataset_metrics in metrics.items():
        print(f"\n{dataset.upper()} Métricas:")
        for metric_name, value in dataset_metrics.items():
            print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
