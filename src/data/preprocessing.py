"""Módulo de preprocesamiento de datos"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import logging
import pickle
from typing import Tuple, Dict, Any, List
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Clase para preprocesamiento de datos"""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.transformers: Dict[str, Any] = {}
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """Ajusta transformadores a los datos.
        
        Detecta columnas numéricas y categóricas, y ajusta los transformadores
        de normalización y codificación según la configuración.
        """
        preprocessing_cfg = self.config.get('preprocessing', {})
        norm_method = preprocessing_cfg.get('normalization', {}).get('method', 'standard')
        enc_method = preprocessing_cfg.get('encoding', {}).get('method', 'onehot')

        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Ajustar scaler para columnas numéricas
        if self._numeric_cols:
            if norm_method == 'minmax':
                num_scaler = MinMaxScaler()
            else:
                num_scaler = StandardScaler()
            num_scaler.fit(df[self._numeric_cols])
            self.transformers['scaler'] = num_scaler
            logger.info(f"Scaler '{norm_method}' ajustado en {len(self._numeric_cols)} columnas numéricas")

        # Ajustar encoder para columnas categóricas
        if self._categorical_cols:
            if enc_method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(df[self._categorical_cols])
            else:
                encoder = {}
                for col in self._categorical_cols:
                    le = LabelEncoder()
                    le.fit(df[col].astype(str))
                    encoder[col] = le
            self.transformers['encoder'] = encoder
            logger.info(f"Encoder '{enc_method}' ajustado en {len(self._categorical_cols)} columnas categóricas")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformaciones a los datos.
        
        Requiere que fit() haya sido llamado previamente.
        """
        if not self.transformers:
            raise RuntimeError("El preprocesador no ha sido ajustado. Llame a fit() primero.")

        preprocessing_cfg = self.config.get('preprocessing', {})
        enc_method = preprocessing_cfg.get('encoding', {}).get('method', 'onehot')

        result = df.copy()

        # Aplicar normalización
        if 'scaler' in self.transformers and self._numeric_cols:
            available_num = [c for c in self._numeric_cols if c in result.columns]
            if available_num:
                result[available_num] = self.transformers['scaler'].transform(result[available_num])

        # Aplicar codificación
        if 'encoder' in self.transformers and self._categorical_cols:
            available_cat = [c for c in self._categorical_cols if c in result.columns]
            if available_cat:
                if enc_method == 'onehot':
                    encoder: OneHotEncoder = self.transformers['encoder']
                    encoded_arr = encoder.transform(result[available_cat])
                    encoded_cols = encoder.get_feature_names_out(available_cat)
                    encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=result.index)
                    result = result.drop(columns=available_cat)
                    result = pd.concat([result, encoded_df], axis=1)
                else:
                    for col in available_cat:
                        le: LabelEncoder = self.transformers['encoder'][col]
                        result[col] = le.transform(result[col].astype(str))

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta y transforma en un solo paso."""
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Revierte las transformaciones aplicadas por transform().

        Revierte la normalización de variables numéricas y la codificación
        de variables categóricas. Requiere que fit() haya sido llamado previamente.
        """
        if not self.transformers:
            raise RuntimeError("El preprocesador no ha sido ajustado. Llame a fit() primero.")

        preprocessing_cfg = self.config.get('preprocessing', {})
        enc_method = preprocessing_cfg.get('encoding', {}).get('method', 'onehot')

        result = df.copy()

        # Revertir codificación categórica
        if 'encoder' in self.transformers and self._categorical_cols:
            if enc_method == 'onehot':
                encoder: OneHotEncoder = self.transformers['encoder']
                # Determinar qué columnas one-hot están presentes en el df
                available_cat = [c for c in self._categorical_cols if c in df.columns]
                ohe_feature_names = encoder.get_feature_names_out(self._categorical_cols)
                present_ohe_cols = [c for c in ohe_feature_names if c in result.columns]

                if present_ohe_cols:
                    # Reconstruir array completo de one-hot (rellenar columnas faltantes con 0)
                    ohe_array = np.zeros((len(result), len(ohe_feature_names)))
                    for i, col in enumerate(ohe_feature_names):
                        if col in result.columns:
                            ohe_array[:, i] = result[col].values

                    # Invertir one-hot encoding
                    original_values = encoder.inverse_transform(ohe_array)
                    original_df = pd.DataFrame(
                        original_values,
                        columns=self._categorical_cols,
                        index=result.index
                    )

                    # Eliminar columnas one-hot y agregar columnas originales
                    result = result.drop(columns=present_ohe_cols)
                    result = pd.concat([result, original_df], axis=1)
            else:
                # Label encoding: revertir con inverse_transform de cada LabelEncoder
                for col in self._categorical_cols:
                    if col in result.columns:
                        le: LabelEncoder = self.transformers['encoder'][col]
                        result[col] = le.inverse_transform(result[col].astype(int))

        # Revertir normalización numérica
        if 'scaler' in self.transformers and self._numeric_cols:
            available_num = [c for c in self._numeric_cols if c in result.columns]
            if available_num:
                result[available_num] = self.transformers['scaler'].inverse_transform(
                    result[available_num]
                )

        return result

    def save_transformers(self, path: str) -> None:
        """Serializa los transformadores ajustados a un archivo pickle.

        Args:
            path: Ruta del archivo donde se guardarán los transformadores.

        Raises:
            RuntimeError: Si no se ha llamado a fit() previamente.
        """
        if not self.transformers:
            raise RuntimeError("No hay transformadores ajustados. Llame a fit() primero.")

        payload = {
            'transformers': self.transformers,
            '_numeric_cols': self._numeric_cols,
            '_categorical_cols': self._categorical_cols,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        logger.info(f"Transformadores guardados en '{path}'")

    def load_transformers(self, path: str) -> None:
        """Carga transformadores desde un archivo pickle.

        Después de cargar, el preprocesador puede llamar a transform() e
        inverse_transform() sin necesidad de llamar a fit() nuevamente.

        Args:
            path: Ruta del archivo pickle con los transformadores.

        Raises:
            FileNotFoundError: Si el archivo no existe.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de transformadores: '{path}'")

        with open(file_path, 'rb') as f:
            payload = pickle.load(f)

        self.transformers = payload['transformers']
        self._numeric_cols = payload['_numeric_cols']
        self._categorical_cols = payload['_categorical_cols']
        logger.info(f"Transformadores cargados desde '{path}'")

    def create_performance_categories(self, df: pd.DataFrame, 
                                      column: str = 'final_grade',
                                      bins: list = [0, 60, 70, 80, 90, 100],
                                      labels: list = ['Bajo', 'Medio-Bajo', 'Medio', 
                                                     'Medio-Alto', 'Alto']) -> pd.DataFrame:
        """Crea categorías de rendimiento"""
        df = df.copy()
        df['performance_category'] = pd.cut(df[column], bins=bins, labels=labels)
        return df
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepara features para modelado"""
        
        # Seleccionar features según configuración
        feature_config = self.config['features']
        digital_features = feature_config['digital']
        academic_features = feature_config['academic'][:-1]  # excluir final_grade
        lifestyle_features = feature_config['lifestyle']
        
        all_features = digital_features + academic_features + lifestyle_features
        available_features = [f for f in all_features if f in df.columns]
        
        X = df[available_features].copy()
        y_reg = df['final_grade'].copy()
        y_prod = df['productivity_score'].copy()
        
        self.feature_names = available_features
        
        logger.info(f"Features preparadas: {available_features}")
        return X, y_reg, y_prod
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   task: str = 'regression') -> Dict[str, Any]:
        """Divide datos en entrenamiento y prueba"""
        
        test_size = self.config['training']['test_size']
        random_state = self.config['training']['random_state']
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Datos divididos: Train={len(X_train)}, Test={len(X_test)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
        
    def prepare_classification_data(self, df: pd.DataFrame, 
                                   target_col: str = 'performance_category') -> Dict[str, Any]:
        """Prepara datos para clasificación"""
        
        X, _, _ = self.prepare_features(df)
        y = df[target_col].copy()
        
        # Codificar target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Escalar y dividir
        X_scaled = self.scaler.fit_transform(X)
        test_size = self.config['training']['test_size']
        random_state = self.config['training']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'class_names': self.label_encoder.classes_
        }
        
    def save_preprocessors(self, path: Path):
        """Guarda preprocesadores"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }, path / 'preprocessors.pkl')
        logger.info(f"✅ Preprocesadores guardados")