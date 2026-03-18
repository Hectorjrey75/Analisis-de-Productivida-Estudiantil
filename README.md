# Student Productivity Analysis Introducción

El Sistema de Análisis de Productividad Estudiantil es una plataforma de análisis de datos que correlaciona hábitos digitales con rendimiento académico, desarrolla modelos predictivos de productividad, identifica factores clave de éxito académico y provee recomendaciones personalizadas basadas en datos para mejorar el desempeño estudiantil.

## Glosario

- **Sistema**: El Sistema de Análisis de Productividad Estudiantil
- **Hábito_Digital**: Patrón de comportamiento relacionado con el uso de dispositivos digitales y aplicaciones
- **Rendimiento_Académico**: Medida cuantitativa del desempeño estudiantil (calificaciones, promedios, logros)
- **Modelo_Predictivo**: Algoritmo de machine learning entrenado para predecir productividad estudiantil
- **Factor_Clave**: Variable que tiene correlación significativa con el éxito académico
- **Recomendación**: Sugerencia personalizada basada en análisis de datos para mejorar productividad
- **Dataset_Estudiantil**: Conjunto de datos que contiene información de hábitos digitales y rendimiento académico
- **Correlación**: Medida estadística de la relación entre dos variables
- **Métrica_Productividad**: Indicador cuantificable de productividad estudiantil

Analiza cómo los hábitos digitales afectan el rendimiento académico y la productividad estudiantil. Entrena modelos de aprendizaje automático (regresión lineal, bosque aleatorio, potenciación de gradiente) con un conjunto de datos de 20 000 filas y genera recomendaciones personalizadas.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python pipeline.py --config config/config.yaml
```

Run specific stages:

```bash
# Data ingestion + preprocessing + feature engineering + training
python pipeline.py --stages data,preprocess,features,train

# Full pipeline with all stages
python pipeline.py --stages data,preprocess,features,correlation,train,recommend,visualize,export

# Custom output directory
python pipeline.py --output-dir results/
```

Available stages (in order): `data` → `preprocess` → `features` → `correlation` → `train` → `recommend` → `visualize` → `export`

## Project Structure

```
├── pipeline.py              # Main pipeline script (CLI entry point)
├── config/
│   └── config.yaml          # Project configuration
├── data/
│   ├── raw/                 # Input CSV dataset
│   └── processed/           # Pipeline outputs (CSV, JSON, plots)
├── src/
│   ├── data/                # Data ingestion & preprocessing
│   ├── features/            # Feature engineering
│   ├── analysis/            # Correlation analysis
│   ├── models/              # Training, prediction, monitoring
│   ├── recommendations/     # Recommendation engine
│   ├── visualization/       # Plots & dashboards
│   └── export/              # Data & model export
├── tests/                   # Test suite (pytest + hypothesis)
└── notebooks/               # Jupyter notebooks for exploration
```

## Configuration

The pipeline is controlled by `config/config.yaml`. Key sections:

**data** — paths and required columns:
```yaml
data:
  raw_path: "data/raw/student_productivity_distraction_dataset_20000.csv"
  processed_path: "data/processed/"
  imputation:
    strategy: "median"   # mean | median | mode | forward_fill | drop
```

**models** — which models to train and their hyperparameters:
```yaml
models:
  types:
    - linear_regression
    - random_forest
    - gradient_boosting
  regression:
    random_forest:
      n_estimators: 100
      max_depth: 10
      random_state: 42
  production_threshold:
    r_squared: 0.7
```

**recommendations** — optimal ranges used to generate student advice:
```yaml
recommendations:
  n_recommendations: 3
  optimal_ranges:
    study_hours_per_day: [6.0, 9.0]
    sleep_hours: [7.0, 9.0]
    phone_usage_hours: [0.0, 3.0]
    social_media_hours: [0.0, 2.0]
    exercise_minutes: [30, 60]
```

**preprocessing** — normalization, encoding, and outlier strategy:
```yaml
preprocessing:
  normalization:
    method: "standard"   # standard | minmax
  encoding:
    method: "onehot"     # onehot | label
  outliers:
    strategy: "clip"     # clip | remove | winsorize
    threshold: 3.0
```

**correlation** — method and significance level:
```yaml
correlation:
  method: "pearson"      # pearson | spearman | kendall
  significance_level: 0.95
```

**export** — output formats and checksums:
```yaml
export:
  model_formats:
    - pickle
    - onnx
  include_metadata: true
  compute_checksums: true
```

## Running Tests

```bash
pytest tests/
```
