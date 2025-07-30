# Credit Risk Modeling - Probability of Default (PD) Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Overview

This project focuses on developing robust **Probability of Default (PD)** models for credit risk assessment using machine learning techniques. The project implements a comprehensive end-to-end pipeline for credit default prediction using the Home Credit dataset, featuring advanced ensemble methods, feature engineering, and model validation techniques.

### Key Features

- **Multi-level Stacking Architecture**: Three-tier ensemble combining XGBoost, LightGBM, and CatBoost
- **Advanced Feature Engineering**: SHAP-based feature selection and target encoding
- **Comprehensive Model Validation**: Performance metrics, calibration analysis, and drift detection
- **Production-Ready Pipeline**: Modular design with proper logging and configuration management
- **Extensive Performance Analysis**: AUC, Gini, KS statistics, PSI, and calibration metrics

### Dataset

The project uses the **Home Credit Default Risk** dataset from Kaggle, which includes:
- Application data (train/test)
- Bureau and bureau balance information
- Previous applications history
- Credit card balance details
- POS cash balance records
- Installments payments data

### Model Performance

- **Best Model**: L2 Logistic Regression (Multi-level stacking)
- **Cross-validation AUC**: ~0.7752 (Good performance)
- **Validation Strategy**: Out-of-fold cross-validation with stacking
- **Feature Selection**: Multi-table feature engineering with target encoding

## Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pd_modeling_project
```

2. Create and activate virtual environment:
```bash
python -m venv credit_env
source credit_env/bin/activate  # On Windows: credit_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

### Running the Pipeline

Execute the complete modeling pipeline:

```bash
python main.py
```

Or run individual components:

```bash
# Data processing only
python main.py --stage data_processing

# Feature engineering
python main.py --stage feature_engineering

# Model training
python main.py --stage model_training
```

### Model Validation

Generate comprehensive performance reports:

```bash
# Run validation notebook
jupyter notebook notebooks/validation_metrics.ipynb

# Generate calibration analysis
python src/validation/calibration_analysis.py

# Perform drift analysis
python src/validation/drift_analysis.py
```

## Project Architecture

### Core Components

1. **Data Pipeline** (`src/data_pipeline/`):
   - Data loaders for all Home Credit tables
   - Automated data validation and quality checks
   - Memory-efficient processing for large datasets

2. **Feature Engineering** (`src/processing/`):
   - Target encoding with cross-validation
   - Missing value imputation strategies
   - Feature selection using SHAP values

3. **Modeling** (`src/modeling/`):
   - Three-level stacking ensemble
   - Hyperparameter optimization with Optuna
   - Cross-validation and out-of-fold predictions

4. **Validation** (`src/validation/`):
   - Comprehensive performance metrics
   - Model calibration analysis
   - Population stability index (PSI) monitoring

### Model Ensemble Architecture

```
Level 1: Base Models
├── XGBoost Classifier
├── LightGBM Classifier
└── CatBoost Classifier

Level 2: Meta Models
├── Extra Trees Classifier
└── Logistic Regression

Level 3: Final Ensemble
└── Extra Trees Classifier (Final Predictions)
```

## Project Organization

```
├── LICENSE                    <- Open-source license
├── Makefile                   <- Makefile with convenience commands
├── README.md                  <- The top-level README for developers
├── main.py                    <- Main pipeline execution script
├── requirements.txt           <- Package dependencies
├── pyproject.toml            <- Project configuration and metadata
├── .gitignore                <- Git ignore file
│
├── data/                      <- Data storage (add to .gitignore)
│   ├── external/             <- Data from third party sources
│   ├── interim/              <- Intermediate transformed data
│   ├── processed/            <- Final canonical datasets for modeling
│   └── raw/                  <- Original immutable data dump
│       ├── application_train.csv        <- Main training dataset
│       ├── application_test.csv         <- Test dataset for submissions
│       ├── bureau.csv                   <- Credit bureau data
│       ├── bureau_balance.csv           <- Monthly bureau balances
│       ├── credit_card_balance.csv      <- Credit card balance data
│       ├── installments_payments.csv    <- Installments payment history
│       ├── POS_CASH_balance.csv         <- POS and cash balance data
│       └── previous_application.csv     <- Previous loan applications
│
├── docs/                      <- Project documentation (MkDocs)
│   ├── mkdocs.yml            <- MkDocs configuration
│   ├── README.md             <- Documentation README
│   ├── .cache/               <- MkDocs cache directory
│   └── docs/                 <- Documentation source files
│
├── models/                    <- Trained models and predictions
│   ├── l1_stacking/          <- Level 1 model outputs
│   ├── l2_stacking/          <- Level 2 meta-models
│   └── l3_stacking/          <- Final ensemble models
│
├── notebooks/                 <- Jupyter notebooks for analysis
│   ├── EDA.ipynb             <- Exploratory Data Analysis
│   ├── pipeline_execution.ipynb         <- Pipeline execution demo
│   ├── validation_metrics.ipynb         <- Performance metrics analysis
│   ├── calibration_analysis.ipynb       <- Model calibration analysis
│   ├── stability_analysis.ipynb         <- Population stability monitoring
│   ├── vintage_analysis.ipynb           <- Vintage analysis
│   └── temp_results/         <- Temporary notebook results
│
├── references/               <- Data dictionaries and explanatory materials
│
├── src/                      <- Source code for the project
│   ├── data_pipeline/        <- Data loading and processing
│   ├── modeling/             <- Machine learning models
│   ├── processing/           <- Data preprocessing modules
│   ├── utils/                <- Utility functions
│   └── validation/           <- Model validation and monitoring
│
├── tests/                    <- Unit and integration tests
│
└── validation_results/       <- Model validation outputs
    ├── calibration_results/  <- Calibration analysis results
    ├── metrics_results/      <- Performance metrics
    ├── metrics_explanation/  <- Metrics explanation documents
    ├── stability_results/    <- Stability analysis results
    ├── vintage_results/      <- Vintage analysis results
```


## Key Metrics and Performance

### Model Performance Metrics

| Metric | Description | Target Range | Current Performance |
|--------|-------------|--------------|-------------------|
| **AUC** | Area Under ROC Curve | 0.5-1.0 | ~0.7750 |
| **Gini** | 2×AUC-1, ranking quality | 0.0-1.0 | ~0.5500 |
| **KS** | Kolmogorov-Smirnov statistic | 0.0-1.0 | ~0.4090 |
| **PSI** | Population Stability Index | <0.1 stable | <0.05 |
| **Brier Score** | Probabilistic accuracy | 0.0-1.0 (lower better) | ~0.0670 |

### Validation Framework

- **Cross-Validation**: Out-of-fold predictions with stacking
- **Performance Metrics**: AUC, Gini, KS, Brier Score, Hosmer-Lemeshow
- **Model Comparison**: Multi-level ensemble evaluation  
- **Calibration Analysis**: Probability calibration assessment
- **Stability Monitoring**: Population Stability Index (PSI) tracking

## Technical Implementation

### Feature Engineering Pipeline

1. **Data Integration**: Merge multiple Home Credit tables
2. **Missing Value Treatment**: Domain-specific imputation strategies
3. **Target Encoding**: Cross-validated encoding for categorical variables
4. **Feature Selection**: SHAP-based importance ranking
5. **Feature Scaling**: Standardization for linear models

### Model Training Process

```python
# Example usage
from src.modeling.stacking import run_l1_stacking, run_l2_stacking, run_l3_stacking

# Level 1: Base models
l1_predictions = run_l1_stacking(X_train, y_train, X_test)

# Level 2: Meta models
l2_predictions = run_l2_stacking(l1_predictions['train'], y_train, l1_predictions['test'])

# Level 3: Final ensemble
final_predictions = run_l3_stacking(l2_predictions['train'], y_train, l2_predictions['test'])
```

### Performance Monitoring

```python
from src.validation.performance_metrics import get_perf_report, plot_pr_curve

# Generate comprehensive metrics
metrics = get_perf_report(y_true, y_score)

# Create performance visualizations
plot_pr_curve(y_true, y_score, filename="reports/figures/pr_curve.png")
```

## Development Workflow

### Code Quality

- **Linting**: Ruff for code formatting and style
- **Type Hints**: Full type annotations for better IDE support
- **Documentation**: Comprehensive docstrings and API documentation
- **Testing**: Unit tests for all core components

### Configuration Management

All model parameters and settings are centralized in `src/config/`:

```python
# Model hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Feature selection parameters
FEATURE_SELECTION = {
    'shap_top_features': 300,
    'variance_threshold': 0.01,
    'correlation_threshold': 0.95
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
ruff check src/

# Build documentation
mkdocs serve
```

## Deployment

### Model Serving

The trained models can be deployed using various methods:

```python
# Load trained model
import pickle
with open('models/l3_stacking/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict_proba(X_new)[:, 1]
```

### Batch Scoring

```bash
# Score a batch of applications
python src/modeling/predict.py --input data/new_applications.csv --output predictions.csv
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Home Credit Group** for providing the dataset
- **Kaggle Community** for insights and methodologies
- **Open Source Contributors** for the machine learning libraries used

## Contact

For questions or suggestions, please open an issue or contact the development team.

## Key Notebooks and Analysis

### 📊 Performance Analysis
- **`validation_metrics.ipynb`**: Comprehensive model performance metrics with AUC, Gini, KS, Brier Score analysis
- **`calibration_analysis.ipynb`**: Model calibration assessment and reliability analysis  
- **`stability_analysis.ipynb`**: Population stability monitoring and drift detection

### 🔧 Model Development
- Multi-level stacking ensemble (L1 → L2 → L3)
- Advanced feature engineering with target encoding
- Cross-validated performance evaluation

---

*This project follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure.*

