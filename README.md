# ML in Agriculture Data for Crops Yield

## Project Overview

This project implements a comprehensive machine learning pipeline for predicting crops yield based on meteorological data, agricultural practices, and field characteristics. The system uses multiple advanced algorithms including gradient boosting, neural networks, and ensemble methods to provide accurate yield predictions.

## Project Features

- **Multi-Model Approach**: 10 different machine learning models (5 classification, 5 regression)
- **Advanced Feature Engineering**: Weather data integration, one-hot encoding, min-max scaling
- **Comprehensive Evaluation**: Cross-validation, multiple metrics, feature importance analysis
- **Real-time Data Integration**: IMS (Israel Meteorological Service) API integration
- **Robust Preprocessing**: Missing data handling, outlier detection, data validation

## Project Structure

```
wheat_yield_prediction_FP_EMI/
├── data/                          # Data storage
│   ├── ims_data/                  # Meteorological data
│   │   ├── station_raw_data/      # Raw weather station data
│   │   ├── station_processed_and_aggregated_data/  # Processed weather data
│   │   └── station_complete_data/ # Complete weather datasets
│   ├── models_sets/               # Training and testing datasets
│   │   ├── train/                 # Training datasets
│   │   └── test/                  # Testing datasets
│   ├── results/                   # Model results and visualizations
│   ├── performance/               # Performance metrics and summaries
│   ├── FE/                        # Feature engineering outputs
│   └── data_understanding/        # Data analysis and statistics
├── pys/                          # Python source code
│   ├── data_preperation/         # Data preprocessing modules
│   ├── ims/                      # Meteorological data handling
│   └── models/                   # Machine learning models
│       ├── predictors/           # Model implementations
│       ├── scores/               # Evaluation metrics
│       └── wraps/                # Model wrappers and pipelines
├── plots/                        # Generated plots and visualizations
└── old_versions/                 # Previous versions of code
```

## 🤖 Machine Learning Models

### Classification Models
- **DecisionTreeClassifier**: Tree-based classification with configurable depth
- **RandomForestClassifier**: Ensemble of decision trees for robust classification
- **LightGBMClassifier**: Gradient boosting with optimized tree construction
- **HistGradientBoostingClassifier**: Histogram-based gradient boosting

### Regression Models
- **DecisionTreeRegressor**: Tree-based regression
- **RandomForestRegressor**: Ensemble regression with multiple trees
- **LightGBMRegressor**: Gradient boosting for regression tasks
- **HistGradientBoostingRegressor**: Histogram-based gradient boosting regression
- **SVR**: Support Vector Regression with multiple kernels
- **MLPRegressor**: Multi-layer perceptron neural network

## Model Parameters

### Tree-Based Models
```python
# Decision Trees & Random Forest
max_depth: [2-20, None]           # Maximum tree depth
min_samples_split: [2-20]         # Minimum samples to split node
min_samples_leaf: [1-15]          # Minimum samples per leaf
n_estimators: [50-200]            # Number of trees (Random Forest)
```

### Gradient Boosting Models
```python
# LightGBM & HistGradientBoosting
learning_rate: [0.1-0.5]          # Learning rate for boosting
max_depth: [2-20, None]           # Maximum tree depth
num_leaves: [31-50]               # Maximum leaves per tree
min_samples_leaf: [1-7]           # Minimum samples per leaf
```

### Neural Network (MLP)
```python
# Multi-Layer Perceptron
hidden_layer_sizes: [(30,), (50,), ..., (150,)]  # Network architecture
activation: ['relu', 'tanh', 'logistic']         # Activation functions
solver: ['adam', 'sgd']                          # Optimization algorithm
alpha: [0.0001, 0.001, 0.01, 0.1, 1]           # L2 regularization
learning_rate: ['constant', 'adaptive']         # Learning rate schedule
```

### Support Vector Regression
```python
# SVR
C: [0.01-200]                     # Regularization parameter
epsilon: [0.001-10]               # Epsilon in epsilon-SVR
kernel: ['linear', 'poly', 'rbf', 'sigmoid']  # Kernel functions
```

## 🔧 Installation & Setup

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Required packages
pip install pandas numpy scikit-learn matplotlib seaborn
pip install lightgbm openpyxl xlsxwriter
pip install requests  # For IMS API
```

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd wheat_yield_prediction_FP_EMI

# Install dependencies
pip install -r requirements.txt

# Set up data directories
mkdir -p data/ims_data/station_raw_data
mkdir -p data/models_sets/train
mkdir -p data/models_sets/test
mkdir -p data/results
```

## 📊 Data Sources

### Agricultural Data
- **Field Data**: Crop types, irrigation methods, fertilization, field sizes
- **Temporal Data**: Planting dates, growing seasons, harvest information
- **Geographic Data**: Field locations, regions, soil characteristics

### Meteorological Data (IMS)
- **Temperature**: Daily min/max temperatures
- **Precipitation**: Rainfall amounts and patterns
- **Humidity**: Relative humidity measurements
- **Wind**: Wind speed and direction
- **Data Period**: 2020-2024

### Data Preprocessing
- **Feature Engineering**: Weather aggregation by weeks, seasonal patterns
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: Min-max normalization for numerical features
- **Missing Data**: Advanced handling with NaN conversion

## Usage

### 1. Data Preparation
```python
# Run data preprocessing
python pys/data_preperation/feature_extraction.py

# Extract meteorological features
python pys/ims/ims_data_extraction.py
```

### 2. Model Training
```python
# Train classification models
python pys/models/predictors/dt_classifier.py
python pys/models/predictors/rf_classifier.py
python pys/models/predictors/lgb_classifier.py
python pys/models/predictors/hgb_classifier.py

# Train regression models
python pys/models/predictors/dt_regressor.py
python pys/models/predictors/rf_regressor.py
python pys/models/predictors/lgb_regressor.py
python pys/models/predictors/hgb_regressor.py
python pys/models/predictors/svr.py
python pys/models/predictors/mlp_regressor.py
```

### 3. Model Evaluation
```python
# Generate performance summaries
python pys/models/scores/model_conclusion.py

# Analyze feature importance
python pys/models/scores/feature_importance_analysis.py
```

## 📊 Model Performance

### Evaluation Metrics
- **Classification**: Accuracy, F1-Score, ROC AUC
- **Regression**: R², Adjusted R², MSE, RMSE
- **Cross-Validation**: 5-fold cross-validation with stratified sampling

### Feature Selection
- **Mutual Information**: Statistical dependency measurement
- **Permutation Importance**: Model-based feature importance
- **Information Gain**: Entropy-based feature selection

## 🔍 Key Features

### Advanced Preprocessing
- **Missing Data Handling**: Native support in tree-based models
- **Feature Scaling**: Min-max normalization for all numerical features
- **Categorical Encoding**: One-hot encoding for Hebrew categorical variables
- **Weather Integration**: Weekly aggregation of meteorological data

### Model Optimization
- **Hyperparameter Tuning**: Grid search and randomized search
- **Cross-Validation**: Robust evaluation across multiple folds
- **Feature Selection**: Automatic selection of most important features
- **Ensemble Methods**: Combination of multiple models for better performance

### Data Quality
- **Outlier Detection**: Statistical methods for identifying anomalies
- **Data Validation**: Comprehensive checks for data integrity
- **Temporal Analysis**: Time-series analysis of weather patterns
- **Geographic Integration**: Location-based weather data matching

## 📈 Results & Visualizations

### Generated Outputs
- **Model Performance**: Excel files with detailed metrics
- **Feature Importance**: Plots showing most important predictors
- **Predictions**: Detailed prediction results for each model
- **Cross-Validation**: Stability analysis across different data splits

### Visualization Types
- **Feature Importance Plots**: Bar charts of feature significance
- **Performance Comparisons**: Model comparison charts
- **Prediction vs Actual**: Scatter plots of predictions
- **Class Distribution**: For classification tasks

## Configuration

### Model Parameters
All model parameters are configurable through the respective configuration files in each model directory.

### Data Paths
Update data paths in the configuration files to match your local setup:
```python
# Example configuration
data_dir = r'/path/to/your/data'
results_dir = r'/path/to/results'
```

## 📝 File Descriptions

### Core Modules
- `feature_extraction.py`: Main feature engineering pipeline
- `ims_data_extraction.py`: Meteorological data processing
- `model_predictors/`: Individual model implementations
- `ml_pipeline.py`: Generic model training pipeline

### Data Files
- `דאטה מעובד.xlsx`: Main agricultural dataset
- `data_20240407.xlsx`: Updated dataset with latest information
- `normalized_&_encoded.xlsx`: Preprocessed features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Israel Meteorological Service for weather data
- Agricultural experts for domain knowledge
- Open-source machine learning community

## 📞 Contact

For questions or support, please contact:
- Email: tuphr961@gmail.com
- Project Link: [https://github.com/yourusername/wheat_yield_prediction_FP_EMI]

## Version History

- **v1.0**: Initial implementation with basic models
- **v2.0**: Added advanced feature engineering and IMS integration
- **v3.0**: Comprehensive model suite with optimization
- **Current**: Enhanced preprocessing and evaluation pipeline

---

**Note**: This project is designed for agricultural yield prediction in Israeli conditions. Adaptations may be needed for other regions or crop types.
