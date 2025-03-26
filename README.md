# WeatherSense: Machine Learning Weather Classification

## Project Overview
WeatherSense is a machine learning project that predicts weather types based on meteorological measurements. The system implements K-Nearest Neighbors (KNN) and Naive Bayes classification algorithms to categorize weather into distinct types (Sunny, Rainy, Cloudy, and Snowy) based on various atmospheric measurements and conditions.

## Project Development Journey

### Initial Dataset Evaluation
The project began with an initial dataset ("monthly_profits.csv") containing meteorological measurements such as:
- Temperature
- Dew Point Temperature
- Relative Humidity
- Wind Speed
- Visibility
- Pressure
- Weather categories

However, correlation analysis revealed extremely weak relationships between the predictive features and the target weather variable:
- The strongest correlations with the weather variable were only 0.18 (Wind Speed) and -0.19 (Pressure)
- The correlation matrix showed almost no meaningful predictive signal in the data
- These weak correlations suggested that the dataset would produce poor classification results

### Current Dataset
A second, more comprehensive dataset ("weather_classification_data.csv") was selected, containing 13,200 observations with:

**Numerical Features:**
- Temperature
- Humidity
- Wind Speed
- Precipitation (%)
- Atmospheric Pressure
- UV Index
- Visibility (km)

**Categorical Features:**
- Cloud Cover (partly cloudy, clear, overcast)
- Season (Winter, Spring, Summer, Autumn)
- Location (inland, mountain, coastal)

**Target Variable:**
- Weather Type (Sunny, Rainy, Cloudy, Snowy)

This dataset shows much stronger feature correlations, with meaningful relationships like:
- Humidity and Precipitation (0.64)
- Visibility and Humidity (-0.48)
- Visibility and Precipitation (-0.46)

## Data Preprocessing
The preprocessing pipeline includes:
- Missing value handling
- Feature standardization for numerical variables
- One-hot encoding for categorical input features (Cloud Cover, Season, Location)
- Label encoding for the target variable (Weather Type)

## Model Implementation
The project implements two classification algorithms:

### K-Nearest Neighbors (KNN)
- Uses distance-based classification
- Hyperparameter tuning for optimal k value, distance metric, and weighting scheme
- Pipeline implementation with preprocessing steps

### Naive Bayes
- Probability-based classification using Gaussian Naive Bayes
- Appropriate for features with complex distributions
- Hyperparameter tuning for variance smoothing

## Feature Engineering
Additional features derived from base measurements:
- Temperature-Humidity interaction
- Wind-Precipitation interaction
- Season-Location combinations

## Evaluation Metrics
Models are evaluated using:
- Accuracy scores
- Classification reports (precision, recall, F1-score)
- Confusion matrices
- Cross-validation performance

## Feature Importance Analysis
The project includes permutation importance analysis to identify the most influential features for weather classification, helping understand the key meteorological factors that determine weather patterns.

## Project Structure
```
WeatherSense/
├── data/
│   ├── monthly_profits.csv          # Initial dataset (insufficient correlations)
│   └── weather_classification_data.csv  # Current dataset
├── notebooks/
│   ├── 01_initial_data_analysis.ipynb  # Analysis showing weak correlations
│   ├── 02_new_data_exploration.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── models/
│   ├── knn_model.pkl
│   └── naive_bayes_model.pkl
├── README.md
└── requirements.txt
```

## Lessons Learned
1. **Data Quality Importance**: The initial dataset demonstrated that even with technically correct implementation, models cannot perform well without meaningful correlations between features and target variables.

2. **Feature Selection**: Meteorological variables like precipitation percentage and visibility proved to be more predictive than basic temperature and pressure readings alone.

3. **Encoding Strategy**: 
   - One-hot encoding was crucial for categorical input features to prevent false ordinal relationships
   - Label encoding was appropriate for the target variable as required by classification algorithms

4. **Model Selection**: Both KNN and Naive Bayes showed different strengths, with KNN likely capturing local weather pattern variations better, while Naive Bayes handled the probability distributions of weather conditions.

## Future Improvements
- Integration of time series analysis for temporal weather patterns
- Implementation of ensemble methods combining KNN and Naive Bayes predictions
- Development of an interactive web interface for real-time predictions
- Extension to multi-step forecasting

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Getting Started
```bash
# Clone the repository
git clone https://github.com/username/WeatherSense.git

# Install dependencies
pip install -r requirements.txt

# Run the main script
python weather_classifier.py
```
