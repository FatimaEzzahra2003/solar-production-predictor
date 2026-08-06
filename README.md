# Solar Production Predictor

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

LSTM-based time series model to forecast solar panel power output from weather and installation data, aimed at improving supply/demand balance in smart-grid architectures.

## Overview

This project forecasts `Active_Power` (solar panel output) using historical weather conditions (temperature, humidity, solar radiation) combined with installation characteristics (panel rating, tilt, orientation, technology). Accurate short-term forecasts help anticipate solar production and support grid balancing decisions.

## Approach

1. **Exploratory Data Analysis (EDA)** — inspection of the raw dataset (weather + panel installation features) and missing timestamps
2. **Data cleaning** — handling of univariate outliers and multivariate outliers (on Global Horizontal Irradiance and Active_Power) via **DBSCAN**
3. **Data enrichment & feature engineering** — creation of relevant time-based and domain features
4. **Normalization** — Min-Max scaling, standard scaling, and log + Min-Max scaling depending on the variable's distribution
5. **Sequence construction** — sliding-window sequences (length 10) built from the normalized time series for the LSTM input
6. **Modeling** — LSTM network trained to predict `Active_Power` from the sequence of weather and panel features

## Dataset

- ~504K rows of solar panel readings combined with weather data (temperature, humidity, global horizontal radiation, rainfall) and installation metadata (technology, rating, number of panels, azimuth, tilt, orientation)
- Split into train / validation / test sets, reshaped into sequences: `(samples, 10 timesteps, 14 features)`

## Model & results

- **Architecture**: LSTM(64) → Dense(1), trained for regression (MSE loss)
- **Test loss (MSE)**: 0.0025 (on Min-Max normalized target)

## Tech stack

- **Data processing**: pandas, numpy
- **Outlier detection**: scikit-learn (DBSCAN)
- **Modeling**: TensorFlow / Keras (LSTM)

## Project structure

```
solar-production-predictor/
└── Code.ipynb   # Full pipeline: EDA → cleaning → feature engineering → LSTM
```

## How to explore this project

Open `Code.ipynb` with Jupyter Notebook or Google Colab. The notebook follows a linear pipeline, from raw data exploration to the final LSTM forecasting model.

## Author

**Fatima Ezzahra Bououdi** — Data Analyst | Data Scientist
[LinkedIn](https://www.linkedin.com/in/fatima-ezzahra-bououdi-5b9615240) · [Portfolio](https://fatima-ezzahra-bououdi.vercel.app/) · fatimaezzahrabououdi@gmail.com
