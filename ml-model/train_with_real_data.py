#!/usr/bin/env python3
"""
Enhanced Stroke Risk Prediction Model Trainer with Real Dataset Support
======================================================================

This script trains the stroke prediction model using real medical datasets.
Supports multiple data sources including Kaggle stroke datasets.

Supported Datasets:
1. Kaggle Stroke Prediction Dataset
2. Healthcare Dataset Stroke Data
3. Custom CSV datasets with proper formatting

Usage:
    python train_with_real_data.py --dataset kaggle_stroke
    python train_with_real_data.py --dataset custom --data-path /path/to/data.csv
    python train_with_real_data.py --dataset synthetic (fallback)

Author: Brain Stroke Risk Prediction Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os
import argparse
import warnings
import requests
import zipfile
from pathlib import Path
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class RealDataStrokeModelTrainer:
    """
    Enhanced stroke risk prediction model trainer with real dataset support
    """

    def __init__(self, dataset_type='kaggle_stroke', data_path=None, output_dir='./'):
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.best_model = None
        self.model_metrics = {}
        self.label_encoders = {}

        # Standard feature mappings for consistency
        self.feature_mappings = {
            'gender': {'Male': 1, 'Female': 0, 'Other': 2},
            'ever_married': {'Yes': 1, 'No': 0},
            'work_type': {
                'Private': 0,
                'Self-employed': 1,
                'Govt_job': 2,
                'children': 3,
                'Never_worked': 4
            },
            'Residence_type': {'Urban': 1, 'Rural': 0},
            'smoking_status': {
                'never smoked': 0,
                'formerly smoked': 1,
                'smokes': 2,
                'Unknown': 3
            }
        }

    def download_kaggle_dataset(self):
        """Download Kaggle stroke dataset"""
        print("📥 Downloading Kaggle Stroke Dataset...")

        # Note: This requires Kaggle API setup
        # Users should download manually from:
        # https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

        dataset_url = "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"

        print(f"""
        ⚠️  MANUAL DOWNLOAD REQUIRED:

        1. Go to: {dataset_url}
        2. Download the 'healthcare-dataset-stroke-data.csv' file
        3. Place it in the 'data/' directory
        4. Run this script again

        Alternatively, set up Kaggle API:
        1. pip install kaggle
        2. Set up API credentials: ~/.kaggle/kaggle.json
        3. Run: kaggle datasets download -d fedesoriano/stroke-prediction-dataset
        """)

        # Check if dataset exists locally
        possible_paths = [
            'data/healthcare-dataset-stroke-data.csv',
            'data/stroke-data.csv',
            'healthcare-dataset-stroke-data.csv',
            'stroke-data.csv'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found dataset at: {path}")
                return path

        print("❌ Dataset not found. Please download manually.")
        return None

    def load_kaggle_stroke_dataset(self, file_path):
        """Load and preprocess Kaggle stroke dataset"""
        print(f"📊 Loading Kaggle dataset from: {file_path}")

        try:
            df = pd.read_csv(file_path)
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Display basic info
            print("\nDataset Info:")
            print(df.info())
            print(f"\nStroke distribution:")
            print(df['stroke'].value_counts())
            print(f"Stroke percentage: {df['stroke'].mean():.2%}")

            # Handle missing values
            print(f"\nMissing values:")
            print(df.isnull().sum())

            # Clean the dataset
            df = self.clean_kaggle_dataset(df)

            return df

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def clean_kaggle_dataset(self, df):
        """Clean and preprocess the Kaggle dataset"""
        print("🧹 Cleaning dataset...")

        # Make a copy
        df = df.copy()

        # Remove ID column if present
        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        # Handle missing values in BMI
        if 'bmi' in df.columns:
            # Fill missing BMI with median based on age and gender groups
            df['bmi'] = df.groupby(['age', 'gender'])['bmi'].transform(
                lambda x: x.fillna(x.median())
            )
            # If still missing, use overall median
            df['bmi'] = df['bmi'].fillna(df['bmi'].median())

        # Handle missing values in smoking_status
        if 'smoking_status' in df.columns:
            df['smoking_status'] = df['smoking_status'].fillna('Unknown')

        # Handle missing values in work_type
        if 'work_type' in df.columns:
            df['work_type'] = df['work_type'].fillna('Unknown')

        # Convert data types
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
        if 'avg_glucose_level' in df.columns:
            df['avg_glucose_level'] = pd.to_numeric(df['avg_glucose_level'], errors='coerce')
        if 'bmi' in df.columns:
            df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

        # Remove rows with critical missing values
        critical_columns = ['age', 'avg_glucose_level', 'stroke']
        df = df.dropna(subset=critical_columns)

        # Add synthetic columns if missing (to match our system)
        if 'family_history_stroke' not in df.columns:
            print("⚠️  Adding synthetic 'family_history_stroke' column...")
            # Estimate based on age and existing stroke cases
            df['family_history_stroke'] = np.random.binomial(
                1, 0.15 + (df['age'] > 60) * 0.1, size=len(df)
            )

        if 'alcohol_consumption' not in df.columns:
            print("⚠️  Adding synthetic 'alcohol_consumption' column...")
            # Estimate based on age, gender, and other factors
            choices = ['Never', 'Occasionally', 'Regularly', 'Heavy']
            probabilities = [0.3, 0.4, 0.25, 0.05]
            df['alcohol_consumption'] = np.random.choice(
                choices, size=len(df), p=probabilities
            )

        # Standardize column names to match our system
        column_mapping = {
            'Residence_type': 'residence_type',
            'Work_type': 'work_type'
        }
        df = df.rename(columns=column_mapping)

        print(f"✅ Cleaned dataset shape: {df.shape}")
        print(f"Final stroke distribution: {df['stroke'].value_counts()}")

        return df

    def load_custom_dataset(self, file_path):
        """Load custom CSV dataset"""
        print(f"📊 Loading custom dataset from: {file_path}")

        try:
            df = pd.read_csv(file_path)
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Validate required columns
            required_columns = [
                'age', 'gender', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'residence_type', 'avg_glucose_level', 'bmi',
                'smoking_status', 'stroke'
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return None

            print("✅ All required columns present")
            return df

        except Exception as e:
            print(f"❌ Error loading custom dataset: {e}")
            return None

    def create_synthetic_dataset(self, n_samples=5000):
        """Create synthetic dataset as fallback"""
        print(f"🔬 Creating synthetic dataset with {n_samples} samples...")

        np.random.seed(42)

        # Generate realistic synthetic data
        data = {}

        # Demographics
        data['gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        data['age'] = np.random.gamma(2, 20) + 18
        data['age'] = np.clip(data['age'], 18, 95).astype(int)

        # Personal status
        married_prob = np.where(data['age'] > 25, 0.7, 0.2)
        data['ever_married'] = np.where(
            np.random.binomial(1, married_prob), 'Yes', 'No'
        )

        # Work type based on age
        work_choices = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        work_probs = np.where(
            data['age'] < 18, [0.0, 0.0, 0.0, 1.0, 0.0],
            np.where(
                data['age'] < 65, [0.6, 0.15, 0.2, 0.0, 0.05],
                [0.3, 0.1, 0.1, 0.0, 0.5]
            )
        )
        data['work_type'] = [
            np.random.choice(work_choices, p=prob) for prob in work_probs
        ]

        # Geographic
        data['residence_type'] = np.random.choice(['Urban', 'Rural'], n_samples, p=[0.65, 0.35])

        # Medical conditions
        hypertension_prob = np.clip((data['age'] - 30) / 50, 0.05, 0.4)
        data['hypertension'] = np.random.binomial(1, hypertension_prob)

        heart_disease_prob = np.clip((data['age'] - 40) / 60, 0.02, 0.25)
        data['heart_disease'] = np.random.binomial(1, heart_disease_prob)

        # New fields
        data['family_history_stroke'] = np.random.binomial(1, 0.15, n_samples)

        # Lifestyle factors
        smoking_choices = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
        data['smoking_status'] = np.random.choice(smoking_choices, n_samples, p=[0.5, 0.2, 0.25, 0.05])

        alcohol_choices = ['Never', 'Occasionally', 'Regularly', 'Heavy']
        data['alcohol_consumption'] = np.random.choice(alcohol_choices, n_samples, p=[0.3, 0.4, 0.25, 0.05])

        # Physiological measures
        bmi_mean = 25 + (data['age'] - 40) * 0.1
        data['bmi'] = np.random.normal(bmi_mean, 4)
        data['bmi'] = np.clip(data['bmi'], 15, 50)

        glucose_base = 85 + (data['age'] - 30) * 0.3
        glucose_effects = (
            (data['bmi'] > 30) * 15 +
            data['hypertension'] * 25 +
            data['heart_disease'] * 20
        )
        data['avg_glucose_level'] = glucose_base + glucose_effects + np.random.normal(0, 20, n_samples)
        data['avg_glucose_level'] = np.clip(data['avg_glucose_level'], 60, 300)

        # Create realistic stroke risk
        stroke_risk = (
            (data['age'] > 60).astype(int) * 0.3 +
            data['hypertension'] * 0.25 +
            data['heart_disease'] * 0.2 +
            (data['avg_glucose_level'] > 140).astype(int) * 0.15 +
            (data['bmi'] > 30).astype(int) * 0.1 +
            (data['smoking_status'] == 'smokes').astype(int) * 0.15 +
            data['family_history_stroke'] * 0.2 +
            np.isin(data['alcohol_consumption'], ['Regularly', 'Heavy']).astype(int) * 0.1
        )

        # Convert to binary with realistic distribution (~5% stroke cases)
        stroke_threshold = np.percentile(stroke_risk, 95)
        data['stroke'] = (stroke_risk > stroke_threshold).astype(int)

        df = pd.DataFrame(data)
        print(f"✅ Synthetic dataset created: {df.shape}")
        print(f"Stroke cases: {data['stroke'].sum()} ({data['stroke'].mean():.1%})")

        return df

    def load_data(self):
        """Load dataset based on specified type"""
        print("=" * 60)
        print("LOADING DATASET")
        print("=" * 60)

        if self.dataset_type == 'kaggle_stroke':
            file_path = self.download_kaggle_dataset()
            if file_path:
                self.df = self.load_kaggle_stroke_dataset(file_path)
            else:
                print("⚠️  Falling back to synthetic data...")
                self.df = self.create_synthetic_dataset()

        elif self.dataset_type == 'custom':
            if not self.data_path:
                print("❌ Custom dataset path not provided")
                return False
            self.df = self.load_custom_dataset(self.data_path)

        else:  # synthetic
            self.df = self.create_synthetic_dataset()

        if self.df is None:
            print("❌ Failed to load dataset")
            return False

        # Perform basic analysis
        self.analyze_dataset()
        return True

    def analyze_dataset(self):
        """Perform exploratory data analysis"""
        print("\n" + "=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)

        print(f"\n📊 Dataset Overview:")
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print(f"\n🎯 Target Distribution:")
        stroke_counts = self.df['stroke'].value_counts()
        print(stroke_counts)
        print(f"Stroke rate: {self.df['stroke'].mean():.2%}")

        # Check for class imbalance
        imbalance_ratio = stroke_counts[0] / stroke_counts[1] if len(stroke_counts) > 1 else 1
        print(f"Class imbalance ratio: {imbalance_ratio:.1f}:1")

        if imbalance_ratio > 10:
            print("⚠️  Severe class imbalance detected! Will apply SMOTE.")

        print(f"\n📈 Numerical Features Summary:")
        numerical_cols = ['age', 'avg_glucose_level', 'bmi']
        print(self.df[numerical_cols].describe())

        print(f"\n📋 Categorical Features:")
        categorical_cols = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
        if 'alcohol_consumption' in self.df.columns:
            categorical_cols.append('alcohol_consumption')

        for col in categorical_cols:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts())

    def preprocess_data(self):
        """Preprocess the dataset for training"""
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)

        # Handle missing values
        print("🧹 Handling missing values...")
        missing_before = self.df.isnull().sum().sum()

        # Fill numerical columns with median
        numerical_cols = ['age', 'avg_glucose_level', 'bmi']
        for col in numerical_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} missing values with median: {median_val}")

        # Fill categorical columns with mode or 'Unknown'
        categorical_cols = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
        if 'alcohol_consumption' in self.df.columns:
            categorical_cols.append('alcohol_consumption')

        for col in categorical_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
                print(f"  Filled {col} missing values with mode: {mode_val}")

        missing_after = self.df.isnull().sum().sum()
        print(f"✅ Missing values: {missing_before} → {missing_after}")

        # Encode categorical variables
        print("🔢 Encoding categorical variables...")
        for column in self.df.columns:
            if column in self.feature_mappings:
                original_unique = self.df[column].nunique()
                self.df[column] = self.df[column].map(self.feature_mappings[column])

                # Handle unmapped values
                if self.df[column].isnull().any():
                    print(f"  Warning: Unmapped values in {column}")
                    self.df[column].fillna(-1, inplace=True)

                print(f"  Encoded {column}: {original_unique} categories")

        # Handle alcohol_consumption if not in mappings
        if 'alcohol_consumption' in self.df.columns and 'alcohol_consumption' not in self.feature_mappings:
            le = LabelEncoder()
            self.df['alcohol_consumption'] = le.fit_transform(self.df['alcohol_consumption'])
            self.label_encoders['alcohol_consumption'] = le
            print(f"  Encoded alcohol_consumption: {len(le.classes_)} categories")

        # Define features and target
        feature_columns = [
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
        ]

        # Add new columns if they exist
        if 'family_history_stroke' in self.df.columns:
            feature_columns.append('family_history_stroke')
        if 'alcohol_consumption' in self.df.columns:
            feature_columns.append('alcohol_consumption')

        # Ensure all features are present
        missing_features = [col for col in feature_columns if col not in self.df.columns]
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return False

        X = self.df[feature_columns].copy()
        y = self.df['stroke'].copy()

        print(f"✅ Feature matrix shape: {X.shape}")
        print(f"✅ Target vector shape: {y.shape}")
        print(f"✅ Features: {feature_columns}")

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance with SMOTE
        stroke_rate = self.y_train.mean()
        if stroke_rate < 0.1:  # Less than 10% positive class
            print("⚖️  Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42, k_neighbors=min(5, self.y_train.sum() - 1))
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"  After SMOTE - Train set shape: {self.X_train.shape}")
            print(f"  After SMOTE - Stroke rate: {self.y_train.mean():.2%}")

        # Scale features
        print("📏 Scaling features...")
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"✅ Training set: {self.X_train.shape}")
        print(f"✅ Test set: {self.X_test.shape}")

        return True

    def train_models(self):
        """Train multiple models with hyperparameter tuning"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)

        # Define models with more sophisticated hyperparameters
        models = {
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'class_weight': ['balanced', None]
                },
                'use_scaled': True
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                },
                'use_scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'use_scaled': False
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced', None]
                },
                'use_scaled': True
            }
        }

        results = {}

        # Use stratified k-fold for better evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, config in models.items():
            print(f"\n🔄 Training {name}...")

            # Select appropriate data
            X_train_data = self.X_train_scaled if config['use_scaled'] else self.X_train
            X_test_data = self.X_test_scaled if config['use_scaled'] else self.X_test

            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='roc_auc',  # Better for imbalanced datasets
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train_data, self.y_train)

            # Predictions
            y_pred = grid_search.predict(X_test_data)
            y_pred_proba = grid_search.predict_proba(X_test_data)[:, 1]

            # Calculate comprehensive metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Cross-validation scores
            cv_scores = cross_val_score(
                grid_search.best_estimator_,
                X_train_data,
                self.y_train,
                cv=cv,
                scoring='roc_auc'
            )

            results[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'use_scaled': config['use_scaled']
            }

            print(f"  ✅ {name}:")
            print(f"     Accuracy: {accuracy:.4f}")
            print(f"     Precision: {precision:.4f}")
            print(f"     Recall: {recall:.4f}")
            print(f"     F1-Score: {f1:.4f}")
            print(f"     ROC-AUC: {roc_auc:.4f}")
            print(f"     CV Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

        # Select best model based on ROC-AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        self.best_model = results[best_model_name]['model']
        self.model_metrics = results

        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

        return results

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        best_name = max(self.model_metrics.keys(), key=lambda k: self.model_metrics[k]['roc_auc'])
        best_result = self.model_metrics[best_name]

        print(f"\n📊 Detailed Results for {best_name}:")
        print("=" * 40)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, best_result['predictions']))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, best_result['predictions'])
        print(f"\nConfusion Matrix:")
        print(cm)

        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall

        print(f"\nDetailed Metrics:")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Precision: {best_result['precision']:.4f}")
        print(f"  F1-Score: {best_result['f1_score']:.4f}")
        print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")

        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = [
                'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
            ]

            # Add new features if present
            if 'family_history_stroke' in self.df.columns:
                feature_names.append('family_history_stroke')
            if 'alcohol_consumption' in self.df.columns:
                feature_names.append('alcohol_consumption')

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 10 Feature Importances:")
            print(importance_df.head(10).to_string(index=False))

    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        # Set up the plotting
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))

        # 1. ROC Curves
        ax1 = plt.subplot(2, 3, 1)
        for name, result in self.model_metrics.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})", linewidth=2)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Precision-Recall Curves
        ax2 = plt.subplot(2, 3, 2)
        for name, result in self.model_metrics.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['probabilities'])
            ax2.plot(recall, precision, label=name, linewidth=2)

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Model Performance Comparison
        ax3 = plt.subplot(2, 3, 3)
        models = list(self.model_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        x = np.arange(len(models))
        width = 0.15

        for i, metric in enumerate(metrics):
            values = [self.model_metrics[m][metric] for m in models]
            ax3.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)

        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x + width * 2)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Confusion Matrix for Best Model
        ax4 = plt.subplot(2, 3, 4)
        best_name = max(self.model_metrics.keys(), key=lambda k: self.model_metrics[k]['roc_auc'])
        cm = confusion_matrix(self.y_test, self.model_metrics[best_name]['predictions'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title(f'Confusion Matrix - {best_name}')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')

        # 5. Feature Importance (if available)
        ax5 = plt.subplot(2, 3, 5)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = [
                'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
            ]
            if 'family_history_stroke' in self.df.columns:
                feature_names.append('family_history_stroke')
            if 'alcohol_consumption' in self.df.columns:
                feature_names.append('alcohol_consumption')

            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10

            ax5.bar(range(len(indices)), importances[indices], alpha=0.8)
            ax5.set_title(f'Top 10 Feature Importances - {best_name}')
            ax5.set_xlabel('Features')
            ax5.set_ylabel('Importance')
            ax5.set_xticks(range(len(indices)))
            ax5.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        else:
            ax5.text(0.5, 0.5, 'Feature importance not available\nfor this model type',
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Feature Importance - Not Available')

        # 6. Dataset Distribution
        ax6 = plt.subplot(2, 3, 6)
        stroke_dist = self.df['stroke'].value_counts()
        colors = ['lightblue', 'lightcoral']
        wedges, texts, autotexts = ax6.pie(stroke_dist.values, labels=['No Stroke', 'Stroke'],
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax6.set_title('Dataset: Stroke Distribution')

        plt.tight_layout()

        # Save the plot
        plot_path = self.output_dir / 'model_evaluation_comprehensive.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive evaluation plot saved: {plot_path}")

        # Generate additional correlation heatmap
        plt.figure(figsize=(12, 10))

        # Prepare correlation matrix
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()

        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, fmt='.2f', square=True, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save correlation plot
        corr_path = self.output_dir / 'feature_correlation_matrix.png'
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"✅ Correlation matrix saved: {corr_path}")

        plt.close('all')  # Close all figures to free memory

    def save_model(self):
        """Save the trained model and related artifacts"""
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)

        # Save the best model
        model_path = self.output_dir / 'stroke_model.pkl'
        joblib.dump(self.best_model, model_path)
        print(f"✅ Model saved: {model_path}")

        # Save the scaler
        scaler_path = self.output_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")

        # Save label encoders if any
        if self.label_encoders:
            encoders_path = self.output_dir / 'label_encoders.pkl'
            joblib.dump(self.label_encoders, encoders_path)
            print(f"✅ Label encoders saved: {encoders_path}")

        # Create comprehensive metadata
        best_name = max(self.model_metrics.keys(), key=lambda k: self.model_metrics[k]['roc_auc'])
        best_result = self.model_metrics[best_name]

        metadata = {
            'model_info': {
                'model_type': type(self.best_model).__name__,
                'model_name': best_name,
                'training_date': datetime.now().isoformat(),
                'dataset_type': self.dataset_type,
                'dataset_shape': self.df.shape,
                'features_used': list(self.X_train.columns) if hasattr(self.X_train, 'columns') else 'N/A'
            },
            'performance_metrics': {
                'accuracy': float(best_result['accuracy']),
                'precision': float(best_result['precision']),
                'recall': float(best_result['recall']),
                'f1_score': float(best_result['f1_score']),
                'roc_auc': float(best_result['roc_auc']),
                'cv_mean': float(best_result['cv_mean']),
                'cv_std': float(best_result['cv_std'])
            },
            'hyperparameters': best_result['best_params'],
            'dataset_info': {
                'total_samples': int(len(self.df)),
                'training_samples': int(len(self.X_train)),
                'test_samples': int(len(self.X_test)),
                'stroke_rate': float(self.df['stroke'].mean()),
                'feature_count': int(self.X_train.shape[1])
            },
            'feature_mappings': self.feature_mappings,
            'model_comparison': {
                name: {
                    'accuracy': float(result['accuracy']),
                    'precision': float(result['precision']),
                    'recall': float(result['recall']),
                    'f1_score': float(result['f1_score']),
                    'roc_auc': float(result['roc_auc'])
                }
                for name, result in self.model_metrics.items()
            },
            'files_created': [
                'stroke_model.pkl',
                'scaler.pkl',
                'model_metadata.json',
                'model_evaluation_comprehensive.png',
                'feature_correlation_matrix.png'
            ] + (['label_encoders.pkl'] if self.label_encoders else [])
        }

        # Save metadata
        metadata_path = self.output_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ Metadata saved: {metadata_path}")

        # Generate model summary report
        self.generate_model_report(metadata)

        print(f"\n🎉 All model artifacts saved to: {self.output_dir}")

    def generate_model_report(self, metadata):
        """Generate a comprehensive model report"""
        report_path = self.output_dir / 'MODEL_REPORT.md'

        best_name = metadata['model_info']['model_name']
        metrics = metadata['performance_metrics']

        report = f"""# Stroke Risk Prediction Model Report

## Model Overview
- **Model Type**: {metadata['model_info']['model_type']}
- **Model Name**: {best_name}
- **Training Date**: {metadata['model_info']['training_date']}
- **Dataset Type**: {metadata['model_info']['dataset_type']}

## Dataset Information
- **Total Samples**: {metadata['dataset_info']['total_samples']:,}
- **Training Samples**: {metadata['dataset_info']['training_samples']:,}
- **Test Samples**: {metadata['dataset_info']['test_samples']:,}
- **Stroke Rate**: {metadata['dataset_info']['stroke_rate']:.2%}
- **Features Used**: {metadata['dataset_info']['feature_count']}

## Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall (Sensitivity)**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
- **ROC-AUC**: {metrics['roc_auc']:.4f}
- **Cross-Validation Score**: {metrics['cv_mean']:.4f} (±{metrics['cv_std']*2:.4f})

## Model Comparison
"""

        for model_name, model_metrics in metadata['model_comparison'].items():
            report += f"\n### {model_name}\n"
            for metric, value in model_metrics.items():
                report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"

        report += f"""
## Best Hyperparameters
"""
        for param, value in metadata['hyperparameters'].items():
            report += f"- **{param}**: {value}\n"

        report += f"""
## Files Generated
"""
        for file in metadata['files_created']:
            report += f"- `{file}`\n"

        report += f"""
## Usage Instructions

### Loading the Model
```python
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load('stroke_model.pkl')
scaler = joblib.load('scaler.pkl')

# For categorical encodings (if needed)
# label_encoders = joblib.load('label_encoders.pkl')
```

### Making Predictions
```python
# Prepare your data (example)
user_data = {{
    'gender': 1,  # 1 for Male, 0 for Female
    'age': 45,
    'hypertension': 0,  # 0 for No, 1 for Yes
    'heart_disease': 0,
    'ever_married': 1,  # 1 for Yes, 0 for No
    'work_type': 0,  # 0-4 based on mapping
    'residence_type': 1,  # 1 for Urban, 0 for Rural
    'avg_glucose_level': 120.5,
    'bmi': 25.3,
    'smoking_status': 0,  # 0-3 based on mapping
    'family_history_stroke': 0,  # 0 for No, 1 for Yes
    'alcohol_consumption': 1  # 0-3 based on mapping
}}

# Convert to numpy array and scale
features = np.array(list(user_data.values())).reshape(1, -1)
features_scaled = scaler.transform(features)

# Make prediction
prediction = model.predict(features_scaled)[0]
probability = model.predict_proba(features_scaled)[0][1]

print(f"Stroke Risk: {{'High' if prediction else 'Low'}}")
print(f"Probability: {{probability:.2%}}")
```

## Medical Disclaimer
⚠️ **IMPORTANT**: This model is for educational and research purposes only.
It should not replace professional medical advice, diagnosis, or treatment.
Always consult with qualified healthcare professionals for medical decisions.

## Model Limitations
- Based on historical data patterns
- May not account for all individual factors
- Requires regular retraining with new data
- Should be validated in clinical settings

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"✅ Model report saved: {report_path}")

    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("🧠" * 20)
        print("BRAIN STROKE RISK PREDICTION MODEL TRAINER")
        print("Enhanced Version with Real Dataset Support")
        print("🧠" * 20)

        try:
            # Load data
            if not self.load_data():
                return False

            # Preprocess data
            if not self.preprocess_data():
                return False

            # Train models
            self.train_models()

            # Evaluate models
            self.evaluate_model()

            # Generate visualizations
            self.generate_visualizations()

            # Save model and artifacts
            self.save_model()

            print("\n" + "🎉" * 20)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("🎉" * 20)

            return True

        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Train Stroke Risk Prediction Model with Real Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_with_real_data.py --dataset kaggle_stroke
  python train_with_real_data.py --dataset custom --data-path ./data/my_stroke_data.csv
  python train_with_real_data.py --dataset synthetic --output-dir ./models/
        """
    )

    parser.add_argument('--dataset',
                       choices=['kaggle_stroke', 'custom', 'synthetic'],
                       default='kaggle_stroke',
                       help='Dataset type to use for training')

    parser.add_argument('--data-path',
                       type=str,
                       help='Path to custom dataset CSV file (required for custom dataset)')

    parser.add_argument('--output-dir',
                       type=str,
                       default='./',
                       help='Output directory for model artifacts')

    args = parser.parse_args()

    # Validate arguments
    if args.dataset == 'custom' and not args.data_path:
        print("❌ Error: --data-path is required when using custom dataset")
        parser.print_help()
        return 1

    # Initialize trainer
    trainer = RealDataStrokeModelTrainer(
        dataset_type=args.dataset,
        data_path=args.data_path,
        output_dir=args.output_dir
    )

    # Run training pipeline
    success = trainer.run_complete_pipeline()

    if success:
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model artifacts saved to: {trainer.output_dir}")
        print(f"\n📚 Next steps:")
        print(f"  1. Review the model report: {trainer.output_dir}/MODEL_REPORT.md")
        print(f"  2. Check evaluation plots: {trainer.output_dir}/*.png")
        print(f"  3. Test the model with new data")
        print(f"  4. Deploy to your application")
        return 0
    else:
        print(f"\n❌ Training failed. Check the error messages above.")
        return 1


if __name__ == "__main__":
    exit(main())
