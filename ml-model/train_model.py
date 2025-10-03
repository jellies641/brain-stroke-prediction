import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import joblib
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

class StrokeModelTrainer:
    """
    Enhanced stroke risk prediction model trainer with real dataset support
    Supports multiple algorithms with hyperparameter tuning
    """

    def __init__(self, data_path=None, use_real_data=True):
        self.data_path = data_path
        self.use_real_data = use_real_data
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.model_metrics = {}

        # Feature mappings for consistency
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
            'residence_type': {'Urban': 1, 'Rural': 0},
            'smoking_status': {
                'never smoked': 0,
                'formerly smoked': 1,
                'smokes': 2,
                'Unknown': 3
            },
            'alcohol_consumption': {
                'Never': 0,
                'Occasionally': 1,
                'Regularly': 2,
                'Heavy': 3
            }
        }

    def create_synthetic_dataset(self, n_samples=5000):
        """Create a comprehensive synthetic stroke dataset"""
        print("Creating synthetic stroke dataset...")

        np.random.seed(42)

        # Generate base features
        data = {}

        # Demographics
        data['gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        data['age'] = np.random.gamma(2, 20) + 18  # Skewed towards older ages
        data['age'] = np.clip(data['age'], 18, 95).astype(int)

        # Personal status
        married_prob = np.where(data['age'] > 25, 0.7, 0.2)
        data['ever_married'] = np.random.binomial(1, married_prob)
        data['ever_married'] = np.where(data['ever_married'], 'Yes', 'No')

        # Work type based on age
        work_probs = np.where(
            data['age'] < 18, [0.0, 0.0, 0.0, 1.0, 0.0],  # children
            np.where(
                data['age'] < 65, [0.6, 0.15, 0.2, 0.0, 0.05],  # working age
                [0.3, 0.1, 0.1, 0.0, 0.5]  # retirement age
            )
        )

        work_types = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        data['work_type'] = [
            np.random.choice(work_types, p=prob) for prob in work_probs
        ]

        # Geographic
        data['residence_type'] = np.random.choice(['Urban', 'Rural'], n_samples, p=[0.65, 0.35])

        # Medical conditions - age dependent
        hypertension_prob = np.clip((data['age'] - 30) / 50, 0.05, 0.4)
        data['hypertension'] = np.random.binomial(1, hypertension_prob)

        heart_disease_prob = np.clip((data['age'] - 40) / 60, 0.02, 0.25)
        data['heart_disease'] = np.random.binomial(1, heart_disease_prob)

        # Family history
        data['family_history_stroke'] = np.random.binomial(1, 0.15, n_samples)

        # Lifestyle factors
        smoking_age_factor = np.where(data['age'] > 30, 0.3, 0.15)
        smoking_dist = np.random.multinomial(
            1, [0.5, 0.2, 0.25, 0.05], n_samples
        )
        smoking_options = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
        data['smoking_status'] = [
            smoking_options[np.argmax(row)] for row in smoking_dist
        ]

        # Alcohol consumption
        alcohol_dist = np.random.multinomial(1, [0.3, 0.4, 0.25, 0.05], n_samples)
        alcohol_options = ['Never', 'Occasionally', 'Regularly', 'Heavy']
        data['alcohol_consumption'] = [
            alcohol_options[np.argmax(row)] for row in alcohol_dist
        ]

        # Physiological measures
        # BMI with realistic distribution
        bmi_mean = 25 + (data['age'] - 40) * 0.1  # Slight increase with age
        data['bmi'] = np.random.normal(bmi_mean, 4)
        data['bmi'] = np.clip(data['bmi'], 15, 50)

        # Glucose level - influenced by age, BMI, and medical conditions
        glucose_base = 85 + (data['age'] - 30) * 0.3
        glucose_bmi_effect = np.where(data['bmi'] > 30, 15, 0)
        glucose_hypertension_effect = data['hypertension'] * 25
        glucose_heart_effect = data['heart_disease'] * 20

        data['avg_glucose_level'] = (
            glucose_base + glucose_bmi_effect +
            glucose_hypertension_effect + glucose_heart_effect +
            np.random.normal(0, 20, n_samples)
        )
        data['avg_glucose_level'] = np.clip(data['avg_glucose_level'], 60, 300)

        # Create stroke risk based on multiple factors
        stroke_risk = self._calculate_stroke_risk(data)

        # Convert to binary with some randomness
        stroke_threshold = np.percentile(stroke_risk, 85)  # Top 15% get stroke
        data['stroke'] = (stroke_risk > stroke_threshold).astype(int)

        # Add some randomness to make it more realistic
        flip_prob = 0.05  # 5% chance of flipping the label
        random_flips = np.random.random(n_samples) < flip_prob
        data['stroke'][random_flips] = 1 - data['stroke'][random_flips]

        # Add new fields for compatibility with updated system
        data['family_history_stroke'] = np.random.binomial(1, 0.15, n_samples)

        alcohol_choices = ['Never', 'Occasionally', 'Regularly', 'Heavy']
        data['alcohol_consumption'] = np.random.choice(alcohol_choices, n_samples, p=[0.3, 0.4, 0.25, 0.05])

        # Convert to DataFrame
        self.df = pd.DataFrame(data)

        print(f"Dataset created with {n_samples} samples")
        print(f"Stroke cases: {data['stroke'].sum()} ({data['stroke'].mean():.1%})")

        return self.df

    def _calculate_stroke_risk(self, data):
        """Calculate stroke risk score based on medical literature"""
        risk_score = np.zeros(len(data['age']))

        # Age factor (strongest predictor)
        risk_score += (data['age'] - 18) / 77 * 0.4  # 40% weight

        # Medical conditions
        risk_score += data['hypertension'] * 0.2  # 20% weight
        risk_score += data['heart_disease'] * 0.15  # 15% weight
        risk_score += data['family_history_stroke'] * 0.1  # 10% weight

        # Physiological measures
        bmi_risk = np.where(data['bmi'] > 30, 0.05, 0)  # Obesity
        risk_score += bmi_risk

        glucose_risk = np.where(data['avg_glucose_level'] > 140, 0.05, 0)  # Diabetes
        risk_score += glucose_risk

        # Lifestyle factors
        smoking_risk = np.where(
            [status == 'smokes' for status in data['smoking_status']], 0.03, 0
        )
        risk_score += smoking_risk

        alcohol_risk = np.where(
            [status in ['Regularly', 'Heavy'] for status in data['alcohol_consumption']], 0.02, 0
        )
        risk_score += alcohol_risk

        return risk_score

    def load_kaggle_dataset(self, file_path):
        """Load and preprocess Kaggle stroke dataset"""
        print(f"Loading Kaggle dataset from: {file_path}")

        try:
            df = pd.read_csv(file_path)

            # Remove ID column if present
            if 'id' in df.columns:
                df = df.drop('id', axis=1)

            # Standardize column names
            if 'Residence_type' in df.columns:
                df = df.rename(columns={'Residence_type': 'residence_type'})
            if 'Work_type' in df.columns:
                df = df.rename(columns={'Work_type': 'work_type'})

            # Add missing columns with synthetic data if needed
            if 'family_history_stroke' not in df.columns:
                print("Adding synthetic 'family_history_stroke' column...")
                df['family_history_stroke'] = np.random.binomial(1, 0.15 + (df['age'] > 60) * 0.1, size=len(df))

            if 'alcohol_consumption' not in df.columns:
                print("Adding synthetic 'alcohol_consumption' column...")
                choices = ['Never', 'Occasionally', 'Regularly', 'Heavy']
                probabilities = [0.3, 0.4, 0.25, 0.05]
                df['alcohol_consumption'] = np.random.choice(choices, size=len(df), p=probabilities)

            # Handle missing values
            if 'bmi' in df.columns:
                df['bmi'] = df.groupby(['age', 'gender'])['bmi'].transform(lambda x: x.fillna(x.median()))
                df['bmi'] = df['bmi'].fillna(df['bmi'].median())

            print(f"Dataset loaded successfully: {df.shape}")
            print(f"Stroke distribution: {df['stroke'].value_counts()}")
            return df

        except Exception as e:
            print(f"Error loading Kaggle dataset: {e}")
            return None

    def load_data(self, data_path=None):
        """Load dataset from file or create synthetic data"""

        # Try to load real data first
        if self.use_real_data:
            # Look for common dataset file names
            possible_paths = [
                'data/healthcare-dataset-stroke-data.csv',
                'data/kaggle-stroke-data.csv',
                'data/stroke-data.csv',
                'healthcare-dataset-stroke-data.csv',
                'stroke-data.csv'
            ]

            # Add provided path if available
            if data_path:
                possible_paths.insert(0, data_path)

            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found real dataset: {path}")
                    self.df = self.load_kaggle_dataset(path)
                    if self.df is not None:
                        break

            if self.df is None:
                print("⚠️  No real dataset found. Creating synthetic dataset...")
                self.create_synthetic_dataset()
        else:
            print("Creating synthetic dataset...")
            self.create_synthetic_dataset()

        print(f"Final dataset shape: {self.df.shape}")
        print("\nDataset info:")
        print(self.df.info())
        return self.df

    def preprocess_data(self):
        """Preprocess the dataset for training"""
        print("\nPreprocessing data...")

        # Handle missing values
        missing_before = self.df.isnull().sum().sum()
        if missing_before > 0:
            print(f"Handling {missing_before} missing values...")

            # Fill numeric columns with median
            numeric_columns = ['age', 'avg_glucose_level', 'bmi']
            for col in numeric_columns:
                if col in self.df.columns and self.df[col].isnull().any():
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  Filled {col} missing values with median: {median_val}")

            # Fill categorical columns with mode or default
            categorical_columns = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status', 'alcohol_consumption']
            for col in categorical_columns:
                if col in self.df.columns and self.df[col].isnull().any():
                    if not self.df[col].mode().empty:
                        mode_val = self.df[col].mode()[0]
                        self.df[col].fillna(mode_val, inplace=True)
                        print(f"  Filled {col} missing values with mode: {mode_val}")

        # Encode categorical variables
        for column, mapping in self.feature_mappings.items():
            if column in self.df.columns:
                original_unique = self.df[column].nunique()
                self.df[column] = self.df[column].map(mapping)

                # Handle unmapped values
                if self.df[column].isnull().any():
                    print(f"  Warning: Unmapped values in {column}, filling with 0")
                    self.df[column].fillna(0, inplace=True)

                print(f"  Encoded {column}: {original_unique} categories")

        # Handle alcohol_consumption if not in mappings
        if 'alcohol_consumption' in self.df.columns and self.df['alcohol_consumption'].dtype == 'object':
            le = LabelEncoder()
            self.df['alcohol_consumption'] = le.fit_transform(self.df['alcohol_consumption'])
            print(f"  Encoded alcohol_consumption: {len(le.classes_)} categories")

        # Define features and target
        feature_columns = [
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'residence_type', 'avg_glucose_level', 'bmi',
            'smoking_status', 'family_history_stroke', 'alcohol_consumption'
        ]

        # Ensure all features are present
        missing_features = [col for col in feature_columns if col not in self.df.columns]
        if missing_features:
            print(f"Warning: Missing features {missing_features}, setting to 0")
            for col in missing_features:
                self.df[col] = 0

        X = self.df[feature_columns]
        y = self.df['stroke']

        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {feature_columns}")

        # Split the data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance with SMOTE if needed
        stroke_rate = self.y_train.mean()
        if stroke_rate < 0.1:  # Less than 10% positive class
            print(f"Applying SMOTE to balance classes (current stroke rate: {stroke_rate:.1%})...")
            smote = SMOTE(random_state=42, k_neighbors=min(5, self.y_train.sum() - 1))
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"After SMOTE - Training set shape: {self.X_train.shape}")
            print(f"After SMOTE - Stroke rate: {self.y_train.mean():.1%}")

        # Scale features with RobustScaler (more robust to outliers)
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Final training stroke rate: {self.y_train.mean():.1%}")
        print(f"Test set stroke rate: {self.y_test.mean():.1%}")

    def train_models(self):
        """Train multiple models and compare performance"""
        print("\nTraining multiple models...")

        # Define models to train
        models = {
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced')
        }

        # Hyperparameter grids
        param_grids = {
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }

        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model,
                param_grids[name],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )

            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                grid_search.fit(self.X_train_scaled, self.y_train)
                y_pred = grid_search.predict(self.X_test_scaled)
                y_pred_proba = grid_search.predict_proba(self.X_test_scaled)[:, 1]
            else:
                grid_search.fit(self.X_train, self.y_train)
                y_pred = grid_search.predict(self.X_test)
                y_pred_proba = grid_search.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Cross-validation score
            if name in ['SVM', 'Logistic Regression']:
                cv_scores = cross_val_score(grid_search.best_estimator_, self.X_train_scaled, self.y_train, cv=5, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(grid_search.best_estimator_, self.X_train, self.y_train, cv=5, scoring='roc_auc')

            results[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            print(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Select best model based on ROC-AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        self.best_model = results[best_model_name]['model']
        self.model_metrics = results

        print(f"\nBest model: {best_model_name}")
        print(f"Best ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

        return results

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\nEvaluating best model...")

        best_name = max(self.model_metrics.keys(), key=lambda k: self.model_metrics[k]['roc_auc'])
        best_result = self.model_metrics[best_name]

        y_pred = best_result['predictions']
        y_pred_proba = best_result['probabilities']

        # Classification report
        print(f"\nClassification Report for {best_name}:")
        print(classification_report(self.y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        # Feature importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = [
                'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'residence_type', 'avg_glucose_level', 'bmi',
                'smoking_status', 'family_history_stroke', 'alcohol_consumption'
            ]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 10 Feature Importances:")
            print(importance_df.head(10))

    def save_model(self, model_path='stroke_model.pkl', scaler_path='scaler.pkl'):
        """Save the trained model and scaler"""
        print(f"\nSaving model to {model_path}")
        print(f"Saving scaler to {scaler_path}")

        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        # Save model metadata
        metadata = {
            'model_type': type(self.best_model).__name__,
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'features': [
                'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'residence_type', 'avg_glucose_level', 'bmi',
                'smoking_status', 'family_history_stroke', 'alcohol_consumption'
            ],
            'metrics': {k: {
                'accuracy': v['accuracy'],
                'roc_auc': v['roc_auc'],
                'cv_mean': v['cv_mean']
            } for k, v in self.model_metrics.items()},
            'feature_mappings': self.feature_mappings
        }

        import json
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print("Model saved successfully!")

    def generate_plots(self):
        """Generate evaluation plots"""
        print("\nGenerating evaluation plots...")

        try:
            plt.style.use('seaborn')
        except:
            pass

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ROC curves
        ax1 = axes[0, 0]
        for name, result in self.model_metrics.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Precision-Recall curves
        ax2 = axes[0, 1]
        for name, result in self.model_metrics.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['probabilities'])
            ax2.plot(recall, precision, label=name)

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Model comparison
        ax3 = axes[1, 0]
        models = list(self.model_metrics.keys())
        accuracies = [self.model_metrics[m]['accuracy'] for m in models]
        roc_aucs = [self.model_metrics[m]['roc_auc'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        ax3.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax3.bar(x + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)

        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Feature distribution by stroke status
        ax4 = axes[1, 1]

        # Age distribution
        stroke_ages = self.df[self.df['stroke'] == 1]['age']
        no_stroke_ages = self.df[self.df['stroke'] == 0]['age']

        ax4.hist(no_stroke_ages, bins=20, alpha=0.7, label='No Stroke', density=True)
        ax4.hist(stroke_ages, bins=20, alpha=0.7, label='Stroke', density=True)
        ax4.set_xlabel('Age')
        ax4.set_ylabel('Density')
        ax4.set_title('Age Distribution by Stroke Status')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'model_evaluation.png'")

        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))

        # Prepare numeric dataframe
        numeric_df = self.df.copy()
        for col, mapping in self.feature_mappings.items():
            if col in numeric_df.columns and numeric_df[col].dtype == 'object':
                numeric_df[col] = numeric_df[col].map(mapping)

        correlation_matrix = numeric_df.corr()

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Correlation matrix saved as 'correlation_matrix.png'")

    def run_full_pipeline(self, data_path=None):
        """Run the complete training pipeline"""
        print("="*50)
        print("BRAIN STROKE RISK PREDICTION MODEL TRAINING")
        print("="*50)

        # Load data
        self.load_data(data_path)

        # Preprocess
        self.preprocess_data()

        # Train models
        self.train_models()

        # Evaluate
        self.evaluate_model()

        # Generate plots
        self.generate_plots()

        # Save model
        self.save_model()

        print("\n" + "="*50)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)

        return self.best_model, self.scaler


def main():
    """Main training function with command line support"""
    parser = argparse.ArgumentParser(description="Train Stroke Risk Prediction Model")
    parser.add_argument('--data-path', type=str, help='Path to dataset CSV file')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of real data')
    parser.add_argument('--samples', type=int, default=5000, help='Number of synthetic samples (default: 5000)')

    args = parser.parse_args()

    print("🧠" * 20)
    print("BRAIN STROKE RISK PREDICTION MODEL TRAINER")
    print("Enhanced Version with Real Dataset Support")
    print("🧠" * 20)

    # Initialize trainer
    trainer = StrokeModelTrainer(
        data_path=args.data_path,
        use_real_data=not args.synthetic
    )

    # Override synthetic dataset size if specified
    if args.synthetic and args.samples != 5000:
        trainer.create_synthetic_dataset = lambda: trainer._create_synthetic_dataset(args.samples)

    try:
        # Run full pipeline
        model, scaler = trainer.run_full_pipeline()

        print(f"\n✅ Training completed successfully!")
        print(f"Final Model: {type(model).__name__}")
        print("\n📁 Files created:")
        print("- stroke_model.pkl")
        print("- scaler.pkl")
        print("- model_metadata.json")
        print("- model_evaluation.png")
        print("- correlation_matrix.png")

        print(f"\n📊 Next Steps:")
        print("1. Review model performance in model_evaluation.png")
        print("2. Check model_metadata.json for detailed metrics")
        print("3. Test the model with new data")
        print("4. Deploy to your application")

        # Show dataset info
        if trainer.use_real_data and trainer.df is not None:
            stroke_rate = trainer.df['stroke'].mean()
            print(f"\n📈 Dataset Summary:")
            print(f"   Total records: {len(trainer.df):,}")
            print(f"   Stroke cases: {trainer.df['stroke'].sum():,} ({stroke_rate:.1%})")
            print(f"   Data source: {'Real dataset' if stroke_rate < 0.15 else 'Synthetic dataset'}")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
