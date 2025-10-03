# Brain Stroke Risk Prediction Model Report

## Dataset Information
- **Total samples**: 5,110
- **Features**: 11
- **Stroke cases**: 249 (4.9%)
- **Training samples**: 4,088
- **Test samples**: 1,022

## Model Performance

### Best Model: Logistic Regression
- **Test AUC Score**: 0.8559
- **Cross-validation AUC**: 0.8938 (±0.0084)

### All Models Comparison:
- **Random Forest**: AUC = 0.8162, CV = 0.9947
- **Gradient Boosting**: AUC = 0.8346, CV = 0.9852
- **Logistic Regression**: AUC = 0.8559, CV = 0.8938
- **SVM**: AUC = 0.8024, CV = 0.9479

## Model Details
- **Algorithm**: Logistic Regression
- **Training approach**: SMOTE for class balancing
- **Feature scaling**: StandardScaler
- **Cross-validation**: 5-fold stratified

## Features Used
1. gender
2. age
3. hypertension
4. heart_disease
5. ever_married
6. work_type
7. residence_type
8. avg_glucose_level
9. bmi
10. smoking_status
11. family_history_stroke

## Usage Instructions

### Loading the Model
```python
import joblib
model_artifacts = joblib.load('models/stroke_prediction_model.pkl')
model = model_artifacts['model']
scaler = model_artifacts['scaler']
label_encoders = model_artifacts['label_encoders']
```

### Making Predictions
```python
# Prepare your input data (ensure same feature order)
# Apply label encoding and scaling
# Then predict
prediction = model.predict(scaled_features)
probability = model.predict_proba(scaled_features)[:, 1]
```

## Important Notes
- This model is for educational/research purposes only
- Always consult healthcare professionals for medical decisions
- Model performance may vary on different populations
- Regular retraining recommended with new data

Generated on: 2025-10-03 19:22:49
