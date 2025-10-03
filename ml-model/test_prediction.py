#!/usr/bin/env python3
"""
Test Script for Brain Stroke Risk Prediction Model
This script demonstrates how to use the trained model for predictions
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class StrokePredictionService:
    def __init__(self, model_path="models/stroke_prediction_model.pkl"):
        """Initialize the prediction service"""
        print("🧠 Loading Brain Stroke Prediction Model...")

        # Load model artifacts
        self.model_artifacts = joblib.load(model_path)
        self.model = self.model_artifacts['model']
        self.scaler = self.model_artifacts['scaler']
        self.label_encoders = self.model_artifacts['label_encoders']
        self.feature_names = self.model_artifacts['feature_names']

        print(f"✅ Model loaded: {self.model_artifacts['model_name']}")
        print(f"📊 Model AUC Score: {self.model_artifacts['model_metrics'][self.model_artifacts['model_name']]['auc_score']:.4f}")

    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        # Create a copy to avoid modifying original data
        data = input_data.copy()

        # Apply label encoding for categorical features
        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

        for feature in categorical_features:
            if feature in data and feature in self.label_encoders:
                # Convert to string to match training
                str_value = str(data[feature])

                # Check if value exists in encoder classes
                if str_value in self.label_encoders[feature].classes_:
                    data[feature] = self.label_encoders[feature].transform([str_value])[0]
                else:
                    # Handle unseen categories by using most common class (0)
                    print(f"⚠️ Warning: Unknown value '{str_value}' for {feature}, using default")
                    data[feature] = 0

        # Ensure all required features are present
        feature_vector = []
        for feature in self.feature_names:
            if feature in data:
                feature_vector.append(data[feature])
            else:
                print(f"⚠️ Warning: Missing feature '{feature}', using default value 0")
                feature_vector.append(0)

        # Convert to numpy array and reshape for single prediction
        feature_array = np.array(feature_vector).reshape(1, -1)

        # Apply scaling
        scaled_features = self.scaler.transform(feature_array)

        return scaled_features

    def predict(self, patient_data):
        """Make prediction for a single patient"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(patient_data)

            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0][1]

            # Determine risk level
            if probability < 0.3:
                risk_level = "LOW"
                risk_color = "🟢"
            elif probability < 0.7:
                risk_level = "MODERATE"
                risk_color = "🟡"
            else:
                risk_level = "HIGH"
                risk_color = "🔴"

            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'risk_color': risk_color
            }

        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None

    def print_prediction_result(self, patient_data, result):
        """Print formatted prediction result"""
        print("\n" + "="*50)
        print("🩺 STROKE RISK PREDICTION RESULT")
        print("="*50)

        print("\n👤 Patient Information:")
        for key, value in patient_data.items():
            if key != 'family_history_stroke':  # Skip synthetic feature
                print(f"   {key.replace('_', ' ').title()}: {value}")

        print(f"\n🎯 Prediction Results:")
        print(f"   Risk Level: {result['risk_color']} {result['risk_level']}")
        print(f"   Stroke Probability: {result['probability']:.1%}")
        print(f"   Binary Prediction: {'Stroke Risk' if result['prediction'] == 1 else 'No Immediate Risk'}")

        print(f"\n💡 Interpretation:")
        if result['risk_level'] == 'LOW':
            print("   • Low stroke risk based on current factors")
            print("   • Maintain healthy lifestyle and regular check-ups")
        elif result['risk_level'] == 'MODERATE':
            print("   • Moderate stroke risk - some concerning factors present")
            print("   • Consider lifestyle modifications and consult healthcare provider")
        else:
            print("   • High stroke risk - multiple risk factors detected")
            print("   • Immediate medical consultation strongly recommended")

        print("\n⚠️  IMPORTANT DISCLAIMER:")
        print("   This prediction is for educational purposes only.")
        print("   Always consult qualified healthcare professionals for medical advice.")
        print("="*50)

def create_sample_patients():
    """Create sample patient data for testing"""
    patients = []

    # Low risk patient - young, healthy
    patients.append({
        'name': 'John Doe (Low Risk)',
        'data': {
            'gender': 'Male',
            'age': 25,
            'hypertension': 0,
            'heart_disease': 0,
            'ever_married': 'No',
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'avg_glucose_level': 85.0,
            'bmi': 24.5,
            'smoking_status': 'never smoked',
            'family_history_stroke': 0
        }
    })

    # Moderate risk patient - middle-aged with some risk factors
    patients.append({
        'name': 'Jane Smith (Moderate Risk)',
        'data': {
            'gender': 'Female',
            'age': 55,
            'hypertension': 1,
            'heart_disease': 0,
            'ever_married': 'Yes',
            'work_type': 'Self-employed',
            'Residence_type': 'Rural',
            'avg_glucose_level': 120.0,
            'bmi': 29.8,
            'smoking_status': 'formerly smoked',
            'family_history_stroke': 1
        }
    })

    # High risk patient - elderly with multiple risk factors
    patients.append({
        'name': 'Robert Johnson (High Risk)',
        'data': {
            'gender': 'Male',
            'age': 78,
            'hypertension': 1,
            'heart_disease': 1,
            'ever_married': 'Yes',
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'avg_glucose_level': 180.0,
            'bmi': 32.1,
            'smoking_status': 'smokes',
            'family_history_stroke': 1
        }
    })

    return patients

def main():
    """Main function to demonstrate model usage"""
    print("🧠 Brain Stroke Risk Prediction Model Test")
    print("=" * 60)

    # Check if model file exists
    model_path = Path("models/stroke_prediction_model.pkl")
    if not model_path.exists():
        print("❌ Model file not found. Please run training first:")
        print("   python train_stroke_model.py")
        return 1

    # Initialize prediction service
    try:
        predictor = StrokePredictionService()
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return 1

    # Create sample patients
    sample_patients = create_sample_patients()

    # Make predictions for sample patients
    print("\n🧪 Testing with sample patients...")
    for i, patient in enumerate(sample_patients, 1):
        print(f"\n--- Patient {i}: {patient['name']} ---")

        result = predictor.predict(patient['data'])
        if result:
            predictor.print_prediction_result(patient['data'], result)
        else:
            print("❌ Failed to make prediction for this patient")

    # Interactive mode
    print("\n" + "="*60)
    print("🔧 INTERACTIVE MODE")
    print("="*60)
    print("You can now enter your own patient data for prediction.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Enter 'demo' for guided input or 'exit' to quit: ").strip().lower()

            if user_input == 'exit':
                break
            elif user_input == 'demo':
                # Simple guided input for demo
                print("\n📝 Enter patient information:")

                custom_patient = {}
                custom_patient['gender'] = input("Gender (Male/Female/Other): ").strip()
                custom_patient['age'] = float(input("Age: "))
                custom_patient['hypertension'] = int(input("Hypertension (0=No, 1=Yes): "))
                custom_patient['heart_disease'] = int(input("Heart Disease (0=No, 1=Yes): "))
                custom_patient['ever_married'] = input("Ever Married (Yes/No): ").strip()
                custom_patient['work_type'] = input("Work Type (Private/Self-employed/Govt_job/children/Never_worked): ").strip()
                custom_patient['Residence_type'] = input("Residence (Urban/Rural): ").strip()
                custom_patient['avg_glucose_level'] = float(input("Average Glucose Level: "))
                custom_patient['bmi'] = float(input("BMI: "))
                custom_patient['smoking_status'] = input("Smoking Status (never smoked/formerly smoked/smokes/Unknown): ").strip()
                custom_patient['family_history_stroke'] = int(input("Family History of Stroke (0=No, 1=Yes): "))

                result = predictor.predict(custom_patient)
                if result:
                    predictor.print_prediction_result(custom_patient, result)
                else:
                    print("❌ Failed to make prediction")
            else:
                print("Invalid input. Please enter 'demo' or 'exit'.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again with valid input.")

    print("\n✅ Testing completed!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
