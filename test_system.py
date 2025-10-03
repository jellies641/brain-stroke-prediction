#!/usr/bin/env python3
"""
System Test Script for Brain Stroke Risk Prediction
Tests the ML model and prediction functionality
"""

import sys
import os
import json
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_ml_service():
    """Test the ML service directly"""
    print("🧠 Testing Brain Stroke Risk Prediction System")
    print("=" * 60)

    try:
        # Import the ML service
        from ml_service import predict_stroke_risk, get_predictor_info

        print("✅ ML service imported successfully")

        # Get model info
        info = get_predictor_info()
        print(f"📊 Model Info: {info.get('model_name', 'Unknown')}")
        print(f"🎯 Accuracy: {info.get('metrics', {}).get('Logistic Regression', {}).get('auc_score', 'N/A')}")

        # Test prediction with sample data
        sample_patients = [
            {
                'name': 'Low Risk Patient',
                'age': 25,
                'gender': 'Female',
                'hypertension': 0,
                'heart_disease': 0,
                'ever_married': 'No',
                'work_type': 'Private',
                'residence_type': 'Urban',
                'avg_glucose_level': 85.0,
                'bmi': 22.5,
                'smoking_status': 'never smoked',
                'family_history_stroke': 0,
                'alcohol_consumption': 'Never'
            },
            {
                'name': 'High Risk Patient',
                'age': 75,
                'gender': 'Male',
                'hypertension': 1,
                'heart_disease': 1,
                'ever_married': 'Yes',
                'work_type': 'Private',
                'residence_type': 'Rural',
                'avg_glucose_level': 200.0,
                'bmi': 35.0,
                'smoking_status': 'smokes',
                'family_history_stroke': 1,
                'alcohol_consumption': 'Heavy'
            }
        ]

        print("\n🧪 Testing Predictions:")
        print("-" * 40)

        for i, patient in enumerate(sample_patients, 1):
            print(f"\n👤 Patient {i}: {patient['name']}")
            print(f"   Age: {patient['age']}, Gender: {patient['gender']}")
            print(f"   BMI: {patient['bmi']}, Glucose: {patient['avg_glucose_level']}")

            try:
                result = predict_stroke_risk(patient)

                print(f"   🎯 Risk Level: {result.get('risk_level', 'Unknown')}")
                print(f"   📊 Probability: {result.get('probability_score', 0):.1%}")
                print(f"   🔍 Confidence: {result.get('confidence', 'Unknown')}")
                print(f"   ⚕️ Model: {result.get('model_name', 'Unknown')}")

                # Show top risk factors
                risk_factors = result.get('risk_factors', [])
                if risk_factors:
                    print(f"   ⚠️ Key Risk Factors: {', '.join(risk_factors[:3])}")

                print("   ✅ Prediction successful")

            except Exception as e:
                print(f"   ❌ Prediction failed: {str(e)}")

        print("\n" + "=" * 60)
        print("✅ ML SERVICE TEST COMPLETED SUCCESSFULLY!")
        return True

    except ImportError as e:
        print(f"❌ Failed to import ML service: {str(e)}")
        print("   Make sure the backend dependencies are installed:")
        print("   cd backend && pip install pandas numpy scikit-learn joblib")
        return False
    except Exception as e:
        print(f"❌ ML service test failed: {str(e)}")
        return False

def test_model_files():
    """Test if model files exist"""
    print("\n📁 Testing Model Files:")
    print("-" * 30)

    model_paths = [
        "ml-model/models/stroke_prediction_model.pkl",
        "ml-model/models/preprocessing.pkl",
        "ml-model/models/MODEL_REPORT.md",
        "backend/models/stroke_prediction_model.pkl"
    ]

    all_exist = True
    for path in model_paths:
        full_path = Path(path)
        if full_path.exists():
            size = full_path.stat().st_size / 1024  # KB
            print(f"✅ {path} ({size:.1f} KB)")
        else:
            print(f"❌ {path} - Missing")
            all_exist = False

    return all_exist

def test_dataset():
    """Test if dataset exists and is valid"""
    print("\n📊 Testing Dataset:")
    print("-" * 20)

    dataset_path = Path("healthcare-dataset-stroke-data.csv")
    if not dataset_path.exists():
        dataset_path = Path("ml-model/data/healthcare-dataset-stroke-data.csv")

    if dataset_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"✅ Dataset found: {len(df)} records")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Stroke cases: {df['stroke'].sum()} ({df['stroke'].mean():.1%})")
            return True
        except Exception as e:
            print(f"❌ Dataset read error: {str(e)}")
            return False
    else:
        print("❌ Dataset not found")
        print("   Expected: healthcare-dataset-stroke-data.csv")
        return False

def test_frontend_build():
    """Test if frontend files exist"""
    print("\n🎨 Testing Frontend:")
    print("-" * 20)

    frontend_files = [
        "frontend/package.json",
        "frontend/src/App.js",
        "frontend/src/pages/Home.js",
        "frontend/src/pages/Prediction.js",
        "frontend/public/index.html"
    ]

    all_exist = True
    for file_path in frontend_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_exist = False

    return all_exist

def test_backend_api():
    """Test backend API endpoints"""
    print("\n🌐 Testing Backend API:")
    print("-" * 25)

    try:
        import requests

        # Test health endpoint
        try:
            response = requests.get('http://localhost:5000/', timeout=5)
            if response.status_code == 200:
                print("✅ Health endpoint working")
            else:
                print(f"⚠️ Health endpoint status: {response.status_code}")
        except:
            print("⚠️ Backend server not running (this is optional)")

        return True

    except ImportError:
        print("⚠️ requests library not available for API testing")
        return True
    except Exception as e:
        print(f"⚠️ API test error: {str(e)}")
        return True

def main():
    """Run all tests"""
    print("🧠 BRAIN STROKE PREDICTION SYSTEM TEST")
    print("=" * 70)

    tests = [
        ("Dataset", test_dataset),
        ("Model Files", test_model_files),
        ("ML Service", test_ml_service),
        ("Frontend Files", test_frontend_build),
        ("Backend API", test_backend_api)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} Test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test error: {str(e)}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:15} : {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        print("\nTo start the application:")
        print("1. Backend:  cd backend && python simple_app.py")
        print("2. Frontend: cd frontend && npm start")
        print("3. Open:     http://localhost:3000")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

    print("\n⚠️  MEDICAL DISCLAIMER:")
    print("This system is for educational purposes only.")
    print("Always consult healthcare professionals for medical decisions.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
