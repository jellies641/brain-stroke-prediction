#!/usr/bin/env python3
"""
Simplified Flask API for Brain Stroke Risk Prediction
Test version of the stroke prediction service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import traceback
import json
import os

# Import our ML service
try:
    from ml_service import predict_stroke_risk, get_predictor_info
    ML_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML service not available - {e}")
    ML_SERVICE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Test data for fallback
SAMPLE_PREDICTION = {
    'prediction': 0,
    'probability_score': 0.25,
    'risk_level': 'MODERATE',
    'confidence': 'MEDIUM',
    'risk_factors': ['Sample risk factor analysis'],
    'recommendations': [
        'Consult with healthcare provider',
        'Maintain healthy lifestyle',
        'Monitor blood pressure regularly'
    ],
    'model_version': '1.0.0-test',
    'model_name': 'Test Model'
}

# In-memory storage for demo purposes
# In production, you would use a proper database
prediction_storage = []
STORAGE_FILE = 'predictions.json'

def load_predictions():
    """Load predictions from file"""
    global prediction_storage
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'r') as f:
                prediction_storage = json.load(f)
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            prediction_storage = []
    else:
        prediction_storage = []

def save_predictions():
    """Save predictions to file"""
    try:
        with open(STORAGE_FILE, 'w') as f:
            json.dump(prediction_storage, f)
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")

def get_next_prediction_id():
    """Get next available prediction ID"""
    if not prediction_storage:
        return 1
    return max(p.get('id', 0) for p in prediction_storage) + 1

# Load predictions on startup
load_predictions()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Brain Stroke Risk Prediction API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'ml_service': 'available' if ML_SERVICE_AVAILABLE else 'unavailable'
    })

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    try:
        if ML_SERVICE_AVAILABLE:
            model_info = get_predictor_info()
        else:
            model_info = {'status': 'ML service not available'}

        return jsonify({
            'api_version': '1.0.0',
            'endpoints': {
                '/': 'Health check',
                '/api/info': 'API information',
                '/api/predict': 'Stroke risk prediction',
                '/api/history': 'Get prediction history',
                '/api/statistics': 'Get user statistics',
                '/api/predictions/<id>': 'Delete specific prediction',
                '/api/test': 'Test prediction with sample data'
            },
            'model_info': model_info,
            'ml_service_status': 'available' if ML_SERVICE_AVAILABLE else 'unavailable'
        })
    except Exception as e:
        logger.error(f"Error in api_info: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        # Define required fields
        required_fields = [
            'age', 'gender', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'avg_glucose_level', 'bmi', 'smoking_status'
        ]

        # Check for missing fields
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)

        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400

        # Validate data types and ranges
        try:
            age = float(data['age'])
            if age < 0 or age > 120:
                return jsonify({'error': 'Age must be between 0 and 120'}), 400

            glucose_level = float(data['avg_glucose_level'])
            if glucose_level < 0 or glucose_level > 500:
                return jsonify({'error': 'Glucose level must be between 0 and 500'}), 400

            bmi = float(data['bmi'])
            if bmi < 10 or bmi > 60:
                return jsonify({'error': 'BMI must be between 10 and 60'}), 400

        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid numeric values: {str(e)}'}), 400

        # Prepare data for ML model
        patient_data = {
            'age': age,
            'gender': str(data['gender']),
            'hypertension': int(bool(data['hypertension'])),
            'heart_disease': int(bool(data['heart_disease'])),
            'ever_married': str(data['ever_married']),
            'work_type': str(data['work_type']),
            'Residence_type': str(data.get('residence_type', data.get('Residence_type', 'Urban'))),
            'avg_glucose_level': glucose_level,
            'bmi': bmi,
            'smoking_status': str(data['smoking_status']),
            'family_history_stroke': int(bool(data.get('family_history_stroke', 0))),
            'alcohol_consumption': str(data.get('alcohol_consumption', 'Never'))
        }

        logger.info(f"Processing prediction request for patient: age={age}, gender={patient_data['gender']}")

        # Get ML prediction
        if ML_SERVICE_AVAILABLE:
            try:
                prediction_result = predict_stroke_risk(patient_data)
            except Exception as e:
                logger.error(f"ML prediction failed: {str(e)}")
                prediction_result = SAMPLE_PREDICTION.copy()
                prediction_result['error'] = f'ML service error: {str(e)}'
        else:
            logger.warning("Using fallback prediction - ML service not available")
            prediction_result = SAMPLE_PREDICTION.copy()
            prediction_result['note'] = 'Using fallback prediction - ML service not available'

        # Store prediction in memory/file
        prediction_id = get_next_prediction_id()
        stored_prediction = {
            'id': prediction_id,
            'created_at': datetime.now().isoformat(),
            'risk_level': prediction_result.get('risk_level', 'UNKNOWN'),
            'probability_score': prediction_result.get('probability_score', 0.0),
            'risk_factors': prediction_result.get('risk_factors', []),
            'recommendations': prediction_result.get('recommendations', []),
            'patient_summary': {
                'age': int(age),
                'gender': patient_data['gender'],
                'bmi': round(bmi, 1),
                'glucose_level': round(glucose_level, 1)
            },
            'model_info': {
                'model_name': prediction_result.get('model_name', 'Unknown'),
                'model_version': prediction_result.get('model_version', '1.0.0')
            }
        }

        prediction_storage.append(stored_prediction)
        save_predictions()

        # Format response
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_result,
            'patient_summary': {
                'age': int(age),
                'gender': patient_data['gender'],
                'bmi': round(bmi, 1),
                'glucose_level': round(glucose_level, 1)
            },
            'disclaimer': {
                'message': 'MEDICAL DISCLAIMER: This prediction is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.',
                'importance': 'CRITICAL'
            }
        }

        logger.info(f"Prediction completed: {prediction_result.get('risk_level', 'UNKNOWN')} risk")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    try:
        # Return all stored predictions
        return jsonify({
            'status': 'success',
            'predictions': prediction_storage,
            'total_count': len(prediction_storage),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in history endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve history',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get user statistics"""
    try:
        if not prediction_storage:
            stats = {
                'total_predictions': 0,
                'risk_distribution': {'LOW': 0, 'MODERATE': 0, 'HIGH': 0},
                'average_age': 0,
                'recent_predictions': 0,
                'success_rate': 100.0
            }
        else:
            # Calculate statistics
            total = len(prediction_storage)
            risk_counts = {'LOW': 0, 'MODERATE': 0, 'HIGH': 0}
            ages = []

            # Count recent predictions (last 7 days)
            recent_count = 0
            now = datetime.now()

            for pred in prediction_storage:
                risk_level = pred.get('risk_level', 'UNKNOWN')
                if risk_level in risk_counts:
                    risk_counts[risk_level] += 1

                age = pred.get('patient_summary', {}).get('age', 0)
                if age > 0:
                    ages.append(age)

                # Check if prediction is recent
                try:
                    pred_date = datetime.fromisoformat(pred.get('created_at', now.isoformat()))
                    if (now - pred_date).days <= 7:
                        recent_count += 1
                except:
                    pass  # Skip invalid dates

            avg_age = sum(ages) / len(ages) if ages else 0

            stats = {
                'total_predictions': total,
                'risk_distribution': risk_counts,
                'average_age': round(avg_age, 1),
                'recent_predictions': recent_count,
                'success_rate': 100.0  # Assume all predictions are successful
            }

        return jsonify({
            'status': 'success',
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in statistics endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve statistics',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/predictions/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    """Delete a specific prediction"""
    try:
        global prediction_storage

        # Find prediction by ID
        prediction_to_delete = None
        for i, pred in enumerate(prediction_storage):
            if pred.get('id') == prediction_id:
                prediction_to_delete = prediction_storage.pop(i)
                break

        if prediction_to_delete is None:
            return jsonify({
                'error': 'Prediction not found',
                'message': f'No prediction found with ID {prediction_id}'
            }), 404

        # Save updated predictions
        save_predictions()

        return jsonify({
            'status': 'success',
            'message': f'Prediction {prediction_id} deleted successfully',
            'deleted_prediction': prediction_to_delete,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in delete prediction endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to delete prediction',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/test', methods=['GET'])
def test_prediction():
    """Test endpoint with sample data"""
    try:
        # Sample test data
        sample_patients = [
            {
                'name': 'Low Risk Patient',
                'data': {
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
                }
            },
            {
                'name': 'High Risk Patient',
                'data': {
                    'age': 70,
                    'gender': 'Male',
                    'hypertension': 1,
                    'heart_disease': 1,
                    'ever_married': 'Yes',
                    'work_type': 'Private',
                    'residence_type': 'Rural',
                    'avg_glucose_level': 180.0,
                    'bmi': 32.1,
                    'smoking_status': 'smokes',
                    'family_history_stroke': 1,
                    'alcohol_consumption': 'Regularly'
                }
            }
        ]

        results = []
        for patient in sample_patients:
            if ML_SERVICE_AVAILABLE:
                try:
                    prediction = predict_stroke_risk(patient['data'])
                except Exception as e:
                    logger.error(f"Test prediction failed: {str(e)}")
                    prediction = SAMPLE_PREDICTION.copy()
                    prediction['error'] = str(e)
            else:
                prediction = SAMPLE_PREDICTION.copy()

            results.append({
                'patient_name': patient['name'],
                'patient_data': patient['data'],
                'prediction': prediction
            })

        return jsonify({
            'status': 'success',
            'test_results': results,
            'ml_service_status': 'available' if ML_SERVICE_AVAILABLE else 'unavailable',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return jsonify({'error': 'Test failed', 'message': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': ['/', '/api/info', '/api/predict', '/api/history', '/api/statistics', '/api/predictions/<id>', '/api/test']
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The request method is not allowed for this endpoint'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    print("🧠 Starting Brain Stroke Risk Prediction API...")
    print("=" * 50)
    print("API Endpoints:")
    print("  GET    /                 - Health check")
    print("  GET    /api/info         - API information")
    print("  POST   /api/predict      - Stroke risk prediction")
    print("  GET    /api/history      - Get prediction history")
    print("  GET    /api/statistics   - Get user statistics")
    print("  DELETE /api/predictions/<id> - Delete specific prediction")
    print("  GET    /api/test         - Test with sample data")
    print("=" * 50)
    print(f"ML Service: {'✅ Available' if ML_SERVICE_AVAILABLE else '❌ Not Available'}")
    print("=" * 50)

    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Avoid reloading issues with ML models
    )
