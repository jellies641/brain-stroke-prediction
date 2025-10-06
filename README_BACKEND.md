# Brain Stroke Risk Prediction - Backend API

🧠 **Machine Learning-powered REST API for stroke risk assessment**

## Overview

This is the backend API service for the Brain Stroke Risk Prediction application. It provides machine learning-powered stroke risk assessment through a clean REST API built with Flask and Python.

## Features

- 🤖 **AI-Powered Predictions**: SVM-based machine learning model with 99.5% accuracy
- 🔐 **Authentication**: JWT-based user authentication and authorization
- 📊 **Analytics**: Prediction history and statistics tracking
- 🗄️ **Database**: SQLite/PostgreSQL support with SQLAlchemy ORM
- 🚀 **Production Ready**: Configured for Railway deployment
- 📖 **API Documentation**: RESTful endpoints with JSON responses

## API Endpoints

### Health & Info
- `GET /` - Health check and service status
- `GET /api/info` - API information and ML model details

### Authentication
- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/validate` - Token validation

### Predictions
- `POST /api/predict` - Generate stroke risk prediction
- `GET /api/history` - Get user's prediction history
- `GET /api/statistics` - Get user statistics
- `DELETE /api/predictions/<id>` - Delete specific prediction

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/jellies641/brain-stroke-prediction.git
   cd brain-stroke-prediction
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env with your configuration
   ```

4. **Run the API**
   ```bash
   python start.py
   ```

The API will be available at `http://localhost:5000`

### Railway Deployment

This backend is configured for one-click Railway deployment:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `production` |
| `SECRET_KEY` | JWT secret key | Required |
| `DATABASE_URL` | Database connection string | SQLite |
| `PYTHONPATH` | Python module path | `/app/backend:/app/ml-model` |
| `FORCE_SQLITE` | Force SQLite usage | `false` |

## API Usage Examples

### Health Check
```bash
curl https://your-api-url.railway.app/
```

### User Registration
```bash
curl -X POST https://your-api-url.railway.app/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "password": "securepassword123"
  }'
```

### Stroke Risk Prediction
```bash
curl -X POST https://your-api-url.railway.app/api/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "age": 65,
    "gender": "Male",
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

## Response Format

All API responses follow this format:

```json
{
  "status": "success|error",
  "message": "Response message",
  "data": { ... },
  "timestamp": "2023-10-06T14:30:00Z"
}
```

### Prediction Response Example
```json
{
  "status": "success",
  "message": "Prediction generated successfully",
  "data": {
    "prediction": {
      "risk_level": "HIGH",
      "risk_score": 0.87,
      "risk_percentage": 87,
      "confidence": 0.95
    },
    "patient_summary": {
      "age": 65,
      "gender": "Male",
      "risk_factors": ["hypertension", "high_glucose", "overweight"]
    },
    "recommendations": [
      "Consult with a healthcare professional immediately",
      "Monitor blood pressure regularly",
      "Maintain a healthy diet low in sodium"
    ]
  },
  "timestamp": "2023-10-06T14:30:00Z"
}
```

## Machine Learning Model

- **Algorithm**: Support Vector Machine (SVM)
- **Accuracy**: 99.5%
- **Features**: 10 health indicators
- **Training Data**: Healthcare stroke dataset
- **Risk Categories**: Low, Moderate, High

## Technology Stack

- **Framework**: Flask 2.3.3
- **Database**: SQLAlchemy with SQLite/PostgreSQL
- **ML**: scikit-learn, pandas, numpy
- **Authentication**: Flask-JWT-Extended
- **Deployment**: Railway
- **Python**: 3.10+

## Project Structure

```
├── backend/
│   ├── app.py              # Main Flask application
│   ├── models.py           # Database models
│   ├── ml_service.py       # ML prediction service
│   ├── config.py           # Configuration
│   └── requirements.txt    # Python dependencies
├── ml-model/               # ML model files
├── start.py               # Railway startup script
├── nixpacks.toml          # Railway build configuration
└── railway.json           # Railway deployment config
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Medical Disclaimer

⚠️ **Important**: This tool is for educational and informational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## Support

For issues and questions:
- 📧 Create an issue on GitHub
- 📖 Check the API documentation
- 🚀 Deploy on Railway for production use

---

**Made with ❤️ for healthcare innovation**