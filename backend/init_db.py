#!/usr/bin/env python3
"""
Database Initialization Script for Brain Stroke Prediction System
Creates all tables and optionally seeds demo data
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from models import db, User, Prediction, init_db, seed_demo_data
from config import get_config

def init_database():
    """Initialize the database with all tables"""
    print("🔧 Initializing Brain Stroke Prediction Database...")
    print("=" * 50)

    # Create Flask app
    app = create_app()

    with app.app_context():
        # Get database URL for display
        db_url = app.config['SQLALCHEMY_DATABASE_URI']
        db_type = "PostgreSQL" if "postgresql" in db_url else "SQLite"

        print(f"📍 Database Type: {db_type}")
        print(f"🔗 Database URL: {db_url.split('@')[-1] if '@' in db_url else db_url}")
        print()

        try:
            # Test database connection
            print("🔌 Testing database connection...")
            db.engine.connect()
            print("✅ Database connection successful!")

            # Drop all tables (if they exist)
            print("\n🗑️  Dropping existing tables...")
            db.drop_all()
            print("✅ Existing tables dropped!")

            # Create all tables
            print("\n📊 Creating database tables...")
            db.create_all()

            # Verify tables were created
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()

            print(f"✅ Created {len(tables)} tables:")
            for table in tables:
                print(f"   - {table}")

            # Create indexes for better performance
            print("\n📈 Creating database indexes...")
            try:
                # Index on user email for faster lookups
                db.engine.execute('''
                    CREATE INDEX IF NOT EXISTS idx_users_email
                    ON users(email)
                ''')

                # Index on prediction user_id and created_at for faster queries
                db.engine.execute('''
                    CREATE INDEX IF NOT EXISTS idx_predictions_user_created
                    ON predictions(user_id, created_at DESC)
                ''')

                # Index on prediction risk_level for statistics
                db.engine.execute('''
                    CREATE INDEX IF NOT EXISTS idx_predictions_risk_level
                    ON predictions(risk_level)
                ''')

                print("✅ Database indexes created!")

            except Exception as e:
                print(f"⚠️  Index creation warning: {e}")

            print(f"\n🎉 Database initialization completed successfully!")
            return True

        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            return False

def seed_database():
    """Seed database with demo data"""
    print("\n🌱 Seeding database with demo data...")

    app = create_app()

    with app.app_context():
        try:
            # Check if demo user already exists
            demo_user = User.find_by_email('demo@strokeprediction.com')

            if demo_user:
                print("ℹ️  Demo user already exists, skipping seed data...")
                return True

            # Create demo user
            print("👤 Creating demo user...")
            demo_user = User.create_user(
                name='Demo User',
                email='demo@strokeprediction.com',
                password='demo123'
            )
            print(f"✅ Demo user created: {demo_user.email}")

            # Create sample predictions
            print("📊 Creating sample predictions...")
            sample_predictions = [
                {
                    'prediction_data': {
                        'risk_level': 'LOW',
                        'probability_score': 0.15,
                        'confidence': 'HIGH',
                        'risk_factors': [
                            'Age within normal range',
                            'Healthy BMI range',
                            'No smoking history',
                            'Normal blood pressure'
                        ],
                        'recommendations': [
                            'Maintain current healthy lifestyle',
                            'Continue regular exercise routine',
                            'Schedule annual health checkups',
                            'Monitor blood pressure periodically'
                        ],
                        'model_name': 'Logistic Regression',
                        'model_version': '1.0.0'
                    },
                    'patient_data': {
                        'age': 28,
                        'gender': 'Female',
                        'hypertension': 0,
                        'heart_disease': 0,
                        'ever_married': 'Yes',
                        'work_type': 'Private',
                        'Residence_type': 'Urban',
                        'avg_glucose_level': 85.5,
                        'bmi': 22.3,
                        'smoking_status': 'never smoked',
                        'family_history_stroke': 0,
                        'alcohol_consumption': 'Occasionally'
                    }
                },
                {
                    'prediction_data': {
                        'risk_level': 'MODERATE',
                        'probability_score': 0.45,
                        'confidence': 'MEDIUM',
                        'risk_factors': [
                            'Elevated glucose level (140.2 mg/dL)',
                            'Overweight BMI (28.7)',
                            'History of hypertension',
                            'Family history of stroke',
                            'Former smoking habit'
                        ],
                        'recommendations': [
                            'Consult healthcare provider immediately',
                            'Monitor blood pressure daily',
                            'Consider dietary modifications',
                            'Increase physical activity',
                            'Regular glucose level monitoring'
                        ],
                        'model_name': 'Logistic Regression',
                        'model_version': '1.0.0'
                    },
                    'patient_data': {
                        'age': 52,
                        'gender': 'Male',
                        'hypertension': 1,
                        'heart_disease': 0,
                        'ever_married': 'Yes',
                        'work_type': 'Govt_job',
                        'Residence_type': 'Rural',
                        'avg_glucose_level': 140.2,
                        'bmi': 28.7,
                        'smoking_status': 'formerly smoked',
                        'family_history_stroke': 1,
                        'alcohol_consumption': 'Regularly'
                    }
                },
                {
                    'prediction_data': {
                        'risk_level': 'HIGH',
                        'probability_score': 0.78,
                        'confidence': 'HIGH',
                        'risk_factors': [
                            'Advanced age (67 years)',
                            'Obese BMI (32.8)',
                            'Hypertension present',
                            'Heart disease history',
                            'Current smoker',
                            'Very high glucose level (195.6 mg/dL)',
                            'Family history of stroke',
                            'Heavy alcohol consumption'
                        ],
                        'recommendations': [
                            'URGENT: Schedule immediate medical consultation',
                            'Consider emergency evaluation if symptoms present',
                            'Implement comprehensive stroke prevention plan',
                            'Immediate smoking cessation program',
                            'Strict dietary modifications required',
                            'Regular cardiology follow-up',
                            'Blood pressure medication adjustment',
                            'Diabetes management optimization'
                        ],
                        'model_name': 'Logistic Regression',
                        'model_version': '1.0.0'
                    },
                    'patient_data': {
                        'age': 67,
                        'gender': 'Male',
                        'hypertension': 1,
                        'heart_disease': 1,
                        'ever_married': 'Yes',
                        'work_type': 'Self-employed',
                        'Residence_type': 'Urban',
                        'avg_glucose_level': 195.6,
                        'bmi': 32.8,
                        'smoking_status': 'smokes',
                        'family_history_stroke': 1,
                        'alcohol_consumption': 'Heavy'
                    }
                }
            ]

            # Create prediction records
            for i, pred_data in enumerate(sample_predictions, 1):
                prediction = Prediction(
                    user_id=demo_user.id,
                    prediction_data=pred_data['prediction_data'],
                    patient_data=pred_data['patient_data']
                )
                db.session.add(prediction)
                print(f"   ✅ Sample prediction {i}: {pred_data['prediction_data']['risk_level']} risk")

            db.session.commit()

            # Verify data was created
            user_count = User.query.count()
            prediction_count = Prediction.query.count()

            print(f"\n📈 Database seeded successfully!")
            print(f"   👥 Users: {user_count}")
            print(f"   📊 Predictions: {prediction_count}")

            print(f"\n🔐 Demo Login Credentials:")
            print(f"   📧 Email: demo@strokeprediction.com")
            print(f"   🔑 Password: demo123")

            return True

        except Exception as e:
            print(f"❌ Database seeding failed: {e}")
            db.session.rollback()
            return False

def check_database():
    """Check database status and display information"""
    print("🔍 Checking database status...")
    print("=" * 40)

    app = create_app()

    with app.app_context():
        try:
            # Test connection
            db.engine.connect()
            db_url = app.config['SQLALCHEMY_DATABASE_URI']
            db_type = "PostgreSQL" if "postgresql" in db_url else "SQLite"

            print(f"✅ Database connected: {db_type}")

            # Check tables
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()

            if tables:
                print(f"📊 Tables ({len(tables)}):")
                for table in tables:
                    try:
                        count = db.engine.execute(f"SELECT COUNT(*) FROM {table}").scalar()
                        print(f"   - {table}: {count} records")
                    except:
                        print(f"   - {table}: unable to count")
            else:
                print("⚠️  No tables found - database needs initialization")

            return True

        except Exception as e:
            print(f"❌ Database check failed: {e}")
            return False

def main():
    """Main function to handle command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='Database management for Brain Stroke Prediction System')
    parser.add_argument('action', choices=['init', 'seed', 'check', 'reset'],
                       help='Action to perform')
    parser.add_argument('--force', action='store_true',
                       help='Force action without confirmation')

    args = parser.parse_args()

    if args.action == 'init':
        print("🚀 Initializing database...")
        if init_database():
            print("\n✅ Database initialization completed successfully!")
        else:
            print("\n❌ Database initialization failed!")
            sys.exit(1)

    elif args.action == 'seed':
        print("🌱 Seeding database with demo data...")
        if not args.force:
            confirm = input("This will add demo data to your database. Continue? (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Cancelled by user")
                sys.exit(1)

        if seed_database():
            print("\n✅ Database seeding completed successfully!")
        else:
            print("\n❌ Database seeding failed!")
            sys.exit(1)

    elif args.action == 'check':
        check_database()

    elif args.action == 'reset':
        print("⚠️  WARNING: This will destroy ALL data in the database!")
        if not args.force:
            confirm = input("Are you absolutely sure? This cannot be undone! (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Cancelled by user")
                sys.exit(1)

        print("\n🔄 Resetting database...")
        if init_database() and seed_database():
            print("\n✅ Database reset completed successfully!")
        else:
            print("\n❌ Database reset failed!")
            sys.exit(1)

if __name__ == '__main__':
    main()
