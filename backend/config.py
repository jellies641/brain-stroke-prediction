import os
from datetime import timedelta

class Config:
    """Base configuration class"""

    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False

    # Database Configuration - Default to SQLite for simplicity
    DATABASE_URL = os.environ.get('DATABASE_URL')

    # Always use SQLite unless PostgreSQL is explicitly configured
    if DATABASE_URL and DATABASE_URL.startswith(('postgres://', 'postgresql://')):
        if DATABASE_URL.startswith('postgres://'):
            # Fix for Railway/Heroku DATABASE_URL format
            DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        SQLALCHEMY_DATABASE_URI = DATABASE_URL
    else:
        # Default to SQLite for both local and production
        SQLALCHEMY_DATABASE_URI = 'sqlite:///stroke_prediction.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False

    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)

    # CORS Configuration - Explicit frontend origins
    CORS_ORIGINS = ['https://web-production-4ea93.up.railway.app', 'http://localhost:3000']
    CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    CORS_HEADERS = ['Content-Type', 'Authorization', 'Accept']
    CORS_SUPPORTS_CREDENTIALS = False

    # ML Model Configuration
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml-model', 'models')

    # Security Configuration
    BCRYPT_LOG_ROUNDS = 12
    WTF_CSRF_ENABLED = False  # Disabled for API

    # Email Configuration (for future use)
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'localhost')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True  # Log SQL queries in development

    # Use SQLite for local development
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///stroke_prediction_dev.db'


class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # In-memory SQLite for tests
    WTF_CSRF_ENABLED = False
    BCRYPT_LOG_ROUNDS = 4  # Faster hashing for tests


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False

    # Security headers
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # CORS for production (Railway/Render auto-generated URLs)
    CORS_ORIGINS = [
        'https://*.railway.app',
        'https://*.onrender.com',
        'https://*.vercel.app',
        'https://*.netlify.app',
        os.environ.get('FRONTEND_URL', '')
    ]

    @staticmethod
    def init_app(app):
        Config.init_app(app)

        # Log to stderr in production
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.WARNING)
        app.logger.addHandler(file_handler)


class RailwayConfig(ProductionConfig):
    """Railway hosting specific configuration"""

    @staticmethod
    def init_app(app):
        ProductionConfig.init_app(app)

        # Railway specific settings
        if 'RAILWAY_ENVIRONMENT' in os.environ:
            # Railway uses PORT environment variable
            app.config['PORT'] = int(os.environ.get('PORT', 5000))


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'railway': RailwayConfig,
    'default': DevelopmentConfig
}

# Helper function to get config
def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')

    # Railway detection
    if 'RAILWAY_ENVIRONMENT' in os.environ:
        return config['railway']

    return config.get(env, config['default'])
