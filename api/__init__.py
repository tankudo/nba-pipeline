"""
API Package for Next Best Action System

This package implements the REST API interface including:
- FastAPI application and router definitions
- Request/response models
- API endpoint handlers
- Authentication middleware
- Rate limiting
- API documentation

Classes:
    PredictionRequest: Pydantic model for prediction requests
    PredictionResponse: Pydantic model for prediction responses
    APIConfig: Configuration for API settings

Variables:
    app: FastAPI application instance
    router: API router instance

Usage:
    from api import app, PredictionRequest
    prediction_request = PredictionRequest(user_id='123', features={...})
"""

from fastapi import FastAPI
from .app import app, router
from .models import (
    PredictionRequest,
    PredictionResponse,
    APIConfig
)
from .middleware import (
    AuthMiddleware,
    RateLimiter
)

__version__ = '1.0.0'

# API configuration settings
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'

# Initialize FastAPI application with middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimiter)

# Include routers
app.include_router(router, prefix=API_PREFIX)

__all__ = [
    'app',
    'router',
    'PredictionRequest',
    'PredictionResponse',
    'APIConfig',
    'AuthMiddleware',
    'RateLimiter'
]