"""
Vercel Serverless Function Entrypoint
Routes all incoming requests to the FastAPI application.
"""

import sys
import os

# Add root directory to sys.path so 'src' and 'web_app' can be imported by Vercel
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from web_app.app import app

# Vercel looks for the 'app' ASGI instance
