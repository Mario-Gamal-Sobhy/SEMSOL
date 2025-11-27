# MongoDB Configuration for SEMSOL
import streamlit as st
import os

# Default Local Configuration
DEFAULT_CONFIG = {
    "connection_string": "mongodb://localhost:27017/",
    "database_name": "semsol_engagement",
    "sessions_collection": "sessions",
    "metrics_collection": "metrics",
    "timeout_ms": 5000
}

def get_mongodb_config():
    """
    Get MongoDB configuration with priority:
    1. Streamlit Secrets (for Cloud)
    2. Environment Variables
    3. Local Default
    """
    config = DEFAULT_CONFIG.copy()
    
    # 1. Check Streamlit Secrets (Cloud)
    if hasattr(st, "secrets") and "mongo" in st.secrets:
        try:
            secrets = st.secrets["mongo"]
            if "connection_string" in secrets:
                config["connection_string"] = secrets["connection_string"]
            if "database_name" in secrets:
                config["database_name"] = secrets["database_name"]
            return config
        except Exception:
            pass
            
    # 2. Check Environment Variables
    if os.getenv("MONGO_CONNECTION_STRING"):
        config["connection_string"] = os.getenv("MONGO_CONNECTION_STRING")
        
    return config

# Export the configuration
MONGODB_CONFIG = get_mongodb_config()

# Auto-save settings for app.py
AUTO_SAVE_CONFIG = {
    "enabled": True,
    "confirm_before_save": False,
    "keep_local_data": True
}
