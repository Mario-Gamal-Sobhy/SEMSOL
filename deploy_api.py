from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import cv2
import time
import os
import logging
from utils.ml_engagement_classifier import EngagementClassifier
import inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semsol-api")

# Model paths
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
CLASSIFIER_PATH = os.path.join(WEIGHTS_DIR, "engagement_classifier.pkl")

# Global model instance
classifier = EngagementClassifier()

app = FastAPI(title="SEMSOL Engagement API")

@app.on_event("startup")
def startup_event():
    logger.info("Starting SEMSOL API - loading classifier")
    if not os.path.exists(CLASSIFIER_PATH):
        logger.warning("Classifier not found at %s", CLASSIFIER_PATH)
    else:
        try:
            classifier.load(CLASSIFIER_PATH)
            logger.info("Classifier loaded")
        except Exception as e:
            logger.error("Failed to load classifier: %s", e)

@app.get("/health")
def health():
    return {"status": "ok", "classifier_loaded": classifier.is_trained}

@app.get("/run-inference")
def run_inference():
    """Run the full inference pipeline with webcam input"""
    try:
        # Run inference.main() with default parameters
        result = inference.main(
            source=0,  # Use webcam
            display_video=True,  # Show the video window
            use_webcam=True
        )
        return {"status": "success", "message": "Inference completed"}
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deploy_api:app", host="127.0.0.1", port=5000, reload=True)