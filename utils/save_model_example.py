"""
Quick script to create a dummy RandomForest classifier and save it as
weights/engagement_classifier.pkl so the API can load it during development.
"""
import os
import numpy as np
import joblib

# Fix: Use relative import since files are in same directory
from .ml_engagement_classifier import EngagementClassifier

def create_dummy_classifier(out_path: str):
    # synthetic dataset
    rng = np.random.RandomState(42)
    X = rng.rand(500, 7)  # features: pitch,yaw,ear,br10,br30,blink_code,face_detected
    # create labels correlated to first feature for demo (1..4)
    y = (np.clip((X[:, 0] * 4).astype(int) + 1, 1, 4)).astype(int)

    clf = EngagementClassifier()
    clf.train(X, y)
    clf.save(out_path)
    print("Saved dummy classifier to:", out_path)

if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    weights_dir = os.path.join(base, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    out_file = os.path.join(weights_dir, "engagement_classifier.pkl")
    create_dummy_classifier(out_file)