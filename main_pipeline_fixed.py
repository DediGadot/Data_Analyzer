#!/usr/bin/env python3
"""
Fixed Main ML Pipeline for Fraud Detection
Same as main_pipeline.py but with anomaly detection fixes applied.
"""

# Apply the fix first
from anomaly_detection_fix import patch_anomaly_detection
patch_anomaly_detection()

# Now import and run the main pipeline
from main_pipeline import main

if __name__ == "__main__":
    results = main()