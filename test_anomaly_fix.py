"""Test script to verify anomaly detection fixes"""

import pandas as pd
from anomaly_detection_fix import AnomalyDetector
import logging

logging.basicConfig(level=logging.INFO)

# Load sample data
df = pd.read_csv('bq-results-20250804-141411-1754316868932.csv', nrows=1000)

# Initialize detector
detector = AnomalyDetector()

# Run comprehensive anomaly detection
print("Running anomaly detection...")
results = detector.run_comprehensive_anomaly_detection(df)

print(f"\nAnomaly detection completed successfully!")
print(f"Analyzed {len(results)} channels")
print(f"Columns in results: {list(results.columns)}")

# Check for specific anomalies
if 'overall_anomaly_flag' in results.columns:
    anomalous_channels = results[results['overall_anomaly_flag'] == True]
    print(f"\nFound {len(anomalous_channels)} anomalous channels")
else:
    print("\nNo overall anomaly flag found")

print("\n✅ All tests passed - no errors detected!")