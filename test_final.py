"""Final test to verify all errors are fixed"""

import pandas as pd
from anomaly_detection import AnomalyDetector
import logging

logging.basicConfig(level=logging.INFO)

# Load sample data
print("Loading sample data...")
df = pd.read_csv('bq-results-20250804-141411-1754316868932.csv', nrows=500)
print(f"Loaded {len(df)} rows")

# Initialize detector
detector = AnomalyDetector()

# Test each detection method individually
print("\n1. Testing temporal anomaly detection...")
try:
    temporal_results = detector.detect_temporal_anomalies(df)
    print(f"✅ Temporal anomaly detection completed: {len(temporal_results)} results")
except Exception as e:
    print(f"❌ Temporal anomaly detection failed: {e}")

print("\n2. Testing geographic anomaly detection...")
try:
    geo_results = detector.detect_geographic_anomalies(df)
    print(f"✅ Geographic anomaly detection completed: {len(geo_results)} results")
except Exception as e:
    print(f"❌ Geographic anomaly detection failed: {e}")

print("\n3. Testing device anomaly detection...")
try:
    device_results = detector.detect_device_anomalies(df)
    print(f"✅ Device anomaly detection completed: {len(device_results)} results")
except Exception as e:
    print(f"❌ Device anomaly detection failed: {e}")

print("\n4. Testing behavioral anomaly detection...")
try:
    behavioral_results = detector.detect_behavioral_anomalies(df)
    print(f"✅ Behavioral anomaly detection completed: {len(behavioral_results)} results")
except Exception as e:
    print(f"❌ Behavioral anomaly detection failed: {e}")

print("\n5. Testing volume anomaly detection...")
try:
    volume_results = detector.detect_volume_anomalies(df)
    print(f"✅ Volume anomaly detection completed: {len(volume_results)} results")
except Exception as e:
    print(f"❌ Volume anomaly detection failed: {e}")

print("\n6. Testing comprehensive anomaly detection...")
try:
    results = detector.run_comprehensive_anomaly_detection(df)
    print(f"✅ Comprehensive anomaly detection completed: {len(results)} results")
    print(f"   Columns: {list(results.columns)[:10]}...")
except Exception as e:
    print(f"❌ Comprehensive anomaly detection failed: {e}")

print("\n✅ All tests completed!")