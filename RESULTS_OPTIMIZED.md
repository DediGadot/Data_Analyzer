# Fraud Detection ML Pipeline Results (Optimized)

Generated: 2025-08-06 12:38:10
Mode: Approximate (Fast)
Processing Speed: 533 records/second

## Performance Summary

- **Total Processing Time**: 46.5 minutes
- **Records Processed**: 1,487,379
- **Peak Memory Usage**: 4306.30 MB
- **CPU Cores Used**: 4
- **Optimization Level**: Approximate

## Key Findings

- **Total Channels Analyzed**: 21,330
- **High-Risk Channels**: 3,263
- **Average Quality Score**: 3.44/10

## Fraud Classification Results

- **Total Rows Classified**: 1,487,379
- **Fraud Detected**: 48,363 (3.3%)
- **Average Quality Score**: 5.00/10
- **Average Risk Score**: 0.286
- **Output File**: `/home/fiod/shimshi/fraud_classification_results.csv`

### Classification Thresholds Used

- Quality Threshold (Low): 3.0
- Quality Threshold (High): 7.0
- Anomaly Threshold: 3
- Risk Threshold: 0.5

### Performance Breakdown

| Step | Time (seconds) | Memory (MB) |
|------|----------------|-------------|
| Data Loading | 70.3 | 1388.4 |
| Feature Engineering | 1646.1 | 4306.3 |
| Quality Scoring | 100.1 | 2633.8 |
| Anomaly Detection | 229.0 | 3132.4 |
| Fraud Classification | 637.6 | 1696.3 |

