# Fraud Detection ML Pipeline Results (Optimized)

Generated: 2025-08-06 09:07:42
Mode: Approximate (Fast)
Processing Speed: 542 records/second

## Performance Summary

- **Total Processing Time**: 45.7 minutes
- **Records Processed**: 1,487,379
- **Peak Memory Usage**: 3062.30 MB
- **CPU Cores Used**: 4
- **Optimization Level**: Approximate

## Key Findings

- **Total Channels Analyzed**: 21,330
- **High-Risk Channels**: 3,241
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
| Data Loading | 70.7 | 1397.1 |
| Feature Engineering | 1636.3 | 2659.5 |
| Quality Scoring | 92.9 | 2620.7 |
| Anomaly Detection | 220.6 | 3062.3 |
| Fraud Classification | 637.0 | 1869.9 |

