# Binning Fixes Summary

## Problem
The PDF generation was failing with "bins must increase monotonically" errors when dealing with small datasets or datasets with limited unique values. This error occurred in several methods:

1. `create_quality_distribution_plot()` - using `plt.hist()`
2. `create_risk_matrix()` - using `pd.cut()`
3. `create_cluster_quality_chart()` - using `pd.qcut()`
4. `create_quality_trend_plot()` - using `pd.qcut()`
5. `create_bot_rate_boxplot()` - using `pd.qcut()` indirectly

## Root Causes
- **Identical values**: When all values in a column are identical, `pd.qcut()` and `pd.cut()` fail
- **Insufficient unique values**: When there are fewer unique values than requested bins/quantiles
- **Small datasets**: t-SNE and clustering algorithms require minimum data points
- **Edge cases**: Single-row datasets, empty datasets, or datasets with very small ranges

## Fixes Implemented

### 1. Quality Distribution Plot (`create_quality_distribution_plot`)
**Before**: Fixed 20 bins regardless of data
```python
plt.hist(df['quality_score'], bins=20, ...)
```

**After**: Dynamic bin calculation with fallbacks
```python
unique_values = len(df['quality_score'].unique())

if unique_values <= 1:
    bins = 1
elif unique_values <= 5:
    bins = min(unique_values, 5)
else:
    bins = min(20, unique_values // 2)

# Fallback to bar chart if histogram fails
try:
    plt.hist(df['quality_score'], bins=bins, ...)
except Exception as e:
    value_counts = df['quality_score'].value_counts().head(10)
    plt.bar(range(len(value_counts)), value_counts.values, ...)
```

### 2. Risk Matrix (`create_risk_matrix`)
**Before**: Fixed bin ranges that might not fit data
```python
bot_rate_bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
volume_bins = [0, 10, 100, 1000, 10000, df['volume'].max()]
df['bot_rate_bin'] = pd.cut(df['bot_rate'], bins=bot_rate_bins, ...)
```

**After**: Dynamic bin adjustment with edge case handling
```python
# Handle identical values
if bot_rate_min == bot_rate_max:
    bot_rate_bins = [bot_rate_min - 0.01, bot_rate_min + 0.01]
    bot_rate_labels = [f'{bot_rate_min:.1%}']
else:
    # Normal case with data-adjusted bins
    bot_rate_bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    if bot_rate_max > 1.0:
        bot_rate_bins[-1] = bot_rate_max + 0.01

# Ensure bins are monotonically increasing
bot_rate_bins = sorted(set(bot_rate_bins))

# Fallback to median-based 2x2 matrix if binning fails
except Exception as e:
    bot_rate_median = df['bot_rate'].median()
    volume_median = df['volume'].median()
    df_temp['bot_rate_bin'] = df['bot_rate'].apply(lambda x: 'High' if x >= bot_rate_median else 'Low')
```

### 3. Cluster Quality Chart (`create_cluster_quality_chart`)
**Before**: Fixed 5 quantiles
```python
quality_df['cluster'] = pd.qcut(quality_df['quality_score'], q=5, ...)
```

**After**: Adaptive quantile selection with fallbacks
```python
unique_values = len(quality_df['quality_score'].unique())
if unique_values <= 1:
    quality_df['cluster'] = 'Cluster 0'
elif unique_values < 5:
    q = min(unique_values, 3)
    quality_df['cluster'] = pd.qcut(..., q=q, duplicates='drop')
else:
    quality_df['cluster'] = pd.qcut(..., q=5, duplicates='drop')

# Ultimate fallback: equal-width bins
except Exception as e:
    range_size = (max_score - min_score) / 3
    quality_df['cluster'] = quality_df['quality_score'].apply(lambda x: 
        'Cluster 0' if x < min_score + range_size
        else 'Cluster 1' if x < min_score + 2 * range_size
        else 'Cluster 2'
    )
```

### 4. Quality Trend Plot (`create_quality_trend_plot`)
**Before**: Fixed 10 periods
```python
quality_df['period'] = pd.qcut(quality_df.index, q=10, ...)
```

**After**: Dynamic period calculation
```python
n_periods = min(10, len(quality_df) // 2)
if n_periods <= 1:
    quality_df['period'] = 'P1'
else:
    quality_df['period'] = pd.qcut(..., q=n_periods, duplicates='drop')

# Fallback: index-based periods
except Exception as e:
    period_size = len(quality_df) // n_periods
    quality_df['period'] = quality_df.index // max(1, period_size)
```

### 5. Bot Rate Boxplot (`create_bot_rate_boxplot`)
**Before**: Assumed multiple quality categories exist
```python
df_copy['quality_category'] = pd.Categorical(df_copy['quality_category'], ...)
box_plot = df_copy.boxplot(column='bot_rate', by='quality_category', ...)
```

**After**: Category validation with fallbacks
```python
# Check if quality_category exists and has multiple values
if 'quality_category' not in df_copy.columns:
    df_copy['quality_category'] = pd.qcut(df_copy['quality_score'], q=4, duplicates='drop')

category_counts = df_copy['quality_category'].value_counts()
if len(category_counts) <= 1:
    # Fallback to simple bar chart
    mean_bot_rate = df_copy['bot_rate'].mean()
    plt.bar([0], [mean_bot_rate], ...)
else:
    box_plot = df_copy.boxplot(...)

# Color customization safety
if hasattr(box_plot, 'artists') and len(box_plot.artists) > 0:
    for patch, color in zip(box_plot.artists, colors_list):
        patch.set_facecolor(color)
```

### 6. Clustering Visualization (`create_cluster_visualization`)
**Before**: Fixed clustering parameters
```python
n_clusters = min(5, len(quality_df) // 10)
kmeans = KMeans(n_clusters=n_clusters, ...)
tsne = TSNE(n_components=2, perplexity=min(30, len(quality_df) - 1))
```

**After**: Adaptive clustering with safety checks
```python
n_samples = len(quality_df)
if n_samples < 2:
    clusters = np.zeros(n_samples)
    n_clusters = 1
else:
    n_clusters = max(1, min(5, n_samples // 10))
    if n_clusters == 1:
        clusters = np.zeros(n_samples)
    else:
        kmeans = KMeans(n_clusters=n_clusters, ...)

# t-SNE safety
if n_samples < 3:
    tsne_results = np.random.rand(n_samples, 2)
else:
    perplexity = min(30, max(1, n_samples - 1))
    tsne = TSNE(..., perplexity=perplexity)

# Fallback for t-SNE failure
except Exception as e:
    tsne_results = features_scaled[:sample_size, :2] if features_scaled.shape[1] >= 2 else np.random.rand(sample_size, 2)
```

### 7. Anomaly Detection (`create_anomaly_heatmap`)
**Before**: Assumed anomalies exist
```python
top_anomalous = anomaly_df.nlargest(min(20, len(anomaly_df)), 'overall_anomaly_count')
```

**After**: Anomaly existence validation
```python
non_zero_anomalies = anomaly_df[anomaly_df['overall_anomaly_count'] > 0]
if len(non_zero_anomalies) > 0:
    top_anomalous = non_zero_anomalies.nlargest(min(20, len(non_zero_anomalies)), 'overall_anomaly_count')
else:
    top_anomalous = anomaly_df.head(min(20, len(anomaly_df)))
    logger.warning("No channels with anomalies > 0 found")
```

## Error Handling Strategy

All fixes follow a consistent pattern:

1. **Data Validation**: Check for edge cases (empty data, single values, insufficient samples)
2. **Adaptive Parameters**: Adjust binning/clustering parameters based on actual data characteristics
3. **Graceful Fallbacks**: Provide alternative visualizations when primary method fails
4. **Logging**: Log warnings and errors for debugging while continuing execution
5. **Preserve Functionality**: Ensure the visualization still provides meaningful insights even with limited data

## Benefits

- **Robust**: Handles datasets of any size, including edge cases
- **Informative**: Still provides useful visualizations even with limited data
- **Maintainable**: Clear error handling and logging for debugging
- **User-friendly**: Continues execution instead of crashing on edge cases

## Testing

The fixes handle these problematic scenarios:
- Single-row datasets
- Datasets with all identical values
- Datasets with very few unique values (2-3 unique values)
- Empty or near-empty datasets
- Datasets with extreme value ranges
- Missing or malformed categorical data

Each fix includes appropriate fallback strategies to ensure the PDF generation completes successfully while still providing meaningful visualizations.