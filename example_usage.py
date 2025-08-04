#!/usr/bin/env python3
"""
Traffic Analyzer - Usage Examples
==================================

Comprehensive examples demonstrating how to use the traffic analyzer
for processing and analyzing large traffic datasets.
"""

from traffic_analyzer import (
    TrafficDataLoader, 
    AdvancedFeatureExtractor,
    ChannelSimilarityAnalyzer,
    TrafficPatternAnalyzer,
    QualityScoreCalculator,
    TemporalAnalyzer,
    TrafficAnalyzerPipeline,
    PerformanceBenchmark,
    DataLoadConfig,
    FeatureConfig
)

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def example_basic_data_loading():
    """Example 1: Basic data loading with different configurations."""
    print("=" * 60)
    print("EXAMPLE 1: BASIC DATA LOADING")
    print("=" * 60)
    
    file_path = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    
    # Load with pandas (default configuration)
    print("\n1.1 Loading with pandas (optimized):")
    loader_pandas = TrafficDataLoader(DataLoadConfig(use_polars=False))
    df_pandas = loader_pandas.load_data(file_path, nrows=10000)
    print(f"Loaded {len(df_pandas):,} records with {len(df_pandas.columns)} columns")
    print(f"Memory usage: {df_pandas.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Load with polars (faster for large datasets)
    print("\n1.2 Loading with polars (faster):")
    loader_polars = TrafficDataLoader(DataLoadConfig(use_polars=True))
    df_polars = loader_polars.load_data(file_path, nrows=10000)
    print(f"Loaded {len(df_polars):,} records with {len(df_polars.columns)} columns")
    
    return df_pandas


def example_feature_extraction(df):
    """Example 2: Advanced feature extraction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: ADVANCED FEATURE EXTRACTION")
    print("=" * 60)
    
    # Configure feature extraction
    config = FeatureConfig(
        extract_platform=True,
        extract_location=True,
        parse_user_agent=True,
        temporal_features=True,
        ip_features=True
    )
    
    extractor = AdvancedFeatureExtractor(config)
    
    print(f"\nOriginal features: {len(df.columns)}")
    df_enriched = extractor.extract_all_features(df)
    print(f"Features after extraction: {len(df_enriched.columns)}")
    print(f"New features added: {len(df_enriched.columns) - len(df.columns)}")
    
    # Show some of the new features
    new_features = [col for col in df_enriched.columns if col not in df.columns]
    print(f"\nSample new features: {new_features[:10]}")
    
    return df_enriched


def example_channel_similarity_analysis(df_enriched):
    """Example 3: Channel similarity analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: CHANNEL SIMILARITY ANALYSIS")
    print("=" * 60)
    
    similarity_analyzer = ChannelSimilarityAnalyzer()
    
    # Calculate similarity matrix
    similarity_matrix = similarity_analyzer.calculate_channel_similarity(df_enriched)
    print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
    print(f"Channels analyzed: {len(similarity_matrix.index)}")
    
    # Find similar channels for a specific channel
    if len(similarity_matrix.index) > 1:
        sample_channel = similarity_matrix.index[0]
        similar_channels = similarity_analyzer.find_similar_channels(similarity_matrix, sample_channel, top_k=5)
        
        print(f"\nTop 5 channels similar to {sample_channel}:")
        for channel, similarity in similar_channels.items():
            print(f"  {channel}: {similarity:.3f}")
    
    return similarity_matrix


def example_traffic_pattern_analysis(df_enriched):
    """Example 4: Traffic pattern analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: TRAFFIC PATTERN ANALYSIS")
    print("=" * 60)
    
    pattern_analyzer = TrafficPatternAnalyzer()
    
    # Device patterns
    print("\n4.1 Device Pattern Analysis:")
    device_patterns = pattern_analyzer.analyze_device_patterns(df_enriched)
    if 'device_distribution' in device_patterns:
        print("Device distribution:")
        print(device_patterns['device_distribution'])
    
    # Geographic patterns
    print("\n4.2 Geographic Pattern Analysis:")
    geo_patterns = pattern_analyzer.analyze_geographic_patterns(df_enriched)
    if 'country_distribution' in geo_patterns:
        print("Top 10 countries:")
        print(geo_patterns['country_distribution'].head(10))
    
    # Browser patterns
    print("\n4.3 Browser Pattern Analysis:")
    browser_patterns = pattern_analyzer.analyze_browser_patterns(df_enriched)
    if 'browser_distribution' in browser_patterns:
        print("Browser distribution:")
        print(browser_patterns['browser_distribution'])
    
    return device_patterns, geo_patterns, browser_patterns


def example_quality_scoring(df_enriched):
    """Example 5: Quality scoring and anomaly detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: QUALITY SCORING & ANOMALY DETECTION")
    print("=" * 60)
    
    quality_calculator = QualityScoreCalculator()
    
    # Calculate quality scores
    quality_scores = quality_calculator.calculate_channel_quality_scores(df_enriched)
    print(f"\nQuality scores calculated for {len(quality_scores)} channels")
    print(f"Average quality score: {quality_scores['quality_score'].mean():.2f}")
    print(f"Quality score range: {quality_scores['quality_score'].min():.2f} - {quality_scores['quality_score'].max():.2f}")
    
    # Show top and bottom channels by quality
    print("\nTop 5 highest quality channels:")
    top_channels = quality_scores.nlargest(5, 'quality_score')[['channelId', 'quality_score', 'traffic_volume']]
    print(top_channels.to_string(index=False))
    
    print("\nBottom 5 lowest quality channels:")
    bottom_channels = quality_scores.nsmallest(5, 'quality_score')[['channelId', 'quality_score', 'traffic_volume']]
    print(bottom_channels.to_string(index=False))
    
    # Detect anomalous channels
    anomalous = quality_calculator.detect_anomalous_channels(quality_scores, threshold=50.0)
    print(f"\nAnomalous channels (quality < 50): {len(anomalous)}")
    
    # Cluster channels by quality
    clustered = quality_calculator.cluster_channels_by_quality(quality_scores)
    cluster_summary = clustered['cluster'].value_counts().sort_index()
    print(f"\nQuality-based clustering:")
    print(cluster_summary)
    
    return quality_scores


def example_temporal_analysis(df_enriched):
    """Example 6: Temporal pattern analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: TEMPORAL PATTERN ANALYSIS")
    print("=" * 60)
    
    temporal_analyzer = TemporalAnalyzer()
    
    # Traffic patterns by channel
    traffic_patterns = temporal_analyzer.analyze_traffic_patterns(df_enriched, group_by='channelId')
    
    print(f"\nTemporal patterns analyzed:")
    for pattern_name, pattern_data in traffic_patterns.items():
        print(f"  {pattern_name}: {pattern_data.shape}")
    
    # Detect temporal anomalies
    if 'hour' in df_enriched.columns:
        anomalies = temporal_analyzer.detect_temporal_anomalies(df_enriched)
        anomalous_hours = anomalies[anomalies['is_anomaly']]
        print(f"\nAnomalous hours detected: {len(anomalous_hours)}")
        if len(anomalous_hours) > 0:
            print("Hours with anomalous traffic:")
            print(anomalous_hours[['hour', 'traffic_count', 'z_score']])
    
    return traffic_patterns


def example_full_pipeline():
    """Example 7: Complete analysis pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: COMPLETE ANALYSIS PIPELINE")
    print("=" * 60)
    
    file_path = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    
    # Initialize pipeline
    pipeline = TrafficAnalyzerPipeline(file_path)
    
    # Run full analysis
    pipeline.run_full_analysis(nrows=20000)
    
    # Get summary report
    print(pipeline.get_summary_report())
    
    # Access specific analysis results
    if 'quality_scores' in pipeline.analysis_results:
        quality_df = pipeline.analysis_results['quality_scores']
        print(f"\nDetailed quality analysis available for {len(quality_df)} channels")
    
    return pipeline


def example_performance_benchmarking():
    """Example 8: Performance benchmarking."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: PERFORMANCE BENCHMARKING")
    print("=" * 60)
    
    file_path = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    
    benchmark = PerformanceBenchmark(file_path)
    
    # Run benchmarks
    print("\nRunning data loading benchmark...")
    benchmark.benchmark_data_loading(nrows=20000)
    
    print("\nRunning feature extraction benchmark...")
    benchmark.benchmark_feature_extraction(nrows=10000)
    
    print("\nRunning analysis operations benchmark...")
    benchmark.benchmark_analysis_operations(nrows=10000)
    
    # Generate report
    print(benchmark.generate_performance_report())
    
    return benchmark


def example_chunked_processing():
    """Example 9: Chunked processing for large datasets."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: CHUNKED PROCESSING")
    print("=" * 60)
    
    file_path = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    
    # Configure for chunked processing
    config = DataLoadConfig(chunk_size=25000, use_polars=False)
    loader = TrafficDataLoader(config)
    extractor = AdvancedFeatureExtractor()
    quality_calculator = QualityScoreCalculator()
    
    all_quality_scores = []
    chunk_count = 0
    total_records = 0
    
    print("Processing dataset in chunks...")
    
    for chunk in loader.load_data_chunked(file_path):
        chunk_count += 1
        total_records += len(chunk)
        
        print(f"Processing chunk {chunk_count}: {len(chunk):,} records")
        
        # Extract features for chunk
        enriched_chunk = extractor.extract_all_features(chunk)
        
        # Calculate quality scores for chunk
        chunk_quality = quality_calculator.calculate_channel_quality_scores(enriched_chunk)
        all_quality_scores.append(chunk_quality)
        
        # Process only first 3 chunks for example
        if chunk_count >= 3:
            break
    
    # Combine results from all chunks
    if all_quality_scores:
        combined_quality = pd.concat(all_quality_scores, ignore_index=True)
        print(f"\nCombined results:")
        print(f"  Total chunks processed: {chunk_count}")
        print(f"  Total records processed: {total_records:,}")
        print(f"  Total channels analyzed: {len(combined_quality)}")
        print(f"  Average quality score: {combined_quality['quality_score'].mean():.2f}")
    
    return combined_quality


def main():
    """Main function demonstrating all examples."""
    print("TRAFFIC ANALYZER - COMPREHENSIVE USAGE EXAMPLES")
    print("=" * 80)
    
    try:
        # Example 1: Basic data loading
        df = example_basic_data_loading()
        
        # Example 2: Feature extraction
        df_enriched = example_feature_extraction(df)
        
        # Example 3: Channel similarity
        similarity_matrix = example_channel_similarity_analysis(df_enriched)
        
        # Example 4: Pattern analysis
        device_patterns, geo_patterns, browser_patterns = example_traffic_pattern_analysis(df_enriched)
        
        # Example 5: Quality scoring
        quality_scores = example_quality_scoring(df_enriched)
        
        # Example 6: Temporal analysis
        traffic_patterns = example_temporal_analysis(df_enriched)
        
        # Example 7: Full pipeline
        pipeline = example_full_pipeline()
        
        # Example 8: Performance benchmarking
        benchmark = example_performance_benchmarking()
        
        # Example 9: Chunked processing
        combined_quality = example_chunked_processing()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()