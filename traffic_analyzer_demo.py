#!/usr/bin/env python3
"""
Traffic Analyzer - Demo Version (No External Dependencies)
==========================================================

This is a demonstration version of the traffic analyzer that works without
external dependencies like pandas, numpy, etc. It shows the core concepts
and structure of the full system.

For the full version, install requirements from requirements.txt:
    pip install -r requirements.txt

Then use traffic_analyzer.py for production workloads.
"""

import json
import csv
import time
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def timing_decorator(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


class SimpleTrafficDataLoader:
    """Simplified data loader for demonstration without pandas/polars dependencies."""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
    
    @timing_decorator
    def load_sample_data(self, file_path: str, max_rows: int = 1000) -> List[Dict[str, Any]]:
        """Load a sample of the data for analysis."""
        logger.info(f"Loading sample data (max {max_rows} rows) from {file_path}")
        
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    
                    # Clean and process the row
                    processed_row = self._process_row(row)
                    if processed_row:
                        data.append(processed_row)
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []
        
        logger.info(f"Loaded {len(data)} records successfully")
        return data
    
    def _process_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Process a single row and convert data types."""
        try:
            processed = {}
            
            # Keep string fields as-is
            string_fields = ['publisherId', 'channelId', 'advertiserId', 'feedId', 
                           'keyword', 'country', 'browser', 'device', 'referrer', 
                           'ip', 'userId', 'userAgent', 'platform', 'location']
            
            for field in string_fields:
                processed[field] = row.get(field, '')
            
            # Convert boolean fields
            bool_fields = ['isLikelyBot', 'isIpDatacenter', 'isIpAnonymous', 
                          'isIpCrawler', 'isIpPublicProxy', 'isIpVPN', 
                          'isIpHostingService', 'isIpTOR', 'isIpResidentialProxy']
            
            for field in bool_fields:
                value = row.get(field, '').lower()
                processed[field] = value == 'true'
            
            # Convert numeric fields
            processed['browserMajorVersion'] = int(row.get('browserMajorVersion', 0) or 0)
            
            # Parse date
            date_str = row.get('date', '')
            if date_str:
                try:
                    processed['date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f UTC')
                except:
                    processed['date'] = datetime.now()
            else:
                processed['date'] = datetime.now()
            
            return processed
            
        except Exception as e:
            logger.warning(f"Error processing row: {e}")
            return None
    
    def load_chunked(self, file_path: str) -> Generator[List[Dict[str, Any]], None, None]:
        """Generator for chunked data processing."""
        logger.info(f"Loading data in chunks of {self.chunk_size}")
        
        chunk = []
        chunk_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    processed_row = self._process_row(row)
                    if processed_row:
                        chunk.append(processed_row)
                    
                    if len(chunk) >= self.chunk_size:
                        chunk_count += 1
                        logger.info(f"Yielding chunk {chunk_count} with {len(chunk)} records")
                        yield chunk
                        chunk = []
                
                # Yield remaining data if any
                if chunk:
                    chunk_count += 1
                    logger.info(f"Yielding final chunk {chunk_count} with {len(chunk)} records")
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error in chunked loading: {e}")


class SimpleFeatureExtractor:
    """Simplified feature extraction for demonstration."""
    
    @timing_decorator
    def extract_features(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract additional features from the data."""
        logger.info("Extracting features...")
        
        enriched_data = []
        
        for record in data:
            enriched_record = record.copy()
            
            # Extract temporal features
            date_obj = record.get('date')
            if date_obj:
                enriched_record['hour'] = date_obj.hour
                enriched_record['day_of_week'] = date_obj.weekday()
                enriched_record['is_weekend'] = date_obj.weekday() >= 5
            
            # Extract platform features from JSON
            platform_json = record.get('platform', '')
            if platform_json:
                try:
                    platform_data = json.loads(platform_json)
                    enriched_record['os_name'] = platform_data.get('os', {}).get('name', '')
                    enriched_record['cpu_architecture'] = platform_data.get('cpu', {}).get('architecture', '')
                    enriched_record['engine_name'] = platform_data.get('engine', {}).get('name', '')
                except:
                    enriched_record['os_name'] = ''
                    enriched_record['cpu_architecture'] = ''
                    enriched_record['engine_name'] = ''
            
            # Extract location features from JSON
            location_json = record.get('location', '')
            if location_json:
                try:
                    location_data = json.loads(location_json)
                    enriched_record['city_name'] = location_data.get('cityName', '')
                    enriched_record['timezone'] = location_data.get('timezone', '')
                    enriched_record['timezone_offset'] = location_data.get('timezoneOffset', 0)
                except:
                    enriched_record['city_name'] = ''
                    enriched_record['timezone'] = ''
                    enriched_record['timezone_offset'] = 0
            
            # Calculate risk score
            risk_flags = ['isIpDatacenter', 'isIpAnonymous', 'isIpCrawler', 
                         'isIpPublicProxy', 'isIpVPN', 'isIpTOR', 'isIpResidentialProxy']
            risk_score = sum(1 for flag in risk_flags if enriched_record.get(flag, False))
            enriched_record['ip_risk_score'] = risk_score
            enriched_record['is_high_risk_ip'] = risk_score >= 2
            
            enriched_data.append(enriched_record)
        
        logger.info(f"Feature extraction completed for {len(enriched_data)} records")
        return enriched_data


class SimplePatternAnalyzer:
    """Simplified pattern analysis for demonstration."""
    
    @timing_decorator
    def analyze_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze traffic patterns."""
        logger.info("Analyzing traffic patterns...")
        
        results = {}
        
        # Device analysis
        device_counts = Counter(record.get('device', 'unknown') for record in data)
        results['device_distribution'] = dict(device_counts)
        
        # Country analysis
        country_counts = Counter(record.get('country', 'unknown') for record in data)
        results['country_distribution'] = dict(country_counts.most_common(10))
        
        # Browser analysis
        browser_counts = Counter(record.get('browser', 'unknown') for record in data)
        results['browser_distribution'] = dict(browser_counts)
        
        # Temporal analysis
        if any('hour' in record for record in data):
            hour_counts = Counter(record.get('hour', 0) for record in data if 'hour' in record)
            results['hourly_distribution'] = dict(hour_counts)
        
        # Risk analysis
        high_risk_count = sum(1 for record in data if record.get('is_high_risk_ip', False))
        bot_count = sum(1 for record in data if record.get('isLikelyBot', False))
        
        results['risk_analysis'] = {
            'total_records': len(data),
            'high_risk_ips': high_risk_count,
            'bot_traffic': bot_count,
            'high_risk_ratio': high_risk_count / len(data) if data else 0,
            'bot_ratio': bot_count / len(data) if data else 0
        }
        
        logger.info("Pattern analysis completed")
        return results


class SimpleQualityScorer:
    """Simplified quality scoring for demonstration."""
    
    @timing_decorator
    def calculate_channel_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate quality scores for each channel."""
        logger.info("Calculating channel quality scores...")
        
        # Group data by channel
        channel_data = defaultdict(list)
        for record in data:
            channel_id = record.get('channelId', 'unknown')
            channel_data[channel_id].append(record)
        
        quality_scores = {}
        
        for channel_id, records in channel_data.items():
            total_records = len(records)
            
            # Calculate risk metrics
            bot_count = sum(1 for r in records if r.get('isLikelyBot', False))
            high_risk_count = sum(1 for r in records if r.get('is_high_risk_ip', False))
            datacenter_count = sum(1 for r in records if r.get('isIpDatacenter', False))
            
            # Calculate ratios
            bot_ratio = bot_count / total_records if total_records > 0 else 0
            high_risk_ratio = high_risk_count / total_records if total_records > 0 else 0
            datacenter_ratio = datacenter_count / total_records if total_records > 0 else 0
            
            # Calculate quality score (0-100, higher is better)
            quality_score = 100 - (
                bot_ratio * 30 +
                high_risk_ratio * 25 +
                datacenter_ratio * 20
            )
            
            quality_scores[channel_id] = {
                'quality_score': max(0, quality_score),
                'traffic_volume': total_records,
                'bot_ratio': bot_ratio,
                'high_risk_ratio': high_risk_ratio,
                'datacenter_ratio': datacenter_ratio
            }
        
        logger.info(f"Quality scores calculated for {len(quality_scores)} channels")
        return quality_scores


class SimpleBenchmark:
    """Simple performance benchmarking."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.results = {}
    
    @timing_decorator
    def benchmark_loading(self, max_rows: int = 10000):
        """Benchmark data loading."""
        loader = SimpleTrafficDataLoader()
        
        start_time = time.perf_counter()
        data = loader.load_sample_data(self.file_path, max_rows)
        load_time = time.perf_counter() - start_time
        
        self.results['loading'] = {
            'time_seconds': load_time,
            'records_loaded': len(data),
            'records_per_second': len(data) / load_time if load_time > 0 else 0
        }
        
        return data
    
    @timing_decorator
    def benchmark_feature_extraction(self, data: List[Dict[str, Any]]):
        """Benchmark feature extraction."""
        extractor = SimpleFeatureExtractor()
        
        start_time = time.perf_counter()
        enriched_data = extractor.extract_features(data)
        extraction_time = time.perf_counter() - start_time
        
        # Count new features
        original_keys = set(data[0].keys()) if data else set()
        enriched_keys = set(enriched_data[0].keys()) if enriched_data else set()
        new_features = len(enriched_keys - original_keys)
        
        self.results['feature_extraction'] = {
            'time_seconds': extraction_time,
            'records_processed': len(enriched_data),
            'new_features_added': new_features,
            'records_per_second': len(enriched_data) / extraction_time if extraction_time > 0 else 0
        }
        
        return enriched_data
    
    @timing_decorator
    def benchmark_analysis(self, data: List[Dict[str, Any]]):
        """Benchmark pattern analysis."""
        analyzer = SimplePatternAnalyzer()
        scorer = SimpleQualityScorer()
        
        start_time = time.perf_counter()
        patterns = analyzer.analyze_patterns(data)
        quality_scores = scorer.calculate_channel_quality(data)
        analysis_time = time.perf_counter() - start_time
        
        self.results['analysis'] = {
            'time_seconds': analysis_time,
            'records_analyzed': len(data),
            'channels_scored': len(quality_scores),
            'patterns_found': len(patterns),
            'records_per_second': len(data) / analysis_time if analysis_time > 0 else 0
        }
        
        return patterns, quality_scores
    
    def generate_report(self) -> str:
        """Generate performance report."""
        if not self.results:
            return "No benchmark results available."
        
        report = ["=" * 50]
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 50)
        
        for category, results in self.results.items():
            report.append(f"\n{category.upper()}:")
            report.append("-" * 30)
            for key, value in results.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
        
        return "\n".join(report)


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("TRAFFIC ANALYZER - DEMO VERSION")
    print("=" * 60)
    print("Note: This is a simplified demo without external dependencies.")
    print("For full functionality, install requirements and use traffic_analyzer.py")
    print("=" * 60)
    
    file_path = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    
    try:
        # Initialize components
        loader = SimpleTrafficDataLoader()
        extractor = SimpleFeatureExtractor()
        analyzer = SimplePatternAnalyzer()
        scorer = SimpleQualityScorer()
        benchmark = SimpleBenchmark(file_path)
        
        print("\n1. LOADING SAMPLE DATA")
        print("-" * 30)
        data = benchmark.benchmark_loading(max_rows=5000)
        
        if not data:
            print("No data loaded. Please check the file path.")
            return
        
        print(f"Sample record keys: {list(data[0].keys())}")
        
        print("\n2. EXTRACTING FEATURES")
        print("-" * 30)
        enriched_data = benchmark.benchmark_feature_extraction(data)
        
        if enriched_data:
            new_keys = set(enriched_data[0].keys()) - set(data[0].keys())
            print(f"New features added: {new_keys}")
        
        print("\n3. ANALYZING PATTERNS")
        print("-" * 30)
        patterns, quality_scores = benchmark.benchmark_analysis(enriched_data)
        
        # Show some results
        print(f"\nDevice Distribution:")
        for device, count in patterns['device_distribution'].items():
            print(f"  {device}: {count}")
        
        print(f"\nTop Countries:")
        for country, count in list(patterns['country_distribution'].items())[:5]:
            print(f"  {country}: {count}")
        
        print(f"\nRisk Analysis:")
        risk_analysis = patterns['risk_analysis']
        print(f"  Total Records: {risk_analysis['total_records']}")
        print(f"  High Risk IPs: {risk_analysis['high_risk_ips']} ({risk_analysis['high_risk_ratio']:.2%})")
        print(f"  Bot Traffic: {risk_analysis['bot_traffic']} ({risk_analysis['bot_ratio']:.2%})")
        
        print(f"\nChannel Quality Scores (Top 5):")
        sorted_channels = sorted(quality_scores.items(), 
                               key=lambda x: x[1]['quality_score'], 
                               reverse=True)[:5]
        for channel_id, metrics in sorted_channels:
            print(f"  {channel_id}: {metrics['quality_score']:.1f} (Volume: {metrics['traffic_volume']})")
        
        print("\n4. TESTING CHUNKED PROCESSING")
        print("-" * 30)
        chunk_count = 0
        total_records = 0
        
        for chunk in loader.load_chunked(file_path):
            chunk_count += 1
            total_records += len(chunk)
            print(f"Processing chunk {chunk_count}: {len(chunk)} records")
            
            if chunk_count >= 3:  # Limit for demo
                break
        
        print(f"Total records processed in chunks: {total_records}")
        
        print("\n5. PERFORMANCE REPORT")
        print("-" * 30)
        print(benchmark.generate_report())
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nTo use the full version with advanced features:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Use traffic_analyzer.py for production workloads")
        print("3. See example_usage.py for comprehensive examples")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()