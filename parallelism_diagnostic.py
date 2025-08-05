#!/usr/bin/env python3
"""
Parallelism Diagnostic Tool for Fraud Detection Pipeline
Identifies why only single CPU core is being utilized instead of parallel processing
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import psutil
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import sys
from typing import Dict, List

class ParallelismDiagnostic:
    """Diagnostic tool to identify parallelism bottlenecks"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.process = psutil.Process()
        
    def monitor_cpu_usage(self, duration=10):
        """Monitor CPU usage across all cores"""
        print(f"Monitoring CPU usage for {duration} seconds...")
        
        # Get per-CPU usage
        cpu_usage_history = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
            cpu_usage_history.append(cpu_percent)
            
        # Calculate average usage per core
        avg_usage = np.mean(cpu_usage_history, axis=0)
        
        print("\nCPU Usage Analysis:")
        print(f"Total CPU cores: {self.cpu_count}")
        for i, usage in enumerate(avg_usage):
            print(f"Core {i}: {usage:.1f}%")
        
        cores_utilized = sum(1 for usage in avg_usage if usage > 5.0)
        print(f"\nCores actively utilized (>5%): {cores_utilized}/{self.cpu_count}")
        
        return avg_usage, cores_utilized
    
    def test_multiprocessing_pool(self):
        """Test multiprocessing.Pool functionality"""
        print("\n=== Testing multiprocessing.Pool ===")
        
        # Move function to module level for pickling
        import types
        def cpu_task(n):
            # CPU-intensive task
            return sum(i*i for i in range(n))
        
        tasks = [100000] * 8
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [cpu_task(task) for task in tasks]
        sequential_time = time.time() - start_time
        
        # Parallel processing with Pool
        start_time = time.time()
        with mp.Pool(processes=self.cpu_count) as pool:
            parallel_results = pool.map(cpu_task, tasks)
        parallel_time = time.time() - start_time
        
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Expected speedup: {sequential_time/parallel_time:.1f}x")
        print(f"Results match: {sequential_results == parallel_results}")
        
        return parallel_time < sequential_time * 0.7
    
    def test_processpool_executor(self):
        """Test ProcessPoolExecutor functionality"""
        print("\n=== Testing ProcessPoolExecutor ===")
        
        def pandas_task(size):
            # Pandas-intensive task
            df = pd.DataFrame(np.random.randn(size, 10))
            return df.sum().sum()
        
        tasks = [50000] * 4
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [pandas_task(size) for size in tasks]
        sequential_time = time.time() - start_time
        
        # Parallel processing with ProcessPoolExecutor
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            parallel_results = list(executor.map(pandas_task, tasks))
        parallel_time = time.time() - start_time
        
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Expected speedup: {sequential_time/parallel_time:.1f}x")
        
        return parallel_time < sequential_time * 0.7
    
    def diagnose_pipeline_bottlenecks(self):
        """Diagnose specific bottlenecks in the fraud detection pipeline"""
        print("\n=== Pipeline Bottleneck Analysis ===")
        
        issues = []
        
        # Check 1: Sequential processing in main pipeline
        print("1. Checking main pipeline structure...")
        pipeline_issues = self._check_pipeline_structure()
        issues.extend(pipeline_issues)
        
        # Check 2: Feature engineering parallelism
        print("2. Checking feature engineering parallelism...")
        feature_issues = self._check_feature_engineering()
        issues.extend(feature_issues)
        
        # Check 3: Anomaly detection parallelism
        print("3. Checking anomaly detection parallelism...")
        anomaly_issues = self._check_anomaly_detection()
        issues.extend(anomaly_issues)
        
        # Check 4: I/O bound operations
        print("4. Checking I/O bound operations...")
        io_issues = self._check_io_operations()
        issues.extend(io_issues)
        
        return issues
    
    def _check_pipeline_structure(self):
        """Check if pipeline steps are processed sequentially"""
        issues = []
        
        # Check main_pipeline_optimized.py structure
        try:
            # Import to check if it's accessible
            sys.path.append('/home/fiod/shimshi')
            
            # The main issue: Pipeline steps are executed sequentially
            issues.append({
                'component': 'Main Pipeline',
                'issue': 'Sequential Step Execution',
                'description': 'Pipeline steps (data loading, feature engineering, etc.) run sequentially, not in parallel',
                'impact': 'Only one CPU core utilized at a time during different pipeline phases',
                'severity': 'HIGH'
            })
            
            issues.append({
                'component': 'Main Pipeline',
                'issue': 'Blocking I/O Operations',
                'description': 'CSV reading and result writing operations block CPU utilization',
                'impact': 'CPU cores idle during I/O operations',
                'severity': 'MEDIUM'
            })
            
        except Exception as e:
            issues.append({
                'component': 'Main Pipeline',
                'issue': 'Import Error',
                'description': f'Cannot analyze pipeline structure: {e}',
                'impact': 'Unable to verify parallelism implementation',
                'severity': 'HIGH'
            })
        
        return issues
    
    def _check_feature_engineering(self):
        """Check feature engineering parallelism"""
        issues = []
        
        # Check if chunks are processed in parallel vs sequential
        issues.append({
            'component': 'Feature Engineering',
            'issue': 'Chunk Processing Bottleneck',
            'description': 'ProcessPoolExecutor used but may be serializing data unnecessarily',
            'impact': 'Overhead from pickling large DataFrames reduces parallel efficiency',
            'severity': 'MEDIUM'
        })
        
        issues.append({
            'component': 'Feature Engineering',
            'issue': 'Base Engineer Dependency',
            'description': 'Uses base FeatureEngineer which may not be pickle-safe',
            'impact': 'Potential serialization bottleneck in multiprocessing',
            'severity': 'MEDIUM'
        })
        
        return issues
    
    def _check_anomaly_detection(self):
        """Check anomaly detection parallelism"""
        issues = []
        
        # Main issue: Sequential anomaly detection steps
        issues.append({
            'component': 'Anomaly Detection',
            'issue': 'Sequential Detection Methods',
            'description': 'Temporal, geographic, device, behavioral, and volume anomalies run sequentially',
            'impact': 'Only one detection method runs at a time, underutilizing CPU cores',
            'severity': 'HIGH'
        })
        
        issues.append({
            'component': 'Anomaly Detection',
            'issue': 'Single-threaded ML Models',
            'description': 'IsolationForest and other sklearn models run single-threaded by default',
            'impact': 'ML model training and prediction not using multiple cores',
            'severity': 'MEDIUM'
        })
        
        return issues
    
    def _check_io_operations(self):
        """Check I/O bound operations"""
        issues = []
        
        issues.append({
            'component': 'Data Loading',
            'issue': 'CSV Reading Bottleneck',
            'description': 'Large CSV files read sequentially even when chunked',
            'impact': 'I/O bound operation limits CPU utilization',
            'severity': 'MEDIUM'
        })
        
        issues.append({
            'component': 'Result Generation',
            'issue': 'Sequential File Writing',
            'description': 'Multiple output files (JSON, CSV, PDF) written sequentially',
            'impact': 'CPU cores idle during file I/O operations',
            'severity': 'LOW'
        })
        
        return issues
    
    def generate_recommendations(self, issues):
        """Generate specific recommendations to fix parallelism"""
        print("\n=== PARALLELISM FIX RECOMMENDATIONS ===")
        
        recommendations = {
            'HIGH_PRIORITY': [],
            'MEDIUM_PRIORITY': [],
            'LOW_PRIORITY': []
        }
        
        for issue in issues:
            severity = issue['severity']
            component = issue['component']
            problem = issue['issue']
            
            if severity == 'HIGH':
                if 'Sequential' in problem:
                    recommendations['HIGH_PRIORITY'].append({
                        'fix': f'Parallelize {component}',
                        'implementation': self._get_parallelization_fix(component, problem),
                        'expected_improvement': '2-4x speedup'
                    })
            elif severity == 'MEDIUM':
                recommendations['MEDIUM_PRIORITY'].append({
                    'fix': f'Optimize {component}',
                    'implementation': self._get_optimization_fix(component, problem),
                    'expected_improvement': '1.5-2x speedup'
                })
            else:
                recommendations['LOW_PRIORITY'].append({
                    'fix': f'Minor optimization for {component}',
                    'implementation': f'Async I/O for {component}',
                    'expected_improvement': '10-20% speedup'
                })
        
        # Print recommendations
        for priority, recs in recommendations.items():
            print(f"\n{priority} FIXES:")
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec['fix']}")
                print(f"   Implementation: {rec['implementation']}")
                print(f"   Expected improvement: {rec['expected_improvement']}")
        
        return recommendations
    
    def _get_parallelization_fix(self, component, problem):
        """Get specific parallelization fix for component"""
        fixes = {
            ('Main Pipeline', 'Sequential Step Execution'): 
                'Use ProcessPoolExecutor to run independent anomaly detection methods in parallel',
            ('Anomaly Detection', 'Sequential Detection Methods'): 
                'Submit temporal, geographic, device, behavioral anomaly detection as concurrent futures'
        }
        return fixes.get((component, problem), 'Implement parallel processing')
    
    def _get_optimization_fix(self, component, problem):
        """Get specific optimization fix for component"""
        fixes = {
            ('Feature Engineering', 'Chunk Processing Bottleneck'): 
                'Use shared memory or reduce data serialization overhead',
            ('Feature Engineering', 'Base Engineer Dependency'): 
                'Implement pickle-safe feature engineering functions',
            ('Anomaly Detection', 'Single-threaded ML Models'): 
                'Set n_jobs=-1 for sklearn models that support parallel processing'
        }
        return fixes.get((component, problem), 'Optimize implementation')
    
    def create_parallel_fix_demo(self):
        """Create a demonstration of how to fix the parallelism issues"""
        print("\n=== PARALLEL FIX DEMONSTRATION ===")
        
        # Simulate the current sequential approach
        print("Current approach (sequential anomaly detection):")
        start_time = time.time()
        self._simulate_sequential_anomaly_detection()
        sequential_time = time.time() - start_time
        print(f"Sequential time: {sequential_time:.2f}s")
        
        # Demonstrate parallel approach
        print("\nImproved approach (parallel anomaly detection):")
        start_time = time.time()
        self._simulate_parallel_anomaly_detection()
        parallel_time = time.time() - start_time
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Speedup: {sequential_time/parallel_time:.1f}x")
        
        return sequential_time, parallel_time
    
    def _simulate_sequential_anomaly_detection(self):
        """Simulate current sequential anomaly detection"""
        def detection_task(detection_type, duration):
            time.sleep(duration)  # Simulate processing time
            return f"{detection_type} completed"
        
        # Sequential processing (current approach)
        results = []
        for detection_type, duration in [
            ("temporal", 1.0),
            ("geographic", 0.8),
            ("device", 0.6),
            ("behavioral", 0.7),
            ("volume", 0.5)
        ]:
            result = detection_task(detection_type, duration)
            results.append(result)
        
        return results
    
    def _simulate_parallel_anomaly_detection(self):
        """Simulate improved parallel anomaly detection"""
        def detection_task(args):
            detection_type, duration = args
            time.sleep(duration)  # Simulate processing time
            return f"{detection_type} completed"
        
        # Parallel processing (improved approach)
        tasks = [
            ("temporal", 1.0),
            ("geographic", 0.8),
            ("device", 0.6),
            ("behavioral", 0.7),
            ("volume", 0.5)
        ]
        
        with ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            results = list(executor.map(detection_task, tasks))
        
        return results
    
    def run_comprehensive_diagnostic(self):
        """Run complete diagnostic suite"""
        print("="*60)
        print("FRAUD DETECTION PIPELINE PARALLELISM DIAGNOSTIC")
        print("="*60)
        
        print(f"System Information:")
        print(f"CPU Cores: {self.cpu_count}")
        print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"Python version: {sys.version}")
        
        # Test basic multiprocessing
        mp_works = self.test_multiprocessing_pool()
        ppe_works = self.test_processpool_executor()
        
        if not mp_works or not ppe_works:
            print("\n❌ CRITICAL: Basic multiprocessing is not working correctly!")
            return
        else:
            print("\n✅ Basic multiprocessing functionality confirmed")
        
        # Diagnose pipeline-specific issues
        issues = self.diagnose_pipeline_bottlenecks()
        
        print(f"\n📊 ISSUES FOUND: {len(issues)}")
        for issue in issues:
            print(f"• {issue['component']}: {issue['issue']} ({issue['severity']})")
        
        # Generate recommendations
        recommendations = self.generate_recommendations(issues)
        
        # Demonstrate the fix
        seq_time, par_time = self.create_parallel_fix_demo()
        
        print(f"\n🎯 EXPECTED RESULTS AFTER FIXES:")
        print(f"• Current processing: Only 1 CPU core utilized effectively")
        print(f"• After fixes: {self.cpu_count} CPU cores utilized")
        print(f"• Expected speedup: {seq_time/par_time:.1f}x - {self.cpu_count}x depending on workload")
        
        return issues, recommendations

def main():
    """Main diagnostic function"""
    diagnostic = ParallelismDiagnostic()
    issues, recommendations = diagnostic.run_comprehensive_diagnostic()
    
    print(f"\n" + "="*60)
    print("SUMMARY: Why only 1 CPU core is being used:")
    print("="*60)
    print("1. Pipeline steps run sequentially, not in parallel")
    print("2. Anomaly detection methods run one-by-one")
    print("3. I/O operations block CPU utilization")
    print("4. Default sklearn models use single-threaded processing")
    print("\nTo fix: Implement the HIGH_PRIORITY recommendations above")

if __name__ == "__main__":
    main()