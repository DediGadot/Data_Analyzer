#!/usr/bin/env python3
"""
Test script to diagnose parallelism issues in the fraud detection pipeline
"""

import multiprocessing as mp
import os
import sys
import time
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import numpy as np

def cpu_intensive_task(n):
    """CPU intensive task to test parallelism"""
    result = 0
    for i in range(n * 100000):
        result += i ** 0.5
    return result

def monitor_cpu_usage():
    """Monitor CPU usage during execution"""
    process = psutil.Process()
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    memory_info = process.memory_info()
    
    print(f"CPU usage per core: {cpu_percent}")
    print(f"Average CPU usage: {sum(cpu_percent)/len(cpu_percent):.1f}%")
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"Number of threads: {process.num_threads()}")

def test_processpool_executor():
    """Test ProcessPoolExecutor"""
    print("\n=== Testing ProcessPoolExecutor ===")
    
    # Create some CPU-intensive work
    tasks = [1000] * 8  # 8 tasks, each doing 100M operations
    
    print("Starting ProcessPoolExecutor test...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_intensive_task, tasks))
    
    duration = time.time() - start_time
    print(f"ProcessPoolExecutor completed in {duration:.2f} seconds")
    print(f"Results: {len(results)} tasks completed")
    
    monitor_cpu_usage()

def test_multiprocessing_pool():
    """Test multiprocessing Pool"""
    print("\n=== Testing multiprocessing.Pool ===") 
    
    tasks = [1000] * 8
    
    print("Starting multiprocessing.Pool test...")
    start_time = time.time()
    
    with mp.Pool(4) as pool:
        results = pool.map(cpu_intensive_task, tasks)
    
    duration = time.time() - start_time
    print(f"Pool completed in {duration:.2f} seconds")
    print(f"Results: {len(results)} tasks completed")
    
    monitor_cpu_usage()

def test_sequential():
    """Test sequential processing"""
    print("\n=== Testing Sequential Processing ===")
    
    tasks = [1000] * 8
    
    print("Starting sequential test...")
    start_time = time.time()
    
    results = [cpu_intensive_task(task) for task in tasks]
    
    duration = time.time() - start_time
    print(f"Sequential completed in {duration:.2f} seconds")
    print(f"Results: {len(results)} tasks completed")
    
    monitor_cpu_usage()

def test_pandas_operations():
    """Test pandas with multiprocessing"""
    print("\n=== Testing Pandas DataFrame Operations ===")
    
    # Create test DataFrame similar to fraud detection data
    df = pd.DataFrame({
        'channelId': [f'CH{i:04d}' for i in range(10000)],
        'ip': np.random.choice(['1.1.1.1', '2.2.2.2', '3.3.3.3'], 10000),
        'isBot': np.random.choice([True, False], 10000),
        'volume': np.random.randint(1, 1000, 10000),
        'quality_score': np.random.uniform(0, 10, 10000)
    })
    
    def process_chunk(chunk):
        """Process a chunk of DataFrame"""
        # Simulate feature engineering operations
        result = chunk.copy()
        result['new_feature'] = result['volume'] * result['quality_score']
        result['bot_rate'] = result.groupby('channelId')['isBot'].transform('mean')
        return result
    
    # Split into chunks
    chunks = [df[i:i+2500] for i in range(0, len(df), 2500)]
    
    print(f"Processing {len(chunks)} chunks with ProcessPoolExecutor...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    duration = time.time() - start_time
    print(f"DataFrame processing completed in {duration:.2f} seconds")
    
    # Combine results
    final_df = pd.concat(results, ignore_index=True)
    print(f"Final DataFrame shape: {final_df.shape}")
    
    monitor_cpu_usage()

def main():
    print("=== PARALLELISM DIAGNOSTIC TEST ===")
    print(f"Python version: {sys.version}")
    print(f"Multiprocessing start method: {mp.get_start_method()}")
    print(f"CPU count: {mp.cpu_count()}")
    print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not in venv')}")
    
    # Initial CPU monitoring
    print("\n=== Initial System State ===")
    monitor_cpu_usage()
    
    # Run tests
    test_sequential()
    test_processpool_executor() 
    test_multiprocessing_pool()
    test_pandas_operations()
    
    print("\n=== TEST COMPLETED ===")
    print("If you saw high CPU usage across multiple cores during the parallel tests,")
    print("then multiprocessing is working correctly on your system.")
    print("If only one core was active, there may be a configuration issue.")

if __name__ == "__main__":
    # Set multiprocessing start method explicitly
    mp.set_start_method('fork', force=True)
    main()