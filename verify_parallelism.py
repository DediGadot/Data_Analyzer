#!/usr/bin/env python3
"""
Quick verification that parallelism is working
"""
import os
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

# Set environment variables for parallelism
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMBA_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

def cpu_work(n):
    """CPU intensive work"""
    total = 0
    for i in range(n * 100000):
        total += i * 0.1
    return total

def main():
    print("=== Parallelism Verification Test ===")
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Environment variables set:")
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
        print(f"  {var}: {os.environ.get(var, 'Not set')}")
    
    # Test sequential vs parallel
    work_items = [500, 500, 500, 500]  # 4 CPU-intensive tasks
    
    print("\n--- Sequential Test ---")
    start = time.time()
    seq_results = [cpu_work(item) for item in work_items]
    seq_time = time.time() - start
    print(f"Sequential time: {seq_time:.2f} seconds")
    
    print("\n--- Parallel Test ---")
    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        par_results = list(executor.map(cpu_work, work_items))
    par_time = time.time() - start
    print(f"Parallel time: {par_time:.2f} seconds")
    
    # Calculate speedup
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    
    if speedup > 2:
        print("✅ PARALLELISM IS WORKING! Good speedup achieved.")
    elif speedup > 1.2:
        print("⚠️  PARTIAL PARALLELISM. Some speedup but not optimal.")  
    else:
        print("❌ NO PARALLELISM. Sequential and parallel times are similar.")
    
    print(f"\nRun 'htop' in another terminal while this runs to see CPU usage.")

if __name__ == "__main__":
    main()