"""
Test script for optimized pipeline performance
"""

import time
import subprocess
import psutil
import os

def test_pipeline(mode, sample_fraction=0.01):
    """Run pipeline test with specified configuration"""
    
    print(f"\n{'='*60}")
    print(f"Testing {mode} mode with {sample_fraction*100}% data")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python", "main_pipeline_optimized.py",
        "--sample-fraction", str(sample_fraction),
        "--n-jobs", "-1"
    ]
    
    if mode == "approximate":
        cmd.append("--approximate")
    
    # Record start metrics
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    
    # Run pipeline
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Record end metrics
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Parse output for metrics
    print(f"\nExecution time: {elapsed:.2f} seconds")
    print(f"Initial memory: {start_memory:.2f} MB")
    
    if result.returncode == 0:
        print("✓ Pipeline completed successfully")
        
        # Extract metrics from output
        for line in result.stdout.split('\n'):
            if 'Processing speed:' in line:
                print(f"  {line.strip()}")
            elif 'Peak memory:' in line:
                print(f"  {line.strip()}")
    else:
        print("✗ Pipeline failed")
        print(f"Error: {result.stderr}")
    
    return elapsed

def main():
    """Run performance comparison tests"""
    
    print("Optimized Pipeline Performance Test")
    print("=" * 60)
    
    # Check if data file exists
    data_path = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Test configurations
    tests = [
        ("full_precision", 0.01),      # 1% data, full precision
        ("approximate", 0.01),          # 1% data, approximate
        ("approximate", 0.1),           # 10% data, approximate (if successful)
    ]
    
    results = {}
    
    for mode, fraction in tests:
        try:
            elapsed = test_pipeline(mode, fraction)
            results[f"{mode}_{fraction}"] = elapsed
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Performance Summary")
    print(f"{'='*60}")
    
    for config, elapsed in results.items():
        print(f"{config}: {elapsed:.2f} seconds")
    
    # Calculate speedup
    if "full_precision_0.01" in results and "approximate_0.01" in results:
        speedup = results["full_precision_0.01"] / results["approximate_0.01"]
        print(f"\nApproximate mode speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()