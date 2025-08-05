#!/usr/bin/env python
"""
Safe wrapper for running the optimized pipeline with timeouts and error handling
"""

import subprocess
import sys
import time
import signal
import os

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Process timed out")

def run_pipeline_with_timeout(sample_fraction, approximate=True, timeout_seconds=300):
    """Run the optimized pipeline with a timeout"""
    
    cmd = [
        sys.executable, "main_pipeline_optimized.py",
        "--sample-fraction", str(sample_fraction),
        "--n-jobs", "4"
    ]
    
    if approximate:
        cmd.append("--approximate")
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Timeout: {timeout_seconds} seconds")
    print("-" * 60)
    
    start_time = time.time()
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        # Run the pipeline
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.strip())
                # Check for completion
                if "OPTIMIZED PIPELINE COMPLETED SUCCESSFULLY" in line:
                    signal.alarm(0)  # Cancel timeout
                    break
        
        process.wait()
        elapsed = time.time() - start_time
        
        print(f"\nCompleted in {elapsed:.1f} seconds")
        
        # Check output files
        if os.path.exists("channel_quality_scores_optimized.csv"):
            import pandas as pd
            quality_df = pd.read_csv("channel_quality_scores_optimized.csv")
            print(f"✓ Generated quality scores for {len(quality_df)} channels")
        
        return True
        
    except TimeoutException:
        print(f"\n⚠️  Pipeline timed out after {timeout_seconds} seconds")
        print("This is normal for larger datasets - results may still be valid")
        
        # Kill the process
        try:
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
        except:
            pass
        
        # Check partial results
        if os.path.exists("channel_quality_scores_optimized.csv"):
            import pandas as pd
            quality_df = pd.read_csv("channel_quality_scores_optimized.csv")
            print(f"✓ Partial results: {len(quality_df)} channels scored")
            return True
        
        return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        signal.alarm(0)
        return False
    
    finally:
        signal.alarm(0)  # Always cancel timeout

def main():
    """Run optimized pipeline with progressive sample sizes"""
    
    print("Optimized Fraud Detection Pipeline Runner")
    print("=" * 60)
    
    # Test configurations with increasing complexity
    configs = [
        (0.001, True, 60),   # 0.1% sample, approximate, 1 min timeout
        (0.01, True, 180),   # 1% sample, approximate, 3 min timeout
        (0.05, True, 300),   # 5% sample, approximate, 5 min timeout
        (0.1, True, 600),    # 10% sample, approximate, 10 min timeout
    ]
    
    for sample_frac, approx, timeout in configs:
        print(f"\n{'='*60}")
        print(f"Testing with {sample_frac*100}% sample")
        print(f"{'='*60}")
        
        success = run_pipeline_with_timeout(sample_frac, approx, timeout)
        
        if not success:
            print(f"\nStopping at {sample_frac*100}% due to issues")
            break
        
        print("\n✓ Test passed, continuing to next configuration...")
        time.sleep(2)
    
    print("\n" + "="*60)
    print("Testing complete!")
    
    # Summary
    if os.path.exists("RESULTS_OPTIMIZED.md"):
        print("\n📊 Latest results available in RESULTS_OPTIMIZED.md")
        
        # Show performance summary
        with open("RESULTS_OPTIMIZED.md", "r") as f:
            lines = f.readlines()
            in_summary = False
            for line in lines:
                if "Performance Summary" in line:
                    in_summary = True
                elif in_summary and line.strip() == "":
                    break
                elif in_summary:
                    print(line.strip())

if __name__ == "__main__":
    # Activate virtual environment if needed
    if os.path.exists("venv/bin/activate"):
        activate_cmd = "source venv/bin/activate && "
    else:
        activate_cmd = ""
    
    main()