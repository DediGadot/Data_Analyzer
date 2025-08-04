"""Test script to verify RESULTS.md generation"""

import subprocess
import os
import time

print("Running fraud detection pipeline with RESULTS.md generation...")
print("This will process a small sample to test the report generation.\n")

# Run the pipeline
start_time = time.time()

# Use a very small sample fraction for testing
cmd = ["python", "main_pipeline.py"]

# Modify main_pipeline.py temporarily to use smaller sample
with open("main_pipeline.py", "r") as f:
    content = f.read()

# Replace sample fraction temporarily
temp_content = content.replace("SAMPLE_FRACTION = 0.01", "SAMPLE_FRACTION = 0.001")

with open("main_pipeline_temp.py", "w") as f:
    f.write(temp_content)

try:
    # Run the modified pipeline
    result = subprocess.run(["python", "main_pipeline_temp.py"], 
                          capture_output=True, 
                          text=True,
                          timeout=300)  # 5 minute timeout
    
    print("Pipeline output:")
    print(result.stdout[-1000:])  # Last 1000 chars
    
    if result.stderr:
        print("\nErrors:")
        print(result.stderr[-500:])
    
    # Check if RESULTS.md was created
    if os.path.exists("RESULTS.md"):
        print("\n✅ RESULTS.md was successfully generated!")
        
        # Show first 50 lines of the file
        with open("RESULTS.md", "r") as f:
            lines = f.readlines()
            print("\nFirst 50 lines of RESULTS.md:")
            print("=" * 60)
            print("".join(lines[:50]))
            print("=" * 60)
            print(f"\nTotal lines in RESULTS.md: {len(lines)}")
    else:
        print("\n❌ RESULTS.md was not generated")
        
except subprocess.TimeoutExpired:
    print("\n⏱️ Pipeline timed out after 5 minutes")
    
except Exception as e:
    print(f"\n❌ Error running pipeline: {e}")
    
finally:
    # Clean up temp file
    if os.path.exists("main_pipeline_temp.py"):
        os.remove("main_pipeline_temp.py")
    
    elapsed = time.time() - start_time
    print(f"\nTest completed in {elapsed:.1f} seconds")