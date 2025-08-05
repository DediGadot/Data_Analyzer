"""
Simple Example: Running the Optimized Fraud Detection Pipeline
Demonstrates basic usage with different optimization settings.
"""

import logging
from main_pipeline_optimized import OptimizedFraudDetectionPipeline, create_optimization_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_basic_example():
    """Run a basic example with the optimized pipeline."""
    
    # Configuration
    data_path = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
    output_dir = "/home/fiod/shimshi/"
    
    # Example 1: Fast approximate processing (recommended for large datasets)
    logger.info("="*60)
    logger.info("EXAMPLE 1: Fast Approximate Processing")
    logger.info("="*60)
    
    # Create configuration for speed-optimized processing
    fast_config = create_optimization_config(
        approximate=True,        # Use approximate algorithms
        n_jobs=4,               # Use 4 CPU cores
        sample_fraction=0.1,    # Process 10% of data
        chunk_size=25000        # Process in 25k record chunks
    )
    
    # Initialize and run pipeline
    pipeline_fast = OptimizedFraudDetectionPipeline(data_path, output_dir, fast_config)
    
    try:
        results_fast = pipeline_fast.run_complete_pipeline()
        
        summary = results_fast.get('pipeline_summary', {})
        logger.info("Fast processing completed successfully!")
        logger.info(f"Time: {summary.get('total_processing_time_minutes', 0):.2f} minutes")
        logger.info(f"Records: {summary.get('records_processed', 0):,}")
        logger.info(f"Speed: {summary.get('records_per_second', 0):.0f} records/sec")
        
    except Exception as e:
        logger.error(f"Fast processing failed: {e}")
    
    # Example 2: Balanced processing (good speed + accuracy)
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Balanced Processing")
    logger.info("="*60)
    
    # Create configuration for balanced processing
    balanced_config = create_optimization_config(
        approximate=False,       # Use full precision algorithms
        n_jobs=4,               # Use 4 CPU cores  
        sample_fraction=0.05,   # Process 5% of data
        chunk_size=20000        # Smaller chunks for memory efficiency
    )
    
    # Initialize and run pipeline
    pipeline_balanced = OptimizedFraudDetectionPipeline(data_path, output_dir, balanced_config)
    
    try:
        results_balanced = pipeline_balanced.run_complete_pipeline()
        
        summary = results_balanced.get('pipeline_summary', {})
        logger.info("Balanced processing completed successfully!")
        logger.info(f"Time: {summary.get('total_processing_time_minutes', 0):.2f} minutes")
        logger.info(f"Records: {summary.get('records_processed', 0):,}")
        logger.info(f"Speed: {summary.get('records_per_second', 0):.0f} records/sec")
        
    except Exception as e:
        logger.error(f"Balanced processing failed: {e}")

def run_command_line_example():
    """Example of running from command line with different options."""
    
    print("\n" + "="*60)
    print("COMMAND LINE USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            "name": "Fast Approximate Processing",
            "command": "python main_pipeline_optimized.py --approximate --sample-fraction 0.1 --n-jobs 4",
            "description": "Use approximate algorithms, 10% sample, 4 cores"
        },
        {
            "name": "Full Precision with Sampling", 
            "command": "python main_pipeline_optimized.py --sample-fraction 0.05 --n-jobs -1",
            "description": "Full precision algorithms, 5% sample, all cores"
        },
        {
            "name": "Memory Optimized",
            "command": "python main_pipeline_optimized.py --approximate --chunk-size 25000 --sample-fraction 0.15",
            "description": "Approximate mode with larger chunks for memory efficiency"
        },
        {
            "name": "Production Ready",
            "command": "python main_pipeline_optimized.py --approximate --sample-fraction 1.0 --n-jobs 8",
            "description": "Full dataset processing with approximate algorithms"
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  Command: {example['command']}")
        print(f"  Description: {example['description']}")
    
    print("\n" + "="*60)
    print("PERFORMANCE TARGETS")
    print("="*60)
    print("✓ Target: Process 1.48M records in <60 minutes")
    print("✓ Memory: Use <7.8GB RAM") 
    print("✓ Hardware: Optimized for 4+ CPU cores")
    print("✓ Throughput: >400 records/second")

def main():
    """Main function to run examples."""
    print("Optimized Fraud Detection Pipeline - Usage Examples")
    print("="*60)
    
    # Run basic examples
    run_basic_example()
    
    # Show command line examples
    run_command_line_example()
    
    print("\n" + "="*60)
    print("FILES GENERATED")
    print("="*60)
    print("📊 RESULTS_OPTIMIZED.md - Performance-focused results")
    print("📈 final_results_optimized.json - Machine-readable results") 
    print("📋 channel_quality_scores_optimized.csv - Channel scores")
    print("🔍 channel_anomaly_scores_optimized.csv - Anomaly detection")
    print("📝 fraud_detection_pipeline_optimized.log - Detailed logs")
    
    print("\n🚀 Ready for production deployment!")

if __name__ == "__main__":
    main()