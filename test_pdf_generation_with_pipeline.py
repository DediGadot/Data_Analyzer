#!/usr/bin/env python3
"""
Test PDF generation with the actual pipeline using a small sample
"""

import sys
import os
import logging

# Add current directory to path
sys.path.append('/home/fiod/shimshi')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pdf_generation():
    """Test PDF generation with a very small sample"""
    try:
        # Import the optimized pipeline
        from main_pipeline_optimized import OptimizedFraudDetectionPipeline
        
        # Create pipeline with minimal settings
        data_path = "/home/fiod/shimshi/bq-results-20250804-141411-1754316868932.csv"
        output_dir = "/home/fiod/shimshi/"
        
        # Use a very small sample for testing
        pipeline = OptimizedFraudDetectionPipeline(
            data_path=data_path,
            output_dir=output_dir,
            n_jobs=1,  # Single core for testing
            approximate=True,  # Use approximate mode for speed
            sample_fraction=0.001  # Use only 0.1% of data for testing
        )
        
        logger.info("Starting PDF generation test with small sample...")
        
        # Run the complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Check if PDF was generated
        summary = results.get('pipeline_summary', {})
        if summary.get('completion_status') == 'SUCCESS':
            logger.info("✅ Pipeline completed successfully!")
            
            # Check for generated files
            pdf_files = []
            for file in os.listdir(output_dir):
                if file.endswith('.pdf') and 'fraud_detection_report' in file:
                    pdf_files.append(file)
            
            if pdf_files:
                logger.info(f"✅ PDF files generated: {pdf_files}")
                
                # Check file sizes to ensure they're not empty
                for pdf_file in pdf_files:
                    file_path = os.path.join(output_dir, pdf_file)
                    file_size = os.path.getsize(file_path)
                    logger.info(f"  {pdf_file}: {file_size} bytes")
                    
                    if file_size < 1000:  # Less than 1KB is probably empty
                        logger.warning(f"  Warning: {pdf_file} seems very small")
                
                return True
            else:
                logger.error("❌ No PDF files found")
                return False
        else:
            logger.error(f"❌ Pipeline failed: {summary}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("PDF GENERATION PIPELINE TEST")
    logger.info("=" * 60)
    
    success = test_pdf_generation()
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 PDF generation test PASSED!")
        logger.info("The fix successfully resolved the PDF generation issue.")
    else:
        logger.error("💥 PDF generation test FAILED!")
        logger.error("There may still be issues with the PDF generation.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)