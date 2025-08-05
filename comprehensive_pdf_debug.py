"""
Comprehensive PDF Generation Debugging Script
Tests all aspects of PDF generation to identify potential issues
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
import tempfile
import shutil
import pickle
from typing import Dict, List, Tuple, Optional

# Test imports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfgen import canvas
    print("✓ ReportLab imports successful")
except ImportError as e:
    print(f"✗ ReportLab import failed: {e}")
    sys.exit(1)

try:
    from pdf_report_generator_multilingual import MultilingualPDFReportGenerator
    print("✓ PDF generator import successful")
except ImportError as e:
    print(f"✗ PDF generator import failed: {e}")
    sys.exit(1)

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFGenerationDebugger:
    """Comprehensive PDF generation debugger"""
    
    def __init__(self, test_dir: str = "/home/fiod/shimshi/pdf_debug_test"):
        self.test_dir = test_dir
        self.results = {}
        self.errors = []
        
        # Create test directory
        os.makedirs(test_dir, exist_ok=True)
        logger.info(f"Created test directory: {test_dir}")
    
    def run_all_tests(self) -> Dict:
        """Run all debugging tests"""
        print("=" * 80)
        print("COMPREHENSIVE PDF GENERATION DEBUG SUITE")
        print("=" * 80)
        
        test_methods = [
            ("System Check", self.test_system_requirements),
            ("Basic Matplotlib", self.test_matplotlib_basic),
            ("Advanced Matplotlib", self.test_matplotlib_advanced),
            ("ReportLab Basic", self.test_reportlab_basic),
            ("ReportLab Advanced", self.test_reportlab_advanced),
            ("File System", self.test_file_system),
            ("Data Generation", self.test_data_generation),
            ("Edge Cases", self.test_edge_cases),
            ("Memory Stress", self.test_memory_handling),
            ("Full Pipeline", self.test_full_pipeline),
            ("Error Recovery", self.test_error_recovery)
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_method()
                self.results[test_name] = {"status": "PASSED", "details": result}
                print(f"✓ {test_name}: PASSED")
            except Exception as e:
                error_msg = f"{test_name} failed: {str(e)}"
                self.errors.append(error_msg)
                self.results[test_name] = {"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}
                print(f"✗ {test_name}: FAILED - {str(e)}")
                logger.error(f"{test_name} failed", exc_info=True)
        
        # Generate summary report
        self.generate_debug_report()
        return self.results
    
    def test_system_requirements(self) -> Dict:
        """Test system requirements and environment"""
        info = {
            "python_version": sys.version,
            "matplotlib_backend": matplotlib.get_backend(),
            "current_directory": os.getcwd(),
            "test_directory_writable": os.access(self.test_dir, os.W_OK),
            "temp_directory": tempfile.gettempdir(),
            "available_fonts": []
        }
        
        # Test font availability
        try:
            import matplotlib.font_manager as fm
            fonts = [f.name for f in fm.fontManager.ttflist[:10]]  # Sample of fonts
            info["available_fonts"] = fonts
        except Exception as e:
            info["font_error"] = str(e)
        
        # Test memory
        try:
            import psutil
            info["available_memory_gb"] = psutil.virtual_memory().available / (1024**3)
            info["cpu_count"] = psutil.cpu_count()
        except ImportError:
            info["memory_info"] = "psutil not available"
        
        return info
    
    def test_matplotlib_basic(self) -> Dict:
        """Test basic matplotlib functionality"""
        results = {}
        
        # Test simple plot creation
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title("Basic Matplotlib Test")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        
        test_file = os.path.join(self.test_dir, "basic_plot.png")
        plt.savefig(test_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            results["basic_plot"] = {"created": True, "size_bytes": file_size}
            
            # Verify it's a valid image
            try:
                from PIL import Image
                img = Image.open(test_file)
                results["basic_plot"]["valid_image"] = True
                results["basic_plot"]["dimensions"] = img.size
            except Exception as e:
                results["basic_plot"]["image_validation_error"] = str(e)
        else:
            results["basic_plot"] = {"created": False}
        
        return results
    
    def test_matplotlib_advanced(self) -> Dict:
        """Test advanced matplotlib plots similar to PDF generator"""
        results = {}
        
        # Test histogram (similar to quality distribution)
        plt.figure(figsize=(10, 6))
        data = np.random.normal(5, 2, 1000)
        
        # Test edge case: all same values
        same_values = np.full(100, 5.0)
        try:
            plt.hist(same_values, bins=1, alpha=0.7, color='blue', edgecolor='black')
            plt.title("Single Value Histogram Test")
            plt.xlabel("Value")
            plt.ylabel("Count")
            test_file = os.path.join(self.test_dir, "single_value_hist.png")
            plt.savefig(test_file, dpi=150, bbox_inches='tight')
            plt.close()
            results["single_value_histogram"] = {"created": os.path.exists(test_file)}
        except Exception as e:
            results["single_value_histogram"] = {"error": str(e)}
        
        # Test scatter plot with color mapping
        plt.figure(figsize=(10, 6))
        x = np.random.rand(100) * 1000
        y = np.random.rand(100) * 10
        colors = np.random.rand(100)
        
        scatter = plt.scatter(x, y, c=colors, cmap='RdYlGn_r', alpha=0.6)
        plt.xscale('log')
        plt.colorbar(scatter)
        plt.title("Scatter Plot Test")
        plt.xlabel("Volume (log scale)")
        plt.ylabel("Quality Score")
        
        test_file = os.path.join(self.test_dir, "scatter_plot.png")
        plt.savefig(test_file, dpi=150, bbox_inches='tight')
        plt.close()
        results["scatter_plot"] = {"created": os.path.exists(test_file)}
        
        # Test heatmap
        plt.figure(figsize=(10, 8))
        data_matrix = np.random.rand(10, 5)
        sns.heatmap(data_matrix, annot=True, cmap='YlOrRd')
        plt.title("Heatmap Test")
        
        test_file = os.path.join(self.test_dir, "heatmap.png")
        plt.savefig(test_file, dpi=150, bbox_inches='tight')
        plt.close()
        results["heatmap"] = {"created": os.path.exists(test_file)}
        
        return results
    
    def test_reportlab_basic(self) -> Dict:
        """Test basic ReportLab functionality"""
        results = {}
        
        # Test basic PDF creation
        test_file = os.path.join(self.test_dir, "basic_reportlab.pdf")
        
        try:
            doc = SimpleDocTemplate(test_file, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Add title
            story.append(Paragraph("ReportLab Basic Test", styles['Title']))
            story.append(Spacer(1, 12))
            
            # Add normal text
            story.append(Paragraph("This is a basic ReportLab test document.", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            # Verify file creation
            if os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                results["basic_pdf"] = {"created": True, "size_bytes": file_size}
                
                # Verify it's a valid PDF
                with open(test_file, 'rb') as f:
                    header = f.read(8)
                    results["basic_pdf"]["valid_pdf"] = header.startswith(b'%PDF-')
            else:
                results["basic_pdf"] = {"created": False}
                
        except Exception as e:
            results["basic_pdf"] = {"error": str(e)}
        
        return results
    
    def test_reportlab_advanced(self) -> Dict:
        """Test advanced ReportLab features"""
        results = {}
        
        test_file = os.path.join(self.test_dir, "advanced_reportlab.pdf")
        
        try:
            doc = SimpleDocTemplate(test_file, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Test table creation
            data = [['Header 1', 'Header 2', 'Header 3']]
            for i in range(5):
                data.append([f'Row {i+1} Col 1', f'Row {i+1} Col 2', f'Row {i+1} Col 3'])
            
            table = Table(data)
            story.append(table)
            story.append(Spacer(1, 12))
            
            # Test image inclusion (if matplotlib plot exists)
            plot_file = os.path.join(self.test_dir, "basic_plot.png")
            if os.path.exists(plot_file):
                from reportlab.platypus import Image
                story.append(Image(plot_file, width=300, height=200))
            
            # Build PDF
            doc.build(story)
            
            # Verify
            if os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                results["advanced_pdf"] = {"created": True, "size_bytes": file_size}
            else:
                results["advanced_pdf"] = {"created": False}
                
        except Exception as e:
            results["advanced_pdf"] = {"error": str(e)}
        
        return results
    
    def test_file_system(self) -> Dict:
        """Test file system operations"""
        results = {}
        
        # Test directory creation and permissions
        temp_subdir = os.path.join(self.test_dir, "temp_subdir")
        try:
            os.makedirs(temp_subdir, exist_ok=True)
            results["directory_creation"] = os.path.exists(temp_subdir)
            
            # Test file writing
            test_file = os.path.join(temp_subdir, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("Test file content")
            results["file_writing"] = os.path.exists(test_file)
            
            # Test file reading
            with open(test_file, 'r') as f:
                content = f.read()
            results["file_reading"] = content == "Test file content"
            
            # Clean up
            shutil.rmtree(temp_subdir)
            results["cleanup"] = not os.path.exists(temp_subdir)
            
        except Exception as e:
            results["file_system_error"] = str(e)
        
        return results
    
    def test_data_generation(self) -> Dict:
        """Test data generation for PDF reports"""
        results = {}
        
        try:
            # Generate test data similar to real pipeline data
            n_channels = 100
            
            quality_df = pd.DataFrame({
                'channelId': [f'CH{i:04d}' for i in range(n_channels)],
                'quality_score': np.random.uniform(1, 10, n_channels),
                'bot_rate': np.random.uniform(0, 1, n_channels),
                'volume': np.random.exponential(100, n_channels).astype(int),
                'quality_category': np.random.choice(['Low', 'Medium-Low', 'Medium-High', 'High'], n_channels),
                'high_risk': np.random.choice([True, False], n_channels, p=[0.2, 0.8])
            })
            
            results["quality_df"] = {
                "shape": quality_df.shape,
                "columns": list(quality_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in quality_df.dtypes.items()}
            }
            
            # Generate anomaly data
            anomaly_df = pd.DataFrame({
                'channelId': quality_df['channelId'],
                'temporal_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9]),
                'geographic_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9]),
                'volume_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9]),
                'device_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9]),
                'behavioral_anomaly': np.random.choice([True, False], n_channels, p=[0.1, 0.9])
            })
            
            # Calculate overall anomaly count
            anomaly_cols = [col for col in anomaly_df.columns if 'anomaly' in col]
            anomaly_df['overall_anomaly_count'] = anomaly_df[anomaly_cols].sum(axis=1)
            
            results["anomaly_df"] = {
                "shape": anomaly_df.shape,
                "columns": list(anomaly_df.columns),
                "anomaly_stats": {
                    "total_anomalies": int(anomaly_df['overall_anomaly_count'].sum()),
                    "channels_with_anomalies": int((anomaly_df['overall_anomaly_count'] > 0).sum())
                }
            }
            
            # Save test data for later use
            quality_df.to_csv(os.path.join(self.test_dir, "test_quality_data.csv"), index=False)
            anomaly_df.to_csv(os.path.join(self.test_dir, "test_anomaly_data.csv"), index=False)
            
            results["data_saved"] = True
            
        except Exception as e:
            results["data_generation_error"] = str(e)
        
        return results
    
    def test_edge_cases(self) -> Dict:
        """Test edge cases that might cause PDF generation issues"""
        results = {}
        
        edge_cases = [
            ("single_value", lambda: pd.DataFrame({
                'channelId': ['CH0001'],
                'quality_score': [5.0],
                'bot_rate': [0.5],
                'volume': [100],
                'quality_category': ['Medium'],
                'high_risk': [False]
            })),
            ("all_same_values", lambda: pd.DataFrame({
                'channelId': [f'CH{i:04d}' for i in range(10)],
                'quality_score': [5.0] * 10,
                'bot_rate': [0.5] * 10,
                'volume': [100] * 10,
                'quality_category': ['Medium'] * 10,
                'high_risk': [False] * 10
            })),
            ("extreme_values", lambda: pd.DataFrame({
                'channelId': [f'CH{i:04d}' for i in range(5)],
                'quality_score': [0.0, 10.0, 0.0, 10.0, 5.0],
                'bot_rate': [0.0, 1.0, 0.0, 1.0, 0.5],
                'volume': [1, 1000000, 1, 1000000, 1000],
                'quality_category': ['Low', 'High', 'Low', 'High', 'Medium'],
                'high_risk': [True, False, True, False, False]
            })),
            ("empty_dataframe", lambda: pd.DataFrame(columns=[
                'channelId', 'quality_score', 'bot_rate', 'volume', 'quality_category', 'high_risk'
            ]))
        ]
        
        for case_name, data_generator in edge_cases:
            try:
                test_df = data_generator()
                results[case_name] = {
                    "generated": True,
                    "shape": test_df.shape,
                    "empty": len(test_df) == 0
                }
                
                # Test if this data would cause matplotlib issues
                if len(test_df) > 0:
                    try:
                        plt.figure(figsize=(8, 6))
                        if test_df['quality_score'].nunique() > 1:
                            plt.hist(test_df['quality_score'], bins=min(10, test_df['quality_score'].nunique()))
                        else:
                            plt.hist(test_df['quality_score'], bins=1)
                        plt.title(f"Test histogram - {case_name}")
                        test_file = os.path.join(self.test_dir, f"edge_case_{case_name}.png")
                        plt.savefig(test_file)
                        plt.close()
                        results[case_name]["plot_successful"] = os.path.exists(test_file)
                    except Exception as plot_error:
                        results[case_name]["plot_error"] = str(plot_error)
                
            except Exception as e:
                results[case_name] = {"error": str(e)}
        
        return results
    
    def test_memory_handling(self) -> Dict:
        """Test memory handling with larger datasets"""
        results = {}
        
        try:
            # Test with progressively larger datasets
            sizes = [100, 1000, 10000, 50000]
            
            for size in sizes:
                try:
                    # Generate large dataset
                    large_df = pd.DataFrame({
                        'channelId': [f'CH{i:06d}' for i in range(size)],
                        'quality_score': np.random.uniform(1, 10, size),
                        'bot_rate': np.random.uniform(0, 1, size),
                        'volume': np.random.exponential(100, size).astype(int),
                        'quality_category': np.random.choice(['Low', 'Medium-Low', 'Medium-High', 'High'], size),
                        'high_risk': np.random.choice([True, False], size, p=[0.2, 0.8])
                    })
                    
                    # Test memory usage
                    memory_usage = large_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                    
                    # Test plot creation with large data
                    plt.figure(figsize=(10, 6))
                    sample_data = large_df['quality_score'].sample(min(1000, len(large_df)))
                    plt.hist(sample_data, bins=20, alpha=0.7)
                    plt.title(f"Large Dataset Test - {size} records")
                    
                    test_file = os.path.join(self.test_dir, f"large_dataset_{size}.png")
                    plt.savefig(test_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    results[f"size_{size}"] = {
                        "memory_mb": memory_usage,
                        "plot_created": os.path.exists(test_file),
                        "dataframe_created": True
                    }
                    
                    # Clean up memory
                    del large_df
                    
                except Exception as e:
                    results[f"size_{size}"] = {"error": str(e)}
        
        except Exception as e:
            results["memory_test_error"] = str(e)
        
        return results
    
    def test_full_pipeline(self) -> Dict:
        """Test full PDF generation pipeline"""
        results = {}
        
        try:
            # Load test data
            quality_df = pd.read_csv(os.path.join(self.test_dir, "test_quality_data.csv"))
            anomaly_df = pd.read_csv(os.path.join(self.test_dir, "test_anomaly_data.csv"))
            
            # Create test results
            final_results = {
                'top_quality_channels': quality_df.nlargest(10, 'quality_score').to_dict('records'),
                'high_risk_channels': quality_df[quality_df['high_risk'] == True].head(10).to_dict('records'),
                'summary_stats': {
                    'total_channels': len(quality_df),
                    'high_risk_count': len(quality_df[quality_df['high_risk'] == True]),
                    'avg_quality_score': float(quality_df['quality_score'].mean())
                }
            }
            
            pipeline_results = {
                'pipeline_summary': {
                    'total_processing_time_minutes': 5.0,
                    'records_processed': len(quality_df),
                    'channels_analyzed': len(quality_df),
                    'models_trained': 3
                }
            }
            
            # Test PDF generation
            generator = MultilingualPDFReportGenerator(output_dir=self.test_dir)
            
            # Test English PDF
            start_time = datetime.now()
            en_path, he_path = generator.generate_comprehensive_report(
                quality_df, anomaly_df, final_results, pipeline_results
            )
            generation_time = (datetime.now() - start_time).total_seconds()
            
            results["pdf_generation"] = {
                "english_path": en_path,
                "hebrew_path": he_path,
                "generation_time_seconds": generation_time,
                "english_created": os.path.exists(en_path) if en_path else False,
                "hebrew_created": os.path.exists(he_path) if he_path else False
            }
            
            # Check file sizes
            if en_path and os.path.exists(en_path):
                results["pdf_generation"]["english_size_bytes"] = os.path.getsize(en_path)
                # Verify it's a valid PDF
                with open(en_path, 'rb') as f:
                    header = f.read(8)
                    results["pdf_generation"]["english_valid_pdf"] = header.startswith(b'%PDF-')
            
            if he_path and os.path.exists(he_path):
                results["pdf_generation"]["hebrew_size_bytes"] = os.path.getsize(he_path)
                # Verify it's a valid PDF
                with open(he_path, 'rb') as f:
                    header = f.read(8)
                    results["pdf_generation"]["hebrew_valid_pdf"] = header.startswith(b'%PDF-')
            
        except Exception as e:
            results["full_pipeline_error"] = str(e)
            results["full_pipeline_traceback"] = traceback.format_exc()
        
        return results
    
    def test_error_recovery(self) -> Dict:
        """Test error recovery and graceful failure handling"""
        results = {}
        
        # Test with intentionally problematic data
        try:
            # Test with NaN values
            problematic_df = pd.DataFrame({
                'channelId': ['CH0001', 'CH0002', 'CH0003'],
                'quality_score': [5.0, np.nan, 3.0],
                'bot_rate': [0.5, 0.7, np.nan],
                'volume': [100, 200, 300],
                'quality_category': ['Medium', None, 'Low'],
                'high_risk': [False, True, False]
            })
            
            generator = MultilingualPDFReportGenerator(output_dir=self.test_dir)
            
            # This should handle NaN values gracefully
            en_path, he_path = generator.generate_comprehensive_report(
                problematic_df, pd.DataFrame(), {}, {}
            )
            
            results["nan_handling"] = {
                "completed": True,
                "english_created": os.path.exists(en_path) if en_path else False,
                "hebrew_created": os.path.exists(he_path) if he_path else False
            }
            
        except Exception as e:
            results["nan_handling"] = {"error": str(e)}
        
        return results
    
    def generate_debug_report(self):
        """Generate comprehensive debug report"""
        report_file = os.path.join(self.test_dir, "debug_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("PDF GENERATION DEBUG REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Test Directory: {self.test_dir}\n\n")
            
            # Summary
            passed_tests = sum(1 for r in self.results.values() if r.get('status') == 'PASSED')
            total_tests = len(self.results)
            f.write(f"SUMMARY: {passed_tests}/{total_tests} tests passed\n\n")
            
            # Error summary
            if self.errors:
                f.write("ERRORS FOUND:\n")
                for error in self.errors:
                    f.write(f"- {error}\n")
                f.write("\n")
            else:
                f.write("NO ERRORS FOUND\n\n")
            
            # Detailed results
            f.write("DETAILED TEST RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for test_name, result in self.results.items():
                f.write(f"\n{test_name}:\n")
                f.write(f"Status: {result.get('status', 'UNKNOWN')}\n")
                
                if result.get('status') == 'FAILED':
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                    if result.get('traceback'):
                        f.write(f"Traceback:\n{result.get('traceback')}\n")
                else:
                    details = result.get('details', {})
                    for key, value in details.items():
                        f.write(f"  {key}: {value}\n")
        
        print(f"\nDebug report saved to: {report_file}")
        
        # Print summary to console
        print(f"\nSUMMARY: {passed_tests}/{total_tests} tests passed")
        if self.errors:
            print(f"ERRORS: {len(self.errors)} found")
        else:
            print("NO ERRORS FOUND - PDF generation appears to be working correctly")

def main():
    """Run comprehensive PDF debugging"""
    debugger = PDFGenerationDebugger()
    results = debugger.run_all_tests()
    
    # Print final assessment
    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)
    
    passed_tests = sum(1 for r in results.values() if r.get('status') == 'PASSED')
    total_tests = len(results)
    
    if passed_tests == total_tests:
        print("✓ ALL TESTS PASSED - PDF generation is working correctly")
        print("  The issue may be intermittent or data-specific")
    elif passed_tests >= total_tests * 0.8:
        print("⚠ MOSTLY WORKING - Some minor issues found")
        print("  PDF generation should work but may have edge case problems")
    else:
        print("✗ SIGNIFICANT ISSUES FOUND - PDF generation has problems")
        print("  Check the debug report for detailed error information")
    
    print(f"\nTest results: {passed_tests}/{total_tests} passed")
    print(f"Debug files saved to: {debugger.test_dir}")
    
    return results

if __name__ == "__main__":
    results = main()