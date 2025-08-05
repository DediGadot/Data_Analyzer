#!/usr/bin/env python3
"""
Comprehensive Hebrew Font Test Script for PDF Generation
Tests both ReportLab and matplotlib Hebrew font capabilities
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test basic imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    logger.info("✓ Matplotlib imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import matplotlib: {e}")
    sys.exit(1)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    logger.info("✓ ReportLab imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import ReportLab: {e}")
    sys.exit(1)

try:
    from bidi.algorithm import get_display
    logger.info("✓ python-bidi imported successfully")
    HAS_BIDI = True
except ImportError:
    logger.warning("✗ python-bidi not available - RTL text will not be processed")
    HAS_BIDI = False

class HebrewFontTester:
    """Comprehensive Hebrew font testing class."""
    
    def __init__(self):
        self.test_dir = Path("/home/fiod/shimshi/font_tests")
        self.test_dir.mkdir(exist_ok=True)
        
        # Hebrew test strings
        self.hebrew_texts = {
            'simple': 'שלום עולם',
            'complex': 'דוח צינור ML לזיהוי הונאות - ניתוח מקיף ותובנות',
            'mixed': 'Test שלום World עולם 123',
            'numbers': 'ציון איכות: 8.5 מתוך 10',
            'punctuation': 'שלום, עולם! איך הולך?'
        }
        
        # Font paths to test
        self.font_paths = [
            '/home/fiod/shimshi/fonts/NotoSansHebrew.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        ]
        
        self.working_fonts = []
        
    def test_font_existence(self):
        """Test if Hebrew fonts exist on the system."""
        logger.info("=== Testing Font Existence ===")
        
        for font_path in self.font_paths:
            if os.path.exists(font_path):
                size = os.path.getsize(font_path)
                logger.info(f"✓ Found font: {font_path} ({size:,} bytes)")
                self.working_fonts.append(font_path)
            else:
                logger.warning(f"✗ Missing font: {font_path}")
        
        if not self.working_fonts:
            logger.error("No Hebrew fonts found!")
            return False
        return True
    
    def test_unicode_support(self):
        """Test basic Unicode Hebrew support."""
        logger.info("=== Testing Unicode Support ===")
        
        try:
            for name, text in self.hebrew_texts.items():
                # Test encoding/decoding
                encoded = text.encode('utf-8')
                decoded = encoded.decode('utf-8')
                assert decoded == text
                logger.info(f"✓ Unicode test '{name}': {text}")
                
                # Test character categories
                import unicodedata
                hebrew_chars = [c for c in text if '\u0590' <= c <= '\u05FF']
                if hebrew_chars:
                    logger.info(f"  Hebrew characters found: {len(hebrew_chars)}")
                    
        except Exception as e:
            logger.error(f"✗ Unicode test failed: {e}")
            return False
        
        return True
    
    def test_bidi_processing(self):
        """Test BiDi (RTL) text processing."""
        logger.info("=== Testing BiDi Processing ===")
        
        if not HAS_BIDI:
            logger.warning("BiDi processing not available")
            return False
            
        try:
            for name, text in self.hebrew_texts.items():
                processed = get_display(text)
                logger.info(f"✓ BiDi '{name}': '{text}' -> '{processed}'")
                
        except Exception as e:
            logger.error(f"✗ BiDi test failed: {e}")
            return False
            
        return True
    
    def test_reportlab_fonts(self):
        """Test ReportLab Hebrew font registration and rendering."""
        logger.info("=== Testing ReportLab Fonts ===")
        
        successful_fonts = []
        
        for font_path in self.working_fonts:
            try:
                # Register font
                font_name = f"TestHebrew_{Path(font_path).stem}"
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                logger.info(f"✓ Registered ReportLab font: {font_name}")
                
                # Test creating PDF with Hebrew text
                pdf_path = self.test_dir / f"test_reportlab_{Path(font_path).stem}.pdf"
                doc = SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=letter,
                    rightMargin=72,
                    leftMargin=72,
                    topMargin=72,
                    bottomMargin=72
                )
                
                # Create custom style
                styles = getSampleStyleSheet()
                hebrew_style = ParagraphStyle(
                    name='Hebrew',
                    parent=styles['Normal'],
                    fontName=font_name,
                    fontSize=14,
                    alignment=TA_RIGHT,
                    wordWrap='RTL'
                )
                
                # Build story
                story = []
                story.append(Paragraph(f"Font Test: {font_name}", styles['Title']))
                story.append(Spacer(1, 0.5 * inch))
                
                for name, text in self.hebrew_texts.items():
                    # Process text with BiDi if available
                    if HAS_BIDI:
                        processed_text = get_display(text)
                    else:
                        processed_text = text
                    
                    story.append(Paragraph(f"Test '{name}':", styles['Heading2']))
                    story.append(Paragraph(f"Original: {text}", styles['Normal']))
                    story.append(Paragraph(f"Hebrew: {processed_text}", hebrew_style))
                    story.append(Spacer(1, 0.3 * inch))
                
                # Build PDF
                doc.build(story)
                
                if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                    logger.info(f"✓ ReportLab PDF created: {pdf_path}")
                    successful_fonts.append((font_path, font_name))
                else:
                    logger.warning(f"✗ ReportLab PDF too small or missing: {pdf_path}")
                    
            except Exception as e:
                logger.error(f"✗ ReportLab test failed for {font_path}: {e}")
                continue
        
        return successful_fonts
    
    def test_matplotlib_fonts(self):
        """Test matplotlib Hebrew font rendering."""
        logger.info("=== Testing Matplotlib Fonts ===")
        
        successful_fonts = []
        
        for font_path in self.working_fonts:
            try:
                # Add font to matplotlib
                fm.fontManager.addfont(font_path)
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                
                logger.info(f"✓ Added matplotlib font: {font_name}")
                
                # Create test plot
                fig, ax = plt.subplots(figsize=(10, 8))
                fig.suptitle(f'Hebrew Font Test: {Path(font_path).stem}', fontsize=16)
                
                y_pos = 0.9
                for name, text in self.hebrew_texts.items():
                    # Process text with BiDi if available
                    if HAS_BIDI:
                        processed_text = get_display(text)
                    else:
                        processed_text = text
                    
                    # Add text to plot
                    ax.text(0.1, y_pos, f"Test '{name}':", fontsize=12, transform=ax.transAxes)
                    ax.text(0.1, y_pos - 0.05, f"Original: {text}", fontsize=10, transform=ax.transAxes)
                    ax.text(0.1, y_pos - 0.1, f"Hebrew: {processed_text}", 
                           fontproperties=font_prop, fontsize=12, transform=ax.transAxes)
                    
                    y_pos -= 0.2
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                # Save plot
                plot_path = self.test_dir / f"test_matplotlib_{Path(font_path).stem}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                if plot_path.exists() and plot_path.stat().st_size > 10000:
                    logger.info(f"✓ Matplotlib plot created: {plot_path}")
                    successful_fonts.append((font_path, font_name))
                else:
                    logger.warning(f"✗ Matplotlib plot too small or missing: {plot_path}")
                    
            except Exception as e:
                logger.error(f"✗ Matplotlib test failed for {font_path}: {e}")
                continue
        
        return successful_fonts
    
    def test_character_coverage(self):
        """Test Hebrew character coverage in fonts."""
        logger.info("=== Testing Hebrew Character Coverage ===")
        
        # Hebrew character ranges
        hebrew_ranges = {
            'Hebrew': (0x0590, 0x05FF),
            'Hebrew Presentation Forms': (0xFB1D, 0xFB4F)
        }
        
        test_chars = [
            'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י',
            'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
            'ן', 'ם', 'ך', 'ף', 'ץ'  # Final forms
        ]
        
        for font_path in self.working_fonts:
            try:
                import matplotlib.ft2font as ft2font
                font = ft2font.FT2Font(font_path)
                
                supported_chars = []
                for char in test_chars:
                    try:
                        glyph_index = font.get_char_index(ord(char))
                        if glyph_index != 0:  # 0 means character not found
                            supported_chars.append(char)
                    except:
                        pass
                
                coverage = len(supported_chars) / len(test_chars) * 100
                logger.info(f"✓ Font {Path(font_path).stem}: {coverage:.1f}% Hebrew coverage ({len(supported_chars)}/{len(test_chars)})")
                
                if supported_chars:
                    logger.info(f"  Supported: {''.join(supported_chars[:10])}{'...' if len(supported_chars) > 10 else ''}")
                
            except Exception as e:
                logger.warning(f"Could not test character coverage for {font_path}: {e}")
        
        return True
    
    def create_comprehensive_test_pdf(self, successful_reportlab_fonts):
        """Create a comprehensive test PDF with all working fonts."""
        if not successful_reportlab_fonts:
            logger.warning("No successful ReportLab fonts to test")
            return None
            
        logger.info("=== Creating Comprehensive Test PDF ===")
        
        try:
            pdf_path = self.test_dir / "comprehensive_hebrew_test.pdf"
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            story.append(Paragraph("Comprehensive Hebrew Font Test", styles['Title']))
            story.append(Spacer(1, 0.5 * inch))
            
            for font_path, font_name in successful_reportlab_fonts:
                # Font section title
                story.append(Paragraph(f"Font: {Path(font_path).name}", styles['Heading1']))
                story.append(Spacer(1, 0.2 * inch))
                
                # Create Hebrew style for this font
                hebrew_style = ParagraphStyle(
                    name=f'Hebrew_{font_name}',
                    parent=styles['Normal'],
                    fontName=font_name,
                    fontSize=14,
                    alignment=TA_RIGHT
                )
                
                # Test each Hebrew text
                for name, text in self.hebrew_texts.items():
                    if HAS_BIDI:
                        processed_text = get_display(text)
                    else:
                        processed_text = text
                    
                    story.append(Paragraph(f"Test: {name}", styles['Heading2']))
                    story.append(Paragraph(f"Raw: {text}", styles['Normal']))
                    story.append(Paragraph(processed_text, hebrew_style))
                    story.append(Spacer(1, 0.2 * inch))
                
                story.append(Spacer(1, 0.3 * inch))
            
            # Build PDF
            doc.build(story)
            
            if pdf_path.exists():
                size = pdf_path.stat().st_size
                logger.info(f"✓ Comprehensive test PDF created: {pdf_path} ({size:,} bytes)")
                return str(pdf_path)
            else:
                logger.error("✗ Failed to create comprehensive test PDF")
                return None
                
        except Exception as e:
            logger.error(f"✗ Error creating comprehensive test PDF: {e}")
            return None
    
    def run_all_tests(self):
        """Run all Hebrew font tests."""
        logger.info("Starting comprehensive Hebrew font testing...")
        
        results = {
            'font_existence': False,
            'unicode_support': False,
            'bidi_processing': False,
            'reportlab_fonts': [],
            'matplotlib_fonts': [],
            'character_coverage': False,
            'comprehensive_pdf': None
        }
        
        # Run tests
        results['font_existence'] = self.test_font_existence()
        results['unicode_support'] = self.test_unicode_support()
        results['bidi_processing'] = self.test_bidi_processing()
        
        if results['font_existence']:
            results['reportlab_fonts'] = self.test_reportlab_fonts()
            results['matplotlib_fonts'] = self.test_matplotlib_fonts()
            results['character_coverage'] = self.test_character_coverage()
            
            if results['reportlab_fonts']:
                results['comprehensive_pdf'] = self.create_comprehensive_test_pdf(results['reportlab_fonts'])
        
        # Summary
        logger.info("=== TEST SUMMARY ===")
        logger.info(f"Font Existence: {'✓' if results['font_existence'] else '✗'}")
        logger.info(f"Unicode Support: {'✓' if results['unicode_support'] else '✗'}")
        logger.info(f"BiDi Processing: {'✓' if results['bidi_processing'] else '✗'}")
        logger.info(f"ReportLab Fonts: {len(results['reportlab_fonts'])} working")
        logger.info(f"Matplotlib Fonts: {len(results['matplotlib_fonts'])} working")
        logger.info(f"Character Coverage: {'✓' if results['character_coverage'] else '✗'}")
        logger.info(f"Comprehensive PDF: {'✓' if results['comprehensive_pdf'] else '✗'}")
        
        if results['comprehensive_pdf']:
            logger.info(f"Test results available in: {self.test_dir}")
            logger.info(f"Main test PDF: {results['comprehensive_pdf']}")
        
        return results

def main():
    """Main function to run Hebrew font tests."""
    tester = HebrewFontTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results['reportlab_fonts'] and results['matplotlib_fonts']:
        logger.info("✓ All tests passed - Hebrew fonts working correctly!")
        return 0
    else:
        logger.error("✗ Some tests failed - Hebrew font support incomplete")
        return 1

if __name__ == "__main__":
    sys.exit(main())