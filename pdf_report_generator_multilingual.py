"""
Multilingual PDF Report Generator with Enhanced Visualizations and Explanations
Supports both English and Hebrew languages with proper RTL layout
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import os
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import RTL support
try:
    from bidi.algorithm import get_display
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False
    logging.warning("python-bidi not available, RTL text may not display correctly")

logger = logging.getLogger(__name__)

class MultilingualPDFReportGenerator:
    """
    Generates comprehensive PDF reports in multiple languages with enhanced visualizations.
    """
    
    def __init__(self, output_dir: str = "/home/fiod/shimshi/"):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "report_figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Language-specific configurations
        self.languages = {
            'en': {
                'name': 'English',
                'direction': 'ltr',
                'alignment': TA_LEFT,
                'font': 'Helvetica'
            },
            'he': {
                'name': 'Hebrew',
                'direction': 'rtl',
                'alignment': TA_RIGHT,
                'font': 'NotoSansHebrew'  # Use proper Hebrew font
            }
        }
        
        # Hebrew font availability
        self.hebrew_font_available = False
        
        # Translations
        self.translations = {
            'en': {
                'title': 'Fraud Detection ML Pipeline Report',
                'subtitle': 'Comprehensive Analysis and Insights',
                'generated': 'Generated:',
                'executive_summary': 'Executive Summary',
                'tldr': 'TL;DR - Key Findings',
                'toc': 'Table of Contents',
                'quality_analysis': 'Quality Score Analysis',
                'quality_distribution': 'Channel Quality Distribution',
                'quality_by_volume': 'Quality Score vs Traffic Volume',
                'risk_analysis': 'Risk Analysis',
                'bot_rate_analysis': 'Bot Rate Analysis by Quality Category',
                'risk_matrix': 'Risk Assessment Matrix',
                'anomaly_detection': 'Anomaly Detection Results',
                'anomaly_heatmap': 'Anomaly Detection Heatmap',
                'anomaly_distribution': 'Anomaly Type Distribution',
                'traffic_similarity': 'Traffic Similarity Analysis',
                'cluster_visualization': 'Traffic Pattern Clusters',
                'cluster_quality': 'Average Quality Score by Cluster',
                'top_channels': 'Top Performing Channels',
                'bottom_channels': 'Bottom Performing Channels',
                'high_risk_channels': 'High Risk Channels',
                'model_performance': 'Model Performance Metrics',
                'recommendations': 'Recommendations',
                'immediate_actions': 'Immediate Actions Required',
                'short_term': 'Short-term Improvements',
                'long_term': 'Long-term Strategy',
                'what_shows': 'What this shows:',
                'how_interpret': 'How to interpret:',
                'key_metrics': 'Key Metrics',
                'total_channels': 'Total Channels Analyzed',
                'high_risk_count': 'High-Risk Channels',
                'anomalous_count': 'Channels with Anomalies',
                'avg_quality': 'Average Quality Score',
                'avg_bot_rate': 'Average Bot Rate',
                'total_volume': 'Total Traffic Volume',
                'critical_findings': 'Critical Findings',
                'action_items': 'Action Items',
                'channel_id': 'Channel ID',
                'quality_score': 'Quality Score',
                'bot_rate': 'Bot Rate',
                'volume': 'Volume',
                'risk_factors': 'Risk Factors',
                'anomaly_count': 'Anomaly Count',
                'cluster': 'Cluster',
                'page': 'Page'
            },
            'he': {
                'title': 'דוח צינור ML לזיהוי הונאות',
                'subtitle': 'ניתוח מקיף ותובנות',
                'generated': 'נוצר בתאריך:',
                'executive_summary': 'סיכום מנהלים',
                'tldr': 'TL;DR - ממצאים עיקריים',
                'toc': 'תוכן עניינים',
                'quality_analysis': 'ניתוח ציון איכות',
                'quality_distribution': 'התפלגות איכות ערוצים',
                'quality_by_volume': 'ציון איכות מול נפח תעבורה',
                'risk_analysis': 'ניתוח סיכונים',
                'bot_rate_analysis': 'ניתוח שיעור בוטים לפי קטגוריית איכות',
                'risk_matrix': 'מטריצת הערכת סיכונים',
                'anomaly_detection': 'תוצאות זיהוי חריגות',
                'anomaly_heatmap': 'מפת חום זיהוי חריגות',
                'anomaly_distribution': 'התפלגות סוגי חריגות',
                'traffic_similarity': 'ניתוח דמיון תעבורה',
                'cluster_visualization': 'אשכולות דפוסי תעבורה',
                'cluster_quality': 'ציון איכות ממוצע לפי אשכול',
                'top_channels': 'הערוצים המובילים',
                'bottom_channels': 'הערוצים החלשים ביותר',
                'high_risk_channels': 'ערוצים בסיכון גבוה',
                'model_performance': 'מדדי ביצועי מודל',
                'recommendations': 'המלצות',
                'immediate_actions': 'פעולות מיידיות נדרשות',
                'short_term': 'שיפורים לטווח קצר',
                'long_term': 'אסטרטגיה לטווח ארוך',
                'what_shows': 'מה זה מראה:',
                'how_interpret': 'איך לפרש:',
                'key_metrics': 'מדדים מרכזיים',
                'total_channels': 'סה״כ ערוצים שנותחו',
                'high_risk_count': 'ערוצים בסיכון גבוה',
                'anomalous_count': 'ערוצים עם חריגות',
                'avg_quality': 'ציון איכות ממוצע',
                'avg_bot_rate': 'שיעור בוטים ממוצע',
                'total_volume': 'נפח תעבורה כולל',
                'critical_findings': 'ממצאים קריטיים',
                'action_items': 'פריטי פעולה',
                'channel_id': 'מזהה ערוץ',
                'quality_score': 'ציון איכות',
                'bot_rate': 'שיעור בוטים',
                'volume': 'נפח',
                'risk_factors': 'גורמי סיכון',
                'anomaly_count': 'מספר חריגות',
                'cluster': 'אשכול',
                'page': 'עמוד'
            }
        }
        
        # Plot descriptions
        self.plot_descriptions = {
            'quality_distribution': {
                'en': {
                    'what': 'This histogram displays how channels are distributed across different quality score ranges. The x-axis shows quality scores (0-10), and the y-axis shows the number of channels in each range.',
                    'how': 'Look for concentration patterns: A right-skewed distribution (more channels with high scores) indicates overall good traffic quality. Left-skewed suggests widespread quality issues. Bimodal distributions may indicate distinct channel groups.'
                },
                'he': {
                    'what': 'היסטוגרמה זו מציגה כיצד הערוצים מתפלגים על פני טווחי ציוני איכות שונים. ציר ה-X מציג ציוני איכות (0-10), וציר ה-Y מציג את מספר הערוצים בכל טווח.',
                    'how': 'חפשו דפוסי ריכוז: התפלגות נוטה ימינה (יותר ערוצים עם ציונים גבוהים) מעידה על איכות תעבורה טובה כללית. נטייה שמאלה מרמזת על בעיות איכות נרחבות. התפלגות דו-שיאית עשויה להצביע על קבוצות ערוצים נפרדות.'
                }
            },
            'quality_by_volume': {
                'en': {
                    'what': 'This scatter plot correlates channel quality scores with traffic volume. Each point represents a channel, with position indicating its quality (y-axis) and traffic volume (x-axis, log scale).',
                    'how': 'Ideal channels appear in the top-right (high quality, high volume). Bottom-right channels (low quality, high volume) pose the highest risk. Top-left channels (high quality, low volume) may be underutilized opportunities.'
                },
                'he': {
                    'what': 'תרשים פיזור זה מתאם בין ציוני איכות ערוצים לנפח תעבורה. כל נקודה מייצגת ערוץ, כאשר המיקום מציין את איכותו (ציר Y) ונפח התעבורה (ציר X, סולם לוגריתמי).',
                    'how': 'ערוצים אידיאליים מופיעים בפינה הימנית העליונה (איכות גבוהה, נפח גבוה). ערוצים בפינה הימנית התחתונה (איכות נמוכה, נפח גבוה) מהווים את הסיכון הגבוה ביותר. ערוצים בפינה השמאלית העליונה (איכות גבוהה, נפח נמוך) עשויים להיות הזדמנויות לא מנוצלות.'
                }
            },
            'bot_rate_analysis': {
                'en': {
                    'what': 'This box plot shows the distribution of bot rates within each quality category. The boxes show quartiles, whiskers show the range, and dots indicate outliers.',
                    'how': 'Lower bot rates in higher quality categories validate the scoring model. Wide boxes indicate high variability within a category. Many outliers suggest inconsistent patterns that need investigation.'
                },
                'he': {
                    'what': 'תרשים קופסה זה מציג את התפלגות שיעורי הבוטים בתוך כל קטגוריית איכות. הקופסאות מציגות רבעונים, השפמים מציגים את הטווח, והנקודות מציינות חריגים.',
                    'how': 'שיעורי בוטים נמוכים יותר בקטגוריות איכות גבוהות מאמתים את מודל הניקוד. קופסאות רחבות מצביעות על שונות גבוהה בתוך קטגוריה. חריגים רבים מרמזים על דפוסים לא עקביים הדורשים חקירה.'
                }
            },
            'risk_matrix': {
                'en': {
                    'what': 'This 2D heatmap maps channels by bot rate (y-axis) and volume (x-axis), with color intensity showing the number of channels in each zone.',
                    'how': 'Red zones (high bot rate + high volume) require immediate action. The darker the color, the more channels in that risk zone. Focus on reducing channels in the upper-right quadrant.'
                },
                'he': {
                    'what': 'מפת חום דו-ממדית זו ממפה ערוצים לפי שיעור בוטים (ציר Y) ונפח (ציר X), כאשר עוצמת הצבע מציגה את מספר הערוצים בכל אזור.',
                    'how': 'אזורים אדומים (שיעור בוטים גבוה + נפח גבוה) דורשים פעולה מיידית. ככל שהצבע כהה יותר, כך יש יותר ערוצים באזור הסיכון. התמקדו בהפחתת ערוצים ברבע הימני העליון.'
                }
            },
            'anomaly_heatmap': {
                'en': {
                    'what': 'This heatmap visualizes which channels (rows) have which types of anomalies (columns). Dark cells indicate the presence of an anomaly.',
                    'how': 'Channels with multiple dark cells across the row have multiple anomaly types and need priority investigation. Columns with many dark cells indicate common anomaly patterns affecting many channels.'
                },
                'he': {
                    'what': 'מפת חום זו מציגה אילו ערוצים (שורות) מכילים אילו סוגי חריגות (עמודות). תאים כהים מציינים נוכחות של חריגה.',
                    'how': 'ערוצים עם מספר תאים כהים לאורך השורה מכילים מספר סוגי חריגות וזקוקים לחקירה בעדיפות. עמודות עם תאים כהים רבים מציינות דפוסי חריגה נפוצים המשפיעים על ערוצים רבים.'
                }
            },
            'anomaly_distribution': {
                'en': {
                    'what': 'This bar chart shows how many channels are affected by each type of anomaly, helping prioritize which anomaly patterns to address first.',
                    'how': 'Focus on anomaly types with the highest counts first, as they affect the most channels. Types with very low counts might indicate rare but potentially serious issues.'
                },
                'he': {
                    'what': 'תרשים עמודות זה מציג כמה ערוצים מושפעים מכל סוג של חריגה, ומסייע לתעדף אילו דפוסי חריגה לטפל בהם קודם.',
                    'how': 'התמקדו תחילה בסוגי חריגות עם הספירות הגבוהות ביותר, מכיוון שהם משפיעים על הכי הרבה ערוצים. סוגים עם ספירות נמוכות מאוד עשויים להצביע על בעיות נדירות אך פוטנציאלית חמורות.'
                }
            },
            'cluster_visualization': {
                'en': {
                    'what': 'This t-SNE plot shows how channels group into natural clusters based on their traffic patterns. Each point is a channel, colored by its cluster assignment.',
                    'how': 'Well-separated clusters indicate distinct traffic patterns. Channels far from any cluster center are outliers. Large clusters may represent common traffic patterns, while small clusters might be niche or suspicious.'
                },
                'he': {
                    'what': 'תרשים t-SNE זה מציג כיצד ערוצים מתקבצים לאשכולות טבעיים על בסיס דפוסי התעבורה שלהם. כל נקודה היא ערוץ, צבועה לפי השיוך לאשכול.',
                    'how': 'אשכולות מופרדים היטב מצביעים על דפוסי תעבורה ברורים. ערוצים רחוקים ממרכז כל אשכול הם חריגים. אשכולות גדולים עשויים לייצג דפוסי תעבורה נפוצים, בעוד אשכולות קטנים עשויים להיות נישתיים או חשודים.'
                }
            },
            'cluster_quality': {
                'en': {
                    'what': 'This bar chart displays the average quality score for each traffic cluster, helping identify which traffic patterns correlate with quality.',
                    'how': 'Clusters with low average quality scores likely contain fraudulent or low-quality traffic patterns. High-scoring clusters represent desirable traffic patterns to encourage.'
                },
                'he': {
                    'what': 'תרשים עמודות זה מציג את ציון האיכות הממוצע עבור כל אשכול תעבורה, ומסייע לזהות אילו דפוסי תעבורה מתואמים עם איכות.',
                    'how': 'אשכולות עם ציוני איכות ממוצעים נמוכים כנראה מכילים דפוסי תעבורה הונאתיים או באיכות נמוכה. אשכולות עם ציונים גבוהים מייצגים דפוסי תעבורה רצויים שיש לעודד.'
                }
            },
            'quality_score_time': {
                'en': {
                    'what': 'This line chart tracks quality score trends over time periods, revealing patterns and changes in traffic quality.',
                    'how': 'Upward trends indicate improving quality. Sudden drops may signal new fraud campaigns. Seasonal patterns help predict future quality fluctuations.'
                },
                'he': {
                    'what': 'תרשים קו זה עוקב אחר מגמות ציוני איכות לאורך תקופות זמן, וחושף דפוסים ושינויים באיכות התעבורה.',
                    'how': 'מגמות עולות מצביעות על שיפור באיכות. ירידות פתאומיות עשויות לאותת על קמפיינים חדשים של הונאה. דפוסים עונתיים מסייעים לחזות תנודות איכות עתידיות.'
                }
            },
            'fraud_distribution': {
                'en': {
                    'what': 'This pie chart breaks down the proportion of different fraud types detected across all channels.',
                    'how': 'Large slices indicate prevalent fraud types requiring systematic solutions. Multiple small slices suggest diverse fraud tactics. Use this to prioritize anti-fraud measures.'
                },
                'he': {
                    'what': 'תרשים עוגה זה מפרק את הפרופורציה של סוגי ההונאה השונים שזוהו בכל הערוצים.',
                    'how': 'פרוסות גדולות מצביעות על סוגי הונאה נפוצים הדורשים פתרונות שיטתיים. פרוסות קטנות מרובות מרמזות על טקטיקות הונאה מגוונות. השתמשו בזה כדי לתעדף אמצעים נגד הונאה.'
                }
            }
        }
    
    def _setup_hebrew_fonts(self):
        """Setup Hebrew fonts for both ReportLab and matplotlib with comprehensive fallback system."""
        logger = logging.getLogger(__name__)
        
        # Font candidates with priorities and capabilities
        font_candidates = [
            {
                'path': '/home/fiod/shimshi/fonts/NotoSansHebrew.ttf',
                'name': 'NotoSansHebrew',
                'priority': 1,
                'has_latin': False,  # Hebrew-only font
                'description': 'Noto Sans Hebrew (Primary)'
            },
            {
                'path': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                'name': 'DejaVuSans',
                'priority': 2,
                'has_latin': True,   # Supports both Hebrew and Latin
                'description': 'DejaVu Sans (Fallback)'
            },
            {
                'path': '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                'name': 'LiberationSans',
                'priority': 3,
                'has_latin': True,   # Supports both Hebrew and Latin
                'description': 'Liberation Sans (Fallback)'
            }
        ]
        
        self.registered_fonts = {}
        self.primary_hebrew_font = None
        self.fallback_latin_font = None
        
        # Try to register fonts in priority order
        for font_info in sorted(font_candidates, key=lambda x: x['priority']):
            if os.path.exists(font_info['path']):
                try:
                    # Test if font file is valid
                    with open(font_info['path'], 'rb') as f:
                        header = f.read(12)
                        if len(header) < 12:
                            logger.warning(f"Font file too small: {font_info['path']}")
                            continue
                    
                    # Register with ReportLab
                    font_name = font_info['name']
                    pdfmetrics.registerFont(TTFont(font_name, font_info['path']))
                    pdfmetrics.registerFont(TTFont(f'{font_name}-Bold', font_info['path']))  # Use same font for bold
                    
                    # Store font info
                    self.registered_fonts[font_name] = {
                        'path': font_info['path'],
                        'has_latin': font_info['has_latin'],
                        'description': font_info['description']
                    }
                    
                    # Set primary Hebrew font
                    if not self.primary_hebrew_font:
                        self.primary_hebrew_font = font_name
                        self.hebrew_font_available = True
                    
                    # Set fallback for Latin characters if font supports them
                    if font_info['has_latin'] and not self.fallback_latin_font:
                        self.fallback_latin_font = font_name
                    
                    # Configure matplotlib
                    self._setup_matplotlib_hebrew_font(font_info['path'])
                    
                    logger.info(f"✓ Registered font: {font_info['description']} -> {font_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to register font {font_info['path']}: {e}")
                    continue
            else:
                logger.debug(f"Font not found: {font_info['path']}")
        
        # Setup font configuration
        if self.hebrew_font_available:
            # Use primary Hebrew font or fallback to Latin-supporting font
            if self.primary_hebrew_font:
                self.languages['he']['font'] = self.primary_hebrew_font
                logger.info(f"✓ Hebrew font configured: {self.primary_hebrew_font}")
            elif self.fallback_latin_font:
                self.languages['he']['font'] = self.fallback_latin_font
                logger.info(f"✓ Hebrew font configured (fallback): {self.fallback_latin_font}")
        else:
            # Ultimate fallback
            logger.error("No Hebrew fonts available, using Helvetica")
            self.languages['he']['font'] = 'Helvetica'
        
        # Summary
        logger.info(f"Font setup complete:")
        logger.info(f"  - Hebrew support: {'✓' if self.hebrew_font_available else '✗'}")
        logger.info(f"  - Primary Hebrew font: {self.primary_hebrew_font or 'None'}")
        logger.info(f"  - Fallback Latin font: {self.fallback_latin_font or 'None'}")
        logger.info(f"  - Registered fonts: {len(self.registered_fonts)}")
    
    def _setup_matplotlib_hebrew_font(self, font_path: str):
        """Configure matplotlib to use Hebrew font with proper fallback system."""
        try:
            # Add font to matplotlib's font manager
            fm.fontManager.addfont(font_path)
            
            # Get font properties
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            
            # Configure matplotlib font preferences
            # Use a list of fonts with Hebrew support, in preference order
            font_list = []
            
            # Add the specific font we just loaded
            if font_name:
                font_list.append(font_name)
            
            # Add reliable fallback fonts with Hebrew support
            font_list.extend(['DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS', 'Lucida Grande'])
            
            # Set matplotlib font configuration
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = font_list
            
            # Ensure proper Unicode handling
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 10
            
            # Store matplotlib font info
            self.matplotlib_hebrew_font = font_name
            
            logging.getLogger(__name__).info(f"✓ Matplotlib configured for Hebrew: {font_name} + fallbacks")
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to configure matplotlib for Hebrew: {e}")
            # Fallback configuration
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
            self.matplotlib_hebrew_font = 'DejaVu Sans'
    
    def _process_hebrew_text(self, text: str) -> str:
        """Process Hebrew text for proper RTL display."""
        if not text or not HAS_BIDI:
            return text
        
        try:
            # Check if text contains Hebrew characters
            if any('\u0590' <= char <= '\u05FF' or '\uFB1D' <= char <= '\uFB4F' for char in text):
                # Apply RTL processing
                return get_display(text)
            return text
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to process Hebrew text: {e}")
            return text
    
    def t(self, key: str, lang: str) -> str:
        """Get translation for a key in specified language with proper RTL processing."""
        text = self.translations.get(lang, self.translations['en']).get(key, key)
        if lang == 'he':
            text = self._process_hebrew_text(text)
        return text
    
    def get_plot_description(self, plot_key: str, lang: str) -> Dict[str, str]:
        """Get plot description in specified language."""
        descriptions = self.plot_descriptions.get(plot_key, {})
        lang_desc = descriptions.get(lang, descriptions.get('en', {}))
        return {
            'what': lang_desc.get('what', ''),
            'how': lang_desc.get('how', '')
        }
    
    def create_styles(self, lang: str):
        """Create language-specific styles."""
        styles = getSampleStyleSheet()
        
        # Determine text alignment based on language
        if self.languages[lang]['direction'] == 'rtl':
            title_align = TA_RIGHT
            normal_align = TA_RIGHT
            heading_align = TA_RIGHT
        else:
            title_align = TA_CENTER
            normal_align = TA_LEFT
            heading_align = TA_LEFT
        
        # Get appropriate font name with smart fallback
        font_name = self.languages[lang]['font']
        if lang == 'he':
            if not self.hebrew_font_available:
                font_name = 'Helvetica'
                logger.warning(f"Using fallback font for Hebrew: {font_name}")
            else:
                # Verify the font is actually registered
                try:
                    pdfmetrics.getFont(font_name)
                except Exception as e:
                    logger.warning(f"Hebrew font {font_name} not accessible, using fallback")
                    font_name = self.fallback_latin_font or 'Helvetica'
            
        # Title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=title_align,
            fontName=font_name
        ))
        
        # Subtitle style
        styles.add(ParagraphStyle(
            name='Subtitle',
            parent=styles['Title'],
            fontSize=18,
            textColor=colors.HexColor('#666666'),
            spaceAfter=20,
            alignment=title_align,
            fontName=font_name
        ))
        
        # Section heading style
        styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=20,
            alignment=heading_align,
            fontName=font_name
        ))
        
        # Subsection heading style
        styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#ff7f0e'),
            spaceAfter=10,
            spaceBefore=15,
            alignment=heading_align,
            fontName=font_name
        ))
        
        # Normal text style
        styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=normal_align,
            fontName=font_name
        ))
        
        # Description style
        styles.add(ParagraphStyle(
            name='Description',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#555555'),
            alignment=normal_align,
            fontName=font_name,
            spaceAfter=6
        ))
        
        return styles
    
    def add_plot_description(self, story: List, plot_key: str, lang: str, styles):
        """Add plot description in the specified language."""
        desc = self.get_plot_description(plot_key, lang)
        if desc['what'] or desc['how']:
            story.append(Spacer(1, 0.2 * inch))
            
            if desc['what']:
                what_label = self._process_hebrew_text(self.t('what_shows', lang)) if lang == 'he' else self.t('what_shows', lang)
                what_text = self._process_hebrew_text(desc['what']) if lang == 'he' else desc['what']
                story.append(Paragraph(f"<b>{what_label}</b> {what_text}", 
                                     styles['Description']))
            
            if desc['how']:
                how_label = self._process_hebrew_text(self.t('how_interpret', lang)) if lang == 'he' else self.t('how_interpret', lang)
                how_text = self._process_hebrew_text(desc['how']) if lang == 'he' else desc['how']
                story.append(Paragraph(f"<b>{how_label}</b> {how_text}", 
                                     styles['Description']))
            
            story.append(Spacer(1, 0.3 * inch))
    
    def create_quality_distribution_plot(self, df: pd.DataFrame, lang: str) -> str:
        """Create quality score distribution histogram."""
        plt.figure(figsize=(10, 6))
        
        try:
            # Check for sufficient data and unique values
            unique_values = len(df['quality_score'].unique())
            
            if unique_values <= 1:
                # Handle single unique value case
                bins = 1
                logger.warning(f"Only {unique_values} unique quality score values, using single bin")
            elif unique_values <= 5:
                # Use fewer bins for limited unique values
                bins = min(unique_values, 5)
                logger.warning(f"Limited unique values ({unique_values}), using {bins} bins")
            else:
                # Standard case
                bins = min(20, unique_values // 2)
            
            # Create the histogram
            plt.hist(df['quality_score'], bins=bins, edgecolor='black', alpha=0.7, color='#1f77b4')
            
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            # Fallback: create a simple bar chart instead
            value_counts = df['quality_score'].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values, color='#1f77b4', alpha=0.7)
            plt.xticks(range(len(value_counts)), [f'{v:.2f}' for v in value_counts.index])
            if lang == 'he':
                plt.xlabel(self._process_hebrew_text('ערכי ציון איכות'), fontsize=12)
            else:
                plt.xlabel('Quality Score Values', fontsize=12)
        
        # Set labels based on language with proper Hebrew support
        if lang == 'he':
            plt.xlabel(self._process_hebrew_text('ציון איכות'), fontsize=12)
            plt.ylabel(self._process_hebrew_text('מספר ערוצים'), fontsize=12)
            plt.title(self._process_hebrew_text('התפלגות ציוני איכות ערוצים'), fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Quality Score', fontsize=12)
            plt.ylabel('Number of Channels', fontsize=12)
            plt.title('Channel Quality Score Distribution', fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_score = df['quality_score'].mean()
        median_score = df['quality_score'].median()
        
        if lang == 'he':
            stats_text = self._process_hebrew_text(f'ממוצע: {mean_score:.2f}\nחציון: {median_score:.2f}')
        else:
            stats_text = f'Mean: {mean_score:.2f}\nMedian: {median_score:.2f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'quality_distribution_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_quality_volume_scatter(self, df: pd.DataFrame, lang: str) -> str:
        """Create quality score vs volume scatter plot."""
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot with color gradient
        scatter = plt.scatter(df['volume'], df['quality_score'], 
                            c=df['bot_rate'], cmap='RdYlGn_r', 
                            alpha=0.6, s=50)
        
        # Set scale and labels
        plt.xscale('log')
        
        if lang == 'he':
            plt.xlabel(self._process_hebrew_text('נפח תעבורה (סולם לוגריתמי)'), fontsize=12)
            plt.ylabel(self._process_hebrew_text('ציון איכות'), fontsize=12)
            plt.title(self._process_hebrew_text('ציון איכות מול נפח תעבורה'), fontsize=14, fontweight='bold')
            cbar = plt.colorbar(scatter)
            cbar.set_label(self._process_hebrew_text('שיעור בוטים'), fontsize=10)
        else:
            plt.xlabel('Traffic Volume (log scale)', fontsize=12)
            plt.ylabel('Quality Score', fontsize=12)
            plt.title('Quality Score vs Traffic Volume', fontsize=14, fontweight='bold')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Bot Rate', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filename = f'quality_volume_scatter_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_bot_rate_boxplot(self, df: pd.DataFrame, lang: str) -> str:
        """Create bot rate analysis by quality category."""
        plt.figure(figsize=(10, 6))
        
        # Prepare data with error handling
        quality_categories = ['Low', 'Medium-Low', 'Medium-High', 'High']
        
        try:
            # Create boxplot
            df_copy = df.copy()
            
            # Check if quality_category column exists and has valid data
            if 'quality_category' not in df_copy.columns:
                logger.warning("quality_category column missing, creating from quality_score")
                # Create categories based on quality score quartiles
                df_copy['quality_category'] = pd.qcut(df_copy['quality_score'], q=4, labels=quality_categories, duplicates='drop')
            
            # Ensure we have data in multiple categories
            category_counts = df_copy['quality_category'].value_counts()
            if len(category_counts) <= 1:
                logger.warning("Only one quality category found, creating simple bar plot instead")
                # Fallback to simple bar chart
                mean_bot_rate = df_copy['bot_rate'].mean()
                plt.bar([0], [mean_bot_rate], color='#1f77b4', alpha=0.7)
                plt.xticks([0], [str(df_copy['quality_category'].iloc[0])])
                box_plot = plt.gca()
            else:
                df_copy['quality_category'] = pd.Categorical(df_copy['quality_category'], 
                                                            categories=quality_categories, 
                                                            ordered=True)
                df_copy = df_copy.sort_values('quality_category')
                
                box_plot = df_copy.boxplot(column='bot_rate', by='quality_category', 
                                          patch_artist=True, figsize=(10, 6))
                
        except Exception as e:
            logger.error(f"Error creating boxplot: {e}")
            # Ultimate fallback: simple bar chart of mean bot rates
            if 'quality_category' in df.columns:
                means = df.groupby('quality_category')['bot_rate'].mean()
            else:
                means = pd.Series([df['bot_rate'].mean()], index=['All Data'])
            
            plt.bar(range(len(means)), means.values, color='#1f77b4', alpha=0.7)
            plt.xticks(range(len(means)), means.index)
            box_plot = plt.gca()
        
        # Customize colors (only if we have actual boxplot artists)
        try:
            if hasattr(box_plot, 'artists') and len(box_plot.artists) > 0:
                colors_list = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c']
                for patch, color in zip(box_plot.artists, colors_list):
                    patch.set_facecolor(color)
        except Exception as e:
            logger.warning(f"Could not customize boxplot colors: {e}")
        
        # Set labels based on language
        if lang == 'he':
            plt.xlabel(self._process_hebrew_text('קטגוריית איכות'), fontsize=12)
            plt.ylabel(self._process_hebrew_text('שיעור בוטים'), fontsize=12)
            plt.title(self._process_hebrew_text('התפלגות שיעור בוטים לפי קטגוריית איכות'), fontsize=14, fontweight='bold')
            plt.suptitle('')  # Remove default title
            # Update x-axis labels for Hebrew only if they match the expected structure
            ax = plt.gca()
            current_labels = ax.get_xticklabels()
            hebrew_labels = [self._process_hebrew_text('נמוך'), self._process_hebrew_text('בינוני-נמוך'), self._process_hebrew_text('בינוני-גבוה'), self._process_hebrew_text('גבוה')]
            
            # Only set Hebrew labels if the count matches
            if len(current_labels) <= len(hebrew_labels):
                ax.set_xticklabels(hebrew_labels[:len(current_labels)])
        else:
            plt.xlabel('Quality Category', fontsize=12)
            plt.ylabel('Bot Rate', fontsize=12)
            plt.title('Bot Rate Distribution by Quality Category', fontsize=14, fontweight='bold')
            plt.suptitle('')  # Remove default title
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save figure
        filename = f'bot_rate_boxplot_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_risk_matrix(self, df: pd.DataFrame, lang: str) -> str:
        """Create risk assessment matrix heatmap."""
        plt.figure(figsize=(10, 8))
        
        try:
            # Create risk matrix data with robust binning
            bot_rate_min, bot_rate_max = df['bot_rate'].min(), df['bot_rate'].max()
            volume_min, volume_max = df['volume'].min(), df['volume'].max()
            
            # Handle edge cases where min == max
            if bot_rate_min == bot_rate_max:
                # Single value case
                bot_rate_bins = [bot_rate_min - 0.01, bot_rate_min + 0.01]
                bot_rate_labels = [f'{bot_rate_min:.1%}']
                logger.warning(f"Single bot_rate value detected: {bot_rate_min}")
            else:
                # Normal case with proper bins
                bot_rate_bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
                # Ensure bins cover the data range
                if bot_rate_max > 1.0:
                    bot_rate_bins[-1] = bot_rate_max + 0.01
                if bot_rate_min < 0:
                    bot_rate_bins[0] = bot_rate_min - 0.01
                bot_rate_labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-100%']
            
            if volume_min == volume_max:
                # Single value case
                volume_bins = [volume_min - 1, volume_min + 1]
                volume_labels = [f'{volume_min:.0f}']
                logger.warning(f"Single volume value detected: {volume_min}")
            else:
                # Normal case with proper bins
                volume_bins = [0, 10, 100, 1000, 10000, max(volume_max + 1, 10000)]
                volume_labels = ['0-10', '10-100', '100-1K', '1K-10K', '10K+']
                
                # Adjust bins if data doesn't fit standard ranges
                if volume_max < 10:
                    volume_bins = [0, volume_max/3, 2*volume_max/3, volume_max + 1]
                    volume_labels = ['Low', 'Medium', 'High']
                elif volume_max < 100:
                    volume_bins = [0, 10, 50, volume_max + 1]
                    volume_labels = ['0-10', '10-50', '50+']
            
            # Ensure bins are monotonically increasing
            bot_rate_bins = sorted(set(bot_rate_bins))
            volume_bins = sorted(set(volume_bins))
            
            # Adjust labels if bins were modified
            if len(bot_rate_bins) - 1 != len(bot_rate_labels):
                bot_rate_labels = [f'Bin{i}' for i in range(len(bot_rate_bins) - 1)]
            if len(volume_bins) - 1 != len(volume_labels):
                volume_labels = [f'Bin{i}' for i in range(len(volume_bins) - 1)]
            
            # Create bins
            df_temp = df.copy()
            df_temp['bot_rate_bin'] = pd.cut(df_temp['bot_rate'], bins=bot_rate_bins, labels=bot_rate_labels, include_lowest=True)
            df_temp['volume_bin'] = pd.cut(df_temp['volume'], bins=volume_bins, labels=volume_labels, include_lowest=True)
        
            # Create pivot table
            risk_matrix = pd.crosstab(df_temp['bot_rate_bin'], df_temp['volume_bin'])
            
        except Exception as e:
            logger.error(f"Error creating risk matrix bins: {e}")
            # Fallback: create simple 2x2 matrix based on medians
            df_temp = df.copy()
            bot_rate_median = df['bot_rate'].median()
            volume_median = df['volume'].median()
            
            df_temp['bot_rate_bin'] = df['bot_rate'].apply(lambda x: 'High' if x >= bot_rate_median else 'Low')
            df_temp['volume_bin'] = df['volume'].apply(lambda x: 'High' if x >= volume_median else 'Low')
            
            risk_matrix = pd.crosstab(df_temp['bot_rate_bin'], df_temp['volume_bin'])
        
        # Create heatmap  
        cbar_label = self._process_hebrew_text(self.t('channel_id', lang)) if lang == 'he' else 'Channel Count'
        sns.heatmap(risk_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': cbar_label})
        
        # Set labels based on language
        if lang == 'he':
            plt.xlabel(self._process_hebrew_text('נפח תעבורה'), fontsize=12)
            plt.ylabel(self._process_hebrew_text('שיעור בוטים'), fontsize=12)
            plt.title(self._process_hebrew_text('מטריצת הערכת סיכונים'), fontsize=14, fontweight='bold')
            # Update labels for Hebrew only if they match the actual data structure
            ax = plt.gca()
            current_xlabels = ax.get_xticklabels()
            current_ylabels = ax.get_yticklabels()
            
            # Only update labels if they match expected counts
            if len(current_xlabels) <= 5:
                hebrew_volume_labels = ['0-10', '10-100', '100-1K', '1K-10K', '10K+'][:len(current_xlabels)]
                ax.set_xticklabels(hebrew_volume_labels)
            
            if len(current_ylabels) <= 5:
                hebrew_bot_rate_labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-100%'][:len(current_ylabels)]
                ax.set_yticklabels(hebrew_bot_rate_labels)
        else:
            plt.xlabel('Traffic Volume', fontsize=12)
            plt.ylabel('Bot Rate', fontsize=12)
            plt.title('Risk Assessment Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'risk_matrix_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_anomaly_heatmap(self, anomaly_df: pd.DataFrame, lang: str) -> str:
        """Create anomaly detection heatmap."""
        if anomaly_df.empty:
            return None
            
        plt.figure(figsize=(12, 8))
        
        # Select anomaly columns
        anomaly_cols = [col for col in anomaly_df.columns 
                       if 'anomaly' in col and col not in ['overall_anomaly_count', 'overall_anomaly_flag']]
        
        if not anomaly_cols:
            return None
        
        # Get top anomalous channels with safety checks
        if 'overall_anomaly_count' in anomaly_df.columns:
            # Ensure we have some non-zero anomaly counts
            non_zero_anomalies = anomaly_df[anomaly_df['overall_anomaly_count'] > 0]
            if len(non_zero_anomalies) > 0:
                top_anomalous = non_zero_anomalies.nlargest(min(20, len(non_zero_anomalies)), 'overall_anomaly_count')
            else:
                # No actual anomalies found, use all data
                top_anomalous = anomaly_df.head(min(20, len(anomaly_df)))
                logger.warning("No channels with anomalies > 0 found")
        else:
            top_anomalous = anomaly_df.head(min(20, len(anomaly_df)))
        
        # Create binary matrix
        anomaly_matrix = top_anomalous[anomaly_cols].astype(int)
        
        # Create heatmap
        cbar_label = self._process_hebrew_text(self.t('anomaly_count', lang)) if lang == 'he' else 'Anomaly Present'
        sns.heatmap(anomaly_matrix, cmap='RdYlGn_r', cbar_kws={'label': cbar_label})
        
        # Set labels based on language
        if lang == 'he':
            plt.xlabel(self._process_hebrew_text('סוג חריגה'), fontsize=12)
            plt.ylabel(self._process_hebrew_text('מזהה ערוץ'), fontsize=12)
            plt.title(self._process_hebrew_text('מפת חום זיהוי חריגות - 20 הערוצים החריגים ביותר'), fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Anomaly Type', fontsize=12)
            plt.ylabel('Channel ID', fontsize=12)
            plt.title('Anomaly Detection Heatmap - Top 20 Most Anomalous Channels', fontsize=14, fontweight='bold')
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        filename = f'anomaly_heatmap_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_anomaly_distribution(self, anomaly_df: pd.DataFrame, lang: str) -> str:
        """Create anomaly type distribution bar chart."""
        if anomaly_df.empty:
            return None
            
        plt.figure(figsize=(10, 6))
        
        # Count anomalies by type
        anomaly_cols = [col for col in anomaly_df.columns 
                       if 'anomaly' in col and col not in ['overall_anomaly_count', 'overall_anomaly_flag']]
        
        if not anomaly_cols:
            return None
        
        anomaly_counts = {}
        for col in anomaly_cols:
            if anomaly_df[col].dtype == bool:
                count = anomaly_df[col].sum()
                clean_name = col.replace('_anomaly', '').replace('_', ' ').title()
                anomaly_counts[clean_name] = count
        
        # Sort by count
        sorted_anomalies = dict(sorted(anomaly_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Create bar chart
        plt.bar(sorted_anomalies.keys(), sorted_anomalies.values(), color='#ff7f0e')
        
        # Set labels based on language
        if lang == 'he':
            plt.xlabel(self._process_hebrew_text('סוג חריגה'), fontsize=12)
            plt.ylabel(self._process_hebrew_text('מספר ערוצים'), fontsize=12)
            plt.title(self._process_hebrew_text('התפלגות סוגי חריגות'), fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Anomaly Type', fontsize=12)
            plt.ylabel('Number of Channels', fontsize=12)
            plt.title('Anomaly Type Distribution', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save figure
        filename = f'anomaly_distribution_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_cluster_visualization(self, quality_df: pd.DataFrame, lang: str) -> str:
        """Create cluster visualization using t-SNE."""
        plt.figure(figsize=(10, 8))
        
        # For demonstration, create synthetic clusters based on quality score and bot rate
        from sklearn.preprocessing import StandardScaler
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        
        # Prepare features
        features = quality_df[['quality_score', 'bot_rate', 'volume']].copy()
        features['volume'] = np.log1p(features['volume'])  # Log transform volume
        
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering with safety checks
        n_samples = len(quality_df)
        if n_samples < 2:
            logger.warning("Insufficient data for clustering, creating single cluster")
            clusters = np.zeros(n_samples)
            n_clusters = 1
        else:
            n_clusters = max(1, min(5, n_samples // 10))  # Ensure reasonable number of clusters
            if n_clusters == 1:
                clusters = np.zeros(n_samples)
            else:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(features_scaled)
                except Exception as e:
                    logger.error(f"K-means clustering failed: {e}")
                    clusters = np.zeros(n_samples)
                    n_clusters = 1
        
        # Perform t-SNE with safety checks
        try:
            if n_samples < 3:
                # Not enough samples for t-SNE, create dummy 2D coordinates
                tsne_results = np.random.rand(n_samples, 2)
                logger.warning("Insufficient data for t-SNE, using random coordinates")
            else:
                perplexity = min(30, max(1, n_samples - 1))
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                sample_size = min(1000, len(features_scaled))
                tsne_results = tsne.fit_transform(features_scaled[:sample_size])
        except Exception as e:
            logger.error(f"t-SNE failed: {e}")
            # Fallback: use PCA or simple 2D projection
            sample_size = min(1000, len(features_scaled))
            tsne_results = features_scaled[:sample_size, :2] if features_scaled.shape[1] >= 2 else np.random.rand(sample_size, 2)  # Limit for performance
        
        # Plot
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                            c=clusters[:len(tsne_results)], cmap='viridis', 
                            alpha=0.6, s=50)
        
        # Set labels based on language
        if lang == 'he':
            plt.xlabel(self._process_hebrew_text('t-SNE מימד 1'), fontsize=12)
            plt.ylabel(self._process_hebrew_text('t-SNE מימד 2'), fontsize=12)
            plt.title(self._process_hebrew_text('אשכולות דפוסי תעבורה'), fontsize=14, fontweight='bold')
            cbar = plt.colorbar(scatter)
            cbar.set_label(self._process_hebrew_text('אשכול'), fontsize=10)
        else:
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.title('Traffic Pattern Clusters', fontsize=14, fontweight='bold')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Cluster', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filename = f'cluster_visualization_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store cluster info for later use
        quality_df['cluster'] = -1  # Initialize with -1
        quality_df.loc[:len(clusters)-1, 'cluster'] = clusters
        
        return filepath
    
    def create_cluster_quality_chart(self, quality_df: pd.DataFrame, lang: str) -> str:
        """Create average quality score by cluster bar chart."""
        plt.figure(figsize=(10, 6))
        
        # Calculate average quality by cluster
        if 'cluster' in quality_df.columns:
            cluster_quality = quality_df[quality_df['cluster'] != -1].groupby('cluster')['quality_score'].mean().sort_values(ascending=False)
        else:
            # Create synthetic clusters if not available
            try:
                unique_values = len(quality_df['quality_score'].unique())
                if unique_values <= 1:
                    # Single value case - create one cluster
                    quality_df['cluster'] = 'Cluster 0'
                    logger.warning("Single unique quality score value, creating single cluster")
                elif unique_values < 5:
                    # Few unique values - use fewer quantiles
                    q = min(unique_values, 3)
                    quality_df['cluster'] = pd.qcut(quality_df['quality_score'], q=q, labels=[f'Cluster {i}' for i in range(q)], duplicates='drop')
                    logger.warning(f"Limited unique values ({unique_values}), using {q} clusters")
                else:
                    # Normal case
                    quality_df['cluster'] = pd.qcut(quality_df['quality_score'], q=5, labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], duplicates='drop')
                    
            except Exception as e:
                logger.error(f"Error creating clusters with qcut: {e}")
                # Ultimate fallback: use simple binning based on value ranges
                min_score, max_score = quality_df['quality_score'].min(), quality_df['quality_score'].max()
                if min_score == max_score:
                    quality_df['cluster'] = 'Cluster 0'
                else:
                    # Create 3 equal-width bins
                    range_size = (max_score - min_score) / 3
                    quality_df['cluster'] = quality_df['quality_score'].apply(lambda x: 
                        'Cluster 0' if x < min_score + range_size
                        else 'Cluster 1' if x < min_score + 2 * range_size
                        else 'Cluster 2'
                    )
            
            cluster_quality = quality_df.groupby('cluster')['quality_score'].mean()
        
        # Create bar chart
        colors_list = plt.cm.viridis(np.linspace(0, 1, len(cluster_quality)))
        bars = plt.bar(cluster_quality.index.astype(str), cluster_quality.values, color=colors_list)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Set labels based on language
        if lang == 'he':
            plt.xlabel(self._process_hebrew_text('אשכול'), fontsize=12)
            plt.ylabel(self._process_hebrew_text('ציון איכות ממוצע'), fontsize=12)
            plt.title(self._process_hebrew_text('ציון איכות ממוצע לפי אשכול'), fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Cluster', fontsize=12)
            plt.ylabel('Average Quality Score', fontsize=12)
            plt.title('Average Quality Score by Cluster', fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save figure
        filename = f'cluster_quality_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_quality_trend_plot(self, quality_df: pd.DataFrame, lang: str) -> str:
        """Create quality score trend over time."""
        plt.figure(figsize=(12, 6))
        
        # For demonstration, create synthetic time-based data
        # Group by volume quartiles as proxy for time periods
        try:
            n_periods = min(10, len(quality_df) // 2)  # Ensure we have enough data per period
            if n_periods <= 1:
                # Not enough data for multiple periods
                quality_df['period'] = 'P1'
                logger.warning("Insufficient data for multiple time periods, using single period")
            else:
                # Use qcut with duplicates handling
                quality_df['period'] = pd.qcut(quality_df.index, q=n_periods, labels=[f'P{i+1}' for i in range(n_periods)], duplicates='drop')
                
        except Exception as e:
            logger.error(f"Error creating time periods with qcut: {e}")
            # Fallback: create periods based on index ranges
            n_periods = min(5, len(quality_df))
            period_size = len(quality_df) // n_periods
            quality_df['period'] = quality_df.index // max(1, period_size)
            quality_df['period'] = quality_df['period'].apply(lambda x: f'P{int(x)+1}')
        
        trend_data = quality_df.groupby('period')['quality_score'].agg(['mean', 'std'])
        
        # Plot trend with confidence interval
        x = range(len(trend_data))
        avg_label = self._process_hebrew_text(self.t('avg_quality', lang)) if lang == 'he' else 'Average Quality'
        plt.plot(x, trend_data['mean'], 'b-', linewidth=2, label=avg_label)
        plt.fill_between(x, 
                        trend_data['mean'] - trend_data['std'], 
                        trend_data['mean'] + trend_data['std'], 
                        alpha=0.3, color='blue')
        
        # Set labels based on language
        if lang == 'he':
            plt.xlabel(self._process_hebrew_text('תקופה'), fontsize=12)
            plt.ylabel(self._process_hebrew_text('ציון איכות'), fontsize=12)
            plt.title(self._process_hebrew_text('מגמת ציון איכות לאורך זמן'), fontsize=14, fontweight='bold')
            plt.legend([self._process_hebrew_text('ציון ממוצע'), self._process_hebrew_text('סטיית תקן')])
        else:
            plt.xlabel('Time Period', fontsize=12)
            plt.ylabel('Quality Score', fontsize=12)
            plt.title('Quality Score Trend Over Time', fontsize=14, fontweight='bold')
            plt.legend(['Average Score', 'Standard Deviation'])
        
        plt.grid(True, alpha=0.3)
        plt.xticks(x, trend_data.index)
        plt.tight_layout()
        
        # Save figure
        filename = f'quality_trend_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_fraud_type_pie(self, anomaly_df: pd.DataFrame, lang: str) -> str:
        """Create fraud type distribution pie chart."""
        if anomaly_df.empty:
            return None
            
        plt.figure(figsize=(10, 8))
        
        # Count anomalies by type
        anomaly_cols = [col for col in anomaly_df.columns 
                       if 'anomaly' in col and col not in ['overall_anomaly_count', 'overall_anomaly_flag']]
        
        if not anomaly_cols:
            return None
        
        anomaly_counts = {}
        for col in anomaly_cols[:6]:  # Limit to top 6 for readability
            if anomaly_df[col].dtype == bool:
                count = anomaly_df[col].sum()
                if count > 0:  # Only include non-zero counts
                    clean_name = col.replace('_anomaly', '').replace('_', ' ').title()
                    anomaly_counts[clean_name] = count
        
        if not anomaly_counts:
            return None
        
        # Create pie chart
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(anomaly_counts)))
        wedges, texts, autotexts = plt.pie(anomaly_counts.values(), 
                                          labels=anomaly_counts.keys(), 
                                          autopct='%1.1f%%',
                                          colors=colors_list,
                                          startangle=90)
        
        # Enhance text
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        # Set title based on language
        if lang == 'he':
            plt.title(self._process_hebrew_text('התפלגות סוגי הונאה שזוהו'), fontsize=14, fontweight='bold', pad=20)
        else:
            plt.title('Detected Fraud Type Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.axis('equal')
        plt.tight_layout()
        
        # Save figure
        filename = f'fraud_type_pie_{lang}.png'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_executive_summary(self, quality_df: pd.DataFrame, anomaly_df: pd.DataFrame, 
                               pipeline_results: Dict, lang: str, story: List, styles):
        """Create executive summary section."""
        story.append(Paragraph(self.t('executive_summary', lang), styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Calculate key metrics
        total_channels = len(quality_df)
        high_risk_channels = len(quality_df[quality_df['high_risk'] == True])
        anomalous_channels = 0
        if not anomaly_df.empty and 'overall_anomaly_count' in anomaly_df.columns:
            anomalous_channels = len(anomaly_df[anomaly_df['overall_anomaly_count'] > 0])
        
        avg_quality_score = quality_df['quality_score'].mean()
        avg_bot_rate = quality_df['bot_rate'].mean()
        total_volume = quality_df['volume'].sum()
        
        # Create TL;DR section
        story.append(Paragraph(f"<b>{self.t('tldr', lang)}</b>", styles['SubsectionHeading']))
        
        # Key metrics table
        metrics_data = [
            [self.t('key_metrics', lang), ''],
            [self.t('total_channels', lang), f'{total_channels:,}'],
            [self.t('high_risk_count', lang), f'{high_risk_channels:,} ({high_risk_channels/total_channels*100:.1f}%)'],
            [self.t('anomalous_count', lang), f'{anomalous_channels:,} ({anomalous_channels/total_channels*100:.1f}%)'],
            [self.t('avg_quality', lang), f'{avg_quality_score:.2f}/10'],
            [self.t('avg_bot_rate', lang), f'{avg_bot_rate*100:.1f}%'],
            [self.t('total_volume', lang), f'{total_volume:,}']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT' if lang != 'he' else 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Critical findings
        story.append(Paragraph(f"<b>{self.t('critical_findings', lang)}</b>", styles['SubsectionHeading']))
        
        findings_text = []
        if high_risk_channels > total_channels * 0.1:
            if lang == 'he':
                findings_text.append(f"• <b>סיכון גבוה:</b> {high_risk_channels} ערוצים ({high_risk_channels/total_channels*100:.1f}%) מסווגים כסיכון גבוה")
            else:
                findings_text.append(f"• <b>High Risk:</b> {high_risk_channels} channels ({high_risk_channels/total_channels*100:.1f}%) classified as high-risk")
        
        if avg_bot_rate > 0.3:
            if lang == 'he':
                findings_text.append(f"• <b>בעיית בוטים:</b> שיעור בוטים ממוצע של {avg_bot_rate*100:.1f}% מצביע על בעיה מערכתית")
            else:
                findings_text.append(f"• <b>Bot Issue:</b> Average bot rate of {avg_bot_rate*100:.1f}% indicates systemic problem")
        
        if anomalous_channels > total_channels * 0.2:
            if lang == 'he':
                findings_text.append(f"• <b>חריגות נרחבות:</b> {anomalous_channels} ערוצים מציגים דפוסים חריגים")
            else:
                findings_text.append(f"• <b>Widespread Anomalies:</b> {anomalous_channels} channels show anomalous patterns")
        
        for finding in findings_text:
            story.append(Paragraph(finding, styles['CustomNormal']))
        
        story.append(Spacer(1, 0.3 * inch))
        
        # Action items
        story.append(Paragraph(f"<b>{self.t('action_items', lang)}</b>", styles['SubsectionHeading']))
        
        if lang == 'he':
            action_items = [
                f"1. חקור מיידית {min(high_risk_channels, 50)} ערוצים בסיכון הגבוה ביותר",
                f"2. בדוק {min(anomalous_channels, 100)} ערוצים עם חריגות מרובות",
                f"3. הטמע סינון אוטומטי לערוצים עם ציון איכות < 3.0",
                f"4. צור התראות בזמן אמת לדפוסי הונאה חדשים"
            ]
        else:
            action_items = [
                f"1. Immediately investigate top {min(high_risk_channels, 50)} high-risk channels",
                f"2. Review {min(anomalous_channels, 100)} channels with multiple anomalies",
                f"3. Implement automated filtering for channels with quality score < 3.0",
                f"4. Set up real-time alerts for new fraud patterns"
            ]
        
        for item in action_items:
            story.append(Paragraph(item, styles['CustomNormal']))
        
        story.append(PageBreak())
    
    def create_recommendations_section(self, quality_df: pd.DataFrame, anomaly_df: pd.DataFrame,
                                     lang: str, story: List, styles):
        """Create recommendations section."""
        story.append(Paragraph(self.t('recommendations', lang), styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Immediate actions
        story.append(Paragraph(f"<b>{self.t('immediate_actions', lang)}</b>", styles['SubsectionHeading']))
        
        high_risk_count = len(quality_df[quality_df['high_risk'] == True])
        
        if lang == 'he':
            immediate_actions = [
                f"1. <b>חסום/חקור ערוצים בסיכון גבוה:</b> {high_risk_count} ערוצים זוהו כסיכון גבוה עם שיעור בוטים ממוצע של {quality_df[quality_df['high_risk'] == True]['bot_rate'].mean()*100:.1f}%",
                f"2. <b>בדוק דפוסים חריגים:</b> התמקד בערוצים עם מספר סוגי חריגות",
                f"3. <b>הגן על הכנסות:</b> פוטנציאל הכנסה בסיכון: ${quality_df[quality_df['high_risk'] == True]['volume'].sum() * 0.1:.2f}"
            ]
        else:
            immediate_actions = [
                f"1. <b>Block/Investigate High-Risk Channels:</b> {high_risk_count} channels identified as high-risk with average bot rate of {quality_df[quality_df['high_risk'] == True]['bot_rate'].mean()*100:.1f}%",
                f"2. <b>Review Anomalous Patterns:</b> Focus on channels with multiple anomaly types",
                f"3. <b>Protect Revenue:</b> Potential revenue at risk: ${quality_df[quality_df['high_risk'] == True]['volume'].sum() * 0.1:.2f}"
            ]
        
        for action in immediate_actions:
            story.append(Paragraph(action, styles['CustomNormal']))
        
        story.append(Spacer(1, 0.2 * inch))
        
        # Short-term improvements
        story.append(Paragraph(f"<b>{self.t('short_term', lang)}</b>", styles['SubsectionHeading']))
        
        medium_low_count = len(quality_df[quality_df['quality_category'] == 'Medium-Low'])
        
        if lang == 'he':
            short_term = [
                f"1. <b>שיפור איכות:</b> עבוד עם {medium_low_count} ערוצים באיכות בינונית-נמוכה",
                f"2. <b>ניטור דפוסים:</b> הגדר התראות לערוצים התואמים דפוסי סיכון גבוה",
                f"3. <b>אכיפת סטנדרטים:</b> דרוש ציון איכות מינימלי של 4.0 לערוצים חדשים"
            ]
        else:
            short_term = [
                f"1. <b>Quality Improvement:</b> Work with {medium_low_count} medium-low quality channels",
                f"2. <b>Pattern Monitoring:</b> Set up alerts for channels matching high-risk patterns",
                f"3. <b>Enforce Standards:</b> Require minimum quality score of 4.0 for new channels"
            ]
        
        for action in short_term:
            story.append(Paragraph(action, styles['CustomNormal']))
        
        story.append(Spacer(1, 0.2 * inch))
        
        # Long-term strategy
        story.append(Paragraph(f"<b>{self.t('long_term', lang)}</b>", styles['SubsectionHeading']))
        
        if lang == 'he':
            long_term = [
                "1. <b>שיפור מודל:</b> אמן מחדש מודלים מדי חודש עם נתונים חדשים",
                "2. <b>אופטימיזציית תהליך:</b> אוטומציה של חסימת ערוצים עבור ציונים < 2.0",
                "3. <b>לוח מחוונים:</b> צור לוח מחוונים בזמן אמת לניטור מתמשך",
                "4. <b>אינטגרציית ML:</b> הטמע ניקוד בזמן אמת לערוצים חדשים"
            ]
        else:
            long_term = [
                "1. <b>Model Enhancement:</b> Retrain models monthly with new data",
                "2. <b>Process Optimization:</b> Automate channel blocking for scores < 2.0",
                "3. <b>Dashboard:</b> Create real-time dashboard for ongoing monitoring",
                "4. <b>ML Integration:</b> Implement real-time scoring for new channels"
            ]
        
        for action in long_term:
            story.append(Paragraph(action, styles['CustomNormal']))
    
    def generate_comprehensive_report(self, quality_df: pd.DataFrame, 
                                    anomaly_df: pd.DataFrame,
                                    final_results: Dict,
                                    pipeline_results: Dict) -> Tuple[str, str]:
        """
        Generate comprehensive PDF reports in both English and Hebrew.
        
        Returns:
            Tuple of (english_pdf_path, hebrew_pdf_path)
        """
        reports = {}
        
        for lang in ['en', 'he']:
            try:
                logger.info(f"Generating {lang.upper()} PDF report...")
                
                # Create document
                if lang == 'he':
                    filename = f"fraud_detection_report_hebrew_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                else:
                    filename = f"fraud_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                filepath = os.path.join(self.output_dir, filename)
                logger.debug(f"Creating PDF at: {filepath}")
                
                doc = SimpleDocTemplate(
                    filepath,
                    pagesize=A4,
                    rightMargin=72,
                    leftMargin=72,
                    topMargin=72,
                    bottomMargin=18,
                )
                
                # Create styles
                styles = self.create_styles(lang)
                
                # Build story
                story = []
                
                # Title page
                story.append(Spacer(1, 2 * inch))
                story.append(Paragraph(self.t('title', lang), styles['CustomTitle']))
                story.append(Paragraph(self.t('subtitle', lang), styles['Subtitle']))
                story.append(Spacer(1, 0.5 * inch))
                story.append(Paragraph(f"{self.t('generated', lang)} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                     styles['CustomNormal']))
                story.append(PageBreak())
                
                # Table of contents
                story.append(Paragraph(self.t('toc', lang), styles['SectionHeading']))
                toc_data = [
                    [self.t('executive_summary', lang), '3'],
                    [self.t('quality_analysis', lang), '4'],
                    [self.t('risk_analysis', lang), '6'],
                    [self.t('anomaly_detection', lang), '8'],
                    [self.t('traffic_similarity', lang), '10'],
                    [self.t('model_performance', lang), '12'],
                    [self.t('recommendations', lang), '13']
                ]
                
                toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
                toc_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT' if lang != 'he' else 'RIGHT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 0, colors.white),
                ]))
                
                story.append(toc_table)
                story.append(PageBreak())
                
                # Executive Summary
                self.create_executive_summary(quality_df, anomaly_df, pipeline_results, lang, story, styles)
                
                # Quality Analysis Section
                story.append(Paragraph(self.t('quality_analysis', lang), styles['SectionHeading']))
                story.append(Spacer(1, 0.2 * inch))
                
                # Quality distribution plot
                story.append(Paragraph(self.t('quality_distribution', lang), styles['SubsectionHeading']))
                try:
                    quality_dist_plot = self.create_quality_distribution_plot(quality_df, lang)
                    if quality_dist_plot and os.path.exists(quality_dist_plot):
                        story.append(Image(quality_dist_plot, width=5*inch, height=3*inch))
                        self.add_plot_description(story, 'quality_distribution', lang, styles)
                    else:
                        logger.warning(f"Quality distribution plot not created for {lang}")
                        story.append(Paragraph(f"[Quality distribution plot unavailable for {lang}]", styles['CustomNormal']))
                except Exception as e:
                    logger.error(f"Error creating quality distribution plot for {lang}: {e}")
                    story.append(Paragraph(f"[Error creating quality distribution plot: {str(e)}]", styles['CustomNormal']))
                
                # Quality vs Volume scatter
                story.append(Paragraph(self.t('quality_by_volume', lang), styles['SubsectionHeading']))
                try:
                    quality_volume_plot = self.create_quality_volume_scatter(quality_df, lang)
                    if quality_volume_plot and os.path.exists(quality_volume_plot):
                        story.append(Image(quality_volume_plot, width=5*inch, height=3*inch))
                        self.add_plot_description(story, 'quality_by_volume', lang, styles)
                    else:
                        logger.warning(f"Quality volume plot not created for {lang}")
                        story.append(Paragraph(f"[Quality volume plot unavailable for {lang}]", styles['CustomNormal']))
                except Exception as e:
                    logger.error(f"Error creating quality volume plot for {lang}: {e}")
                    story.append(Paragraph(f"[Error creating quality volume plot: {str(e)}]", styles['CustomNormal']))
                
                # Top and bottom channels tables
                story.append(Paragraph(self.t('top_channels', lang), styles['SubsectionHeading']))
                top_channels = quality_df.nlargest(5, 'quality_score')[['channelId', 'quality_score', 'bot_rate', 'volume']]
                top_data = [[self.t('channel_id', lang), self.t('quality_score', lang), 
                           self.t('bot_rate', lang), self.t('volume', lang)]]
                for _, row in top_channels.iterrows():
                    top_data.append([row['channelId'][:10] + '...', f"{row['quality_score']:.2f}", 
                                   f"{row['bot_rate']*100:.1f}%", f"{row['volume']:,}"])
                
                top_table = Table(top_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                top_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                ]))
                story.append(top_table)
                story.append(Spacer(1, 0.3 * inch))
                
                story.append(PageBreak())
                
                # Risk Analysis Section
                story.append(Paragraph(self.t('risk_analysis', lang), styles['SectionHeading']))
                story.append(Spacer(1, 0.2 * inch))
                
                # Bot rate boxplot
                story.append(Paragraph(self.t('bot_rate_analysis', lang), styles['SubsectionHeading']))
                try:
                    bot_rate_plot = self.create_bot_rate_boxplot(quality_df, lang)
                    if bot_rate_plot and os.path.exists(bot_rate_plot):
                        story.append(Image(bot_rate_plot, width=5*inch, height=3*inch))
                        self.add_plot_description(story, 'bot_rate_analysis', lang, styles)
                    else:
                        logger.warning(f"Bot rate plot not created for {lang}")
                        story.append(Paragraph(f"[Bot rate plot unavailable for {lang}]", styles['CustomNormal']))
                except Exception as e:
                    logger.error(f"Error creating bot rate plot for {lang}: {e}")
                    story.append(Paragraph(f"[Error creating bot rate plot: {str(e)}]", styles['CustomNormal']))
                
                # Risk matrix
                story.append(Paragraph(self.t('risk_matrix', lang), styles['SubsectionHeading']))
                try:
                    risk_matrix_plot = self.create_risk_matrix(quality_df, lang)
                    if risk_matrix_plot and os.path.exists(risk_matrix_plot):
                        story.append(Image(risk_matrix_plot, width=5*inch, height=4*inch))
                        self.add_plot_description(story, 'risk_matrix', lang, styles)
                    else:
                        logger.warning(f"Risk matrix plot not created for {lang}")
                        story.append(Paragraph(f"[Risk matrix plot unavailable for {lang}]", styles['CustomNormal']))
                except Exception as e:
                    logger.error(f"Error creating risk matrix plot for {lang}: {e}")
                    story.append(Paragraph(f"[Error creating risk matrix plot: {str(e)}]", styles['CustomNormal']))
                
                story.append(PageBreak())
                
                # Anomaly Detection Section
                story.append(Paragraph(self.t('anomaly_detection', lang), styles['SectionHeading']))
                story.append(Spacer(1, 0.2 * inch))
                
                if not anomaly_df.empty:
                    # Anomaly heatmap
                    story.append(Paragraph(self.t('anomaly_heatmap', lang), styles['SubsectionHeading']))
                    anomaly_heatmap = self.create_anomaly_heatmap(anomaly_df, lang)
                    if anomaly_heatmap:
                        story.append(Image(anomaly_heatmap, width=6*inch, height=4*inch))
                        self.add_plot_description(story, 'anomaly_heatmap', lang, styles)
                    
                    # Anomaly distribution
                    story.append(Paragraph(self.t('anomaly_distribution', lang), styles['SubsectionHeading']))
                    anomaly_dist = self.create_anomaly_distribution(anomaly_df, lang)
                    if anomaly_dist:
                        story.append(Image(anomaly_dist, width=5*inch, height=3*inch))
                        self.add_plot_description(story, 'anomaly_distribution', lang, styles)
                    
                    # Fraud type pie chart
                    fraud_pie = self.create_fraud_type_pie(anomaly_df, lang)
                    if fraud_pie:
                        story.append(Image(fraud_pie, width=5*inch, height=4*inch))
                        self.add_plot_description(story, 'fraud_distribution', lang, styles)
                
                story.append(PageBreak())
                
                # Traffic Similarity Section
                story.append(Paragraph(self.t('traffic_similarity', lang), styles['SectionHeading']))
                story.append(Spacer(1, 0.2 * inch))
                
                # Cluster visualization
                story.append(Paragraph(self.t('cluster_visualization', lang), styles['SubsectionHeading']))
                cluster_viz = self.create_cluster_visualization(quality_df, lang)
                if cluster_viz:
                    story.append(Image(cluster_viz, width=5*inch, height=4*inch))
                    self.add_plot_description(story, 'cluster_visualization', lang, styles)
                
                # Cluster quality
                story.append(Paragraph(self.t('cluster_quality', lang), styles['SubsectionHeading']))
                cluster_quality = self.create_cluster_quality_chart(quality_df, lang)
                if cluster_quality:
                    story.append(Image(cluster_quality, width=5*inch, height=3*inch))
                    self.add_plot_description(story, 'cluster_quality', lang, styles)
                
                story.append(PageBreak())
                
                # Additional visualizations
                # Quality trend
                quality_trend = self.create_quality_trend_plot(quality_df, lang)
                if quality_trend:
                    story.append(Paragraph(self.t('quality_analysis', lang), styles['SubsectionHeading']))
                    story.append(Image(quality_trend, width=6*inch, height=3*inch))
                    self.add_plot_description(story, 'quality_score_time', lang, styles)
                
                story.append(PageBreak())
                
                # Recommendations
                self.create_recommendations_section(quality_df, anomaly_df, lang, story, styles)
                
                # Build PDF
                doc.build(story)
                
                logger.info(f"{lang.upper()} PDF report generated: {filepath}")
                reports[lang] = filepath
                
            except Exception as e:
                logger.error(f"Failed to generate {lang.upper()} PDF report: {e}")
                reports[lang] = None
        
        return reports['en'], reports['he']