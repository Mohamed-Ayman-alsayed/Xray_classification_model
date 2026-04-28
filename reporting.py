"""
Report Generation Module for AI Health System
===========================================

Generates automated PDF and HTML lab reports based on model predictions.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

try:
    import jinja2
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Warning: Jinja2 not available. HTML reports will use basic templates.")

class LabReportGenerator:
    """Generates medical lab reports in PDF and HTML formats"""
    
    def __init__(self, template_dir: str = None):
        self.template_dir = template_dir
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Setup Jinja2 for HTML templates if available
        if JINJA2_AVAILABLE and template_dir and os.path.exists(template_dir):
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )
        else:
            self.jinja_env = None
            if template_dir and not os.path.exists(template_dir):
                print(f"Warning: Template directory not found: {template_dir}")
                print("HTML reports will use built-in templates.")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='ResultText',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        ))
    
    def generate_pdf_report(self, prediction_data: Dict[str, Any], 
                           output_path: str = None) -> str:
        """Generate PDF lab report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"lab_report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph("AI Health System - Chest X-Ray Analysis Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Report metadata
        metadata_data = [
            ['Report Date:', datetime.now().strftime("%B %d, %Y")],
            ['Report Time:', datetime.now().strftime("%I:%M %p")],
            ['Patient ID:', prediction_data.get('patient_id', 'N/A')],
            ['Study Date:', prediction_data.get('study_date', 'N/A')],
            ['Analysis Type:', 'Chest X-Ray Classification']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Analysis Results
        results_header = Paragraph("Analysis Results", self.styles['SectionHeader'])
        story.append(results_header)
        
        # Diagnosis
        diagnosis = prediction_data.get('diagnosis', 'Unknown')
        confidence = prediction_data.get('confidence', 0.0)
        
        diagnosis_text = f"Diagnosis: {diagnosis}"
        confidence_text = f"Confidence: {confidence:.1%}"
        
        story.append(Paragraph(diagnosis_text, self.styles['ResultText']))
        story.append(Paragraph(confidence_text, self.styles['ResultText']))
        story.append(Spacer(1, 15))
        
        # Findings
        findings_header = Paragraph("Clinical Findings", self.styles['SectionHeader'])
        story.append(findings_header)
        
        findings = self._get_findings_text(diagnosis, confidence)
        story.append(Paragraph(findings, self.styles['NormalText']))
        story.append(Spacer(1, 15))
        
        # Recommendations
        recommendations_header = Paragraph("Recommendations", self.styles['SectionHeader'])
        story.append(recommendations_header)
        
        recommendations = self._get_recommendations_text(diagnosis, confidence)
        story.append(Paragraph(recommendations, self.styles['NormalText']))
        story.append(Spacer(1, 20))
        
        # Technical Details
        tech_header = Paragraph("Technical Details", self.styles['SectionHeader'])
        story.append(tech_header)
        
        tech_data = [
            ['Model Used:', prediction_data.get('model_name', 'Custom CNN')],
            ['Input Image Size:', f"{prediction_data.get('image_size', 'Unknown')}"],
            ['Processing Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Analysis Duration:', f"{prediction_data.get('processing_time', 'Unknown')} seconds"]
        ]
        
        tech_table = Table(tech_data, colWidths=[2*inch, 4*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(tech_table)
        story.append(Spacer(1, 20))
        
        # Disclaimer
        disclaimer = Paragraph(
            "This report is generated by an AI system and should be reviewed by qualified medical professionals. "
            "The findings are based on automated image analysis and may require clinical correlation.",
            self.styles['NormalText']
        )
        disclaimer.alignment = TA_CENTER
        story.append(disclaimer)
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF report generated: {output_path}")
        return output_path
    
    def generate_html_report(self, prediction_data: Dict[str, Any], 
                            output_path: str = None) -> str:
        """Generate HTML lab report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"lab_report_{timestamp}.html"
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Health System - Lab Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            margin: 0;
            font-size: 28px;
        }
        .metadata {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        .metadata table {
            width: 100%;
            border-collapse: collapse;
        }
        .metadata td {
            padding: 8px;
            border-bottom: 1px solid #bdc3c7;
        }
        .metadata td:first-child {
            font-weight: bold;
            color: #2c3e50;
            width: 30%;
        }
        .section {
            margin-bottom: 25px;
        }
        .section h2 {
            color: #2c3e50;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 15px;
        }
        .result {
            background-color: #d5f4e6;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #27ae60;
        }
        .result strong {
            color: #27ae60;
        }
        .disclaimer {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin-top: 30px;
            text-align: center;
            color: #856404;
        }
        .tech-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .tech-table th, .tech-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .tech-table th {
            background-color: #34495e;
            color: white;
        }
        .tech-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Health System - Chest X-Ray Analysis Report</h1>
        </div>
        
        <div class="metadata">
            <table>
                <tr><td>Report Date:</td><td>{{ report_date }}</td></tr>
                <tr><td>Report Time:</td><td>{{ report_time }}</td></tr>
                <tr><td>Patient ID:</td><td>{{ patient_id }}</td></tr>
                <tr><td>Study Date:</td><td>{{ study_date }}</td></tr>
                <tr><td>Analysis Type:</td><td>Chest X-Ray Classification</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Analysis Results</h2>
            <div class="result">
                <strong>Diagnosis:</strong> {{ diagnosis }}<br>
                <strong>Confidence:</strong> {{ confidence }}
            </div>
        </div>
        
        <div class="section">
            <h2>Clinical Findings</h2>
            <p>{{ findings }}</p>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <p>{{ recommendations }}</p>
        </div>
        
        <div class="section">
            <h2>Technical Details</h2>
            <table class="tech-table">
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr><td>Model Used</td><td>{{ model_name }}</td></tr>
                <tr><td>Input Image Size</td><td>{{ image_size }}</td></tr>
                <tr><td>Processing Date</td><td>{{ processing_date }}</td></tr>
                <tr><td>Analysis Duration</td><td>{{ processing_time }} seconds</td></tr>
            </table>
        </div>
        
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This report is generated by an AI system and should be reviewed by qualified medical professionals. 
            The findings are based on automated image analysis and may require clinical correlation.
        </div>
    </div>
</body>
</html>
        """
        
        # Render template
        if self.jinja_env:
            template = self.jinja_env.from_string(html_template)
        else:
            # Fallback to basic string replacement
            template = html_template
        
        # Prepare data for template
        template_data = {
            'report_date': datetime.now().strftime("%B %d, %Y"),
            'report_time': datetime.now().strftime("%I:%M %p"),
            'patient_id': prediction_data.get('patient_id', 'N/A'),
            'study_date': prediction_data.get('study_date', 'N/A'),
            'diagnosis': prediction_data.get('diagnosis', 'Unknown'),
            'confidence': f"{prediction_data.get('confidence', 0.0):.1%}",
            'findings': self._get_findings_text(
                prediction_data.get('diagnosis', 'Unknown'),
                prediction_data.get('confidence', 0.0)
            ),
            'recommendations': self._get_recommendations_text(
                prediction_data.get('diagnosis', 'Unknown'),
                prediction_data.get('confidence', 0.0)
            ),
            'model_name': prediction_data.get('model_name', 'Custom CNN'),
            'image_size': prediction_data.get('image_size', 'Unknown'),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'processing_time': prediction_data.get('processing_time', 'Unknown')
        }
        
        # Generate HTML content
        if self.jinja_env:
            html_content = template.render(**template_data)
        else:
            # Basic string replacement fallback
            html_content = html_template
            for key, value in template_data.items():
                html_content = html_content.replace(f"{{{{ {key} }}}}", str(value))
        
        # Save HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report generated: {output_path}")
        return output_path
    
    def _get_findings_text(self, diagnosis: str, confidence: float) -> str:
        """Generate clinical findings text based on diagnosis and confidence"""
        if diagnosis.lower() == 'pneumonia':
            if confidence >= 0.9:
                return "High-confidence detection of pneumonia with clear radiographic evidence. The AI system has identified characteristic patterns consistent with pulmonary infection."
            elif confidence >= 0.7:
                return "Moderate-confidence detection of pneumonia. Radiographic findings suggest pulmonary infection, though clinical correlation is recommended."
            else:
                return "Low-confidence detection of pneumonia. Some radiographic features may suggest infection, but further clinical evaluation is strongly recommended."
        else:
            if confidence >= 0.9:
                return "High-confidence normal chest X-ray. No significant radiographic abnormalities detected by the AI system."
            elif confidence >= 0.7:
                return "Moderate-confidence normal chest X-ray. No obvious abnormalities detected, though subtle findings may require clinical review."
            else:
                return "Low-confidence normal chest X-ray. Some radiographic features may warrant further clinical review."
    
    def _get_recommendations_text(self, diagnosis: str, confidence: float) -> str:
        """Generate clinical recommendations based on diagnosis and confidence"""
        if diagnosis.lower() == 'pneumonia':
            if confidence >= 0.9:
                return "Immediate clinical correlation recommended. Consider appropriate antibiotic therapy based on clinical presentation and local resistance patterns. Follow-up imaging may be indicated to monitor treatment response."
            elif confidence >= 0.7:
                return "Clinical correlation strongly recommended. Consider empiric antibiotic therapy while awaiting culture results. Close clinical monitoring and follow-up imaging advised."
            else:
                return "Clinical correlation essential. Consider additional diagnostic testing (e.g., sputum culture, blood tests) to confirm diagnosis. Treatment decisions should be based on clinical assessment."
        else:
            if confidence >= 0.9:
                return "Routine clinical follow-up as indicated by patient's clinical presentation. No immediate intervention required based on radiographic findings."
            elif confidence >= 0.7:
                return "Clinical correlation recommended to ensure no subtle findings were missed. Follow-up as clinically indicated."
            else:
                return "Clinical correlation advised. Consider additional imaging or clinical assessment if symptoms persist or worsen."

def create_report_generator(template_dir: str = None) -> LabReportGenerator:
    """Factory function to create report generator"""
    return LabReportGenerator(template_dir=template_dir)
