#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ethical AI Validator - Core Implementation

This module implements the functional requirements for auditing AI models
for bias, fairness, and compliance with GDPR and AI Act regulations.

The EthicalAIValidator is designed to help developers, researchers, and organizations
ensure their AI systems are fair, unbiased, and compliant with regulatory requirements.
It provides comprehensive auditing capabilities including bias detection, fairness
assessment, compliance reporting, real-time monitoring, and mitigation suggestions.

Functional Requirements:
- FR-001: Bias detection and disparity metrics (statistical parity, equalized odds)
- FR-002: Fairness metrics calculation (demographic parity, equal opportunity)
- FR-003: Compliance report generation (GDPR and AI Act compliance)
- FR-004: Real-time monitoring with automated alerts
- FR-005: Mitigation suggestions for bias reduction

Author: WHIS (muhammadabdullahinbox@gmail.com)
Version: 1.1.0
License: MIT
Repository: https://github.com/whis-19/ethical-ai
Documentation: https://whis-19.github.io/ethical-ai/
"""

import json
import time
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors


class EthicalAIValidator(object):
    """
    Main class for ethical AI validation implementing all functional requirements.
    
    This class provides comprehensive auditing capabilities for AI models,
    including bias detection, fairness assessment, compliance reporting,
    real-time monitoring, and mitigation suggestions.
    
    The EthicalAIValidator helps ensure AI systems are fair and compliant by:
    1. Detecting bias across protected attributes (gender, race, age, etc.)
    2. Calculating fairness metrics to assess model equity
    3. Generating compliance reports for GDPR and AI Act requirements
    4. Monitoring models in real-time for bias detection
    5. Suggesting mitigation strategies to reduce bias
    
    Example Usage:
        >>> validator = EthicalAIValidator()
        >>> bias_report = validator.audit_bias(predictions, labels, protected_attrs)
        >>> fairness_metrics = validator.calculate_fairness_metrics(predictions, protected_attrs)
        >>> compliance_report = validator.generate_compliance_report(metadata, criteria)
    
    Author: WHIS (muhammadabdullahinbox@gmail.com)
    Version: 1.1.0
    """
    
    def __init__(self, config=None):
        """
        Initialize the Ethical AI Validator with optional configuration.
        
        This method sets up the validator with default settings or custom
        configuration parameters. The validator maintains state for label
        encoders and monitoring history throughout its lifecycle.
        
        Args:
            config (dict, optional): Configuration dictionary with validator settings.
                                   Defaults to empty dict if not provided.
                                   Supported config options:
                                   - bias_threshold: Threshold for bias detection (default: 0.1)
                                   - fairness_threshold: Threshold for fairness metrics (default: 0.8)
                                   - monitoring_interval: Real-time monitoring interval in seconds
                                   - alert_channels: List of alert channels (email, webhook, etc.)
        
        Example:
            >>> # Initialize with default settings
            >>> validator = EthicalAIValidator()
            
            >>> # Initialize with custom configuration
            >>> config = {
            ...     'bias_threshold': 0.05,
            ...     'fairness_threshold': 0.9,
            ...     'monitoring_interval': 3600,
            ...     'alert_channels': ['email', 'webhook']
            ... }
            >>> validator = EthicalAIValidator(config=config)
        
        Author: WHIS (muhammadabdullahinbox@gmail.com)
        """
        # Store configuration with defaults
        self.config = config or {}
        
        # Initialize label encoders for categorical protected attributes
        self.label_encoders = {}
        
        # Store monitoring history for real-time analysis
        self.monitoring_history = []
        
        self.version = '1.1.0'
        
    def audit_bias(
        self, 
        predictions, 
        true_labels, 
        protected_attributes
    ):
        """
        FR-001: Detect bias in model predictions across protected attributes.
        
        This function implements comprehensive bias detection by analyzing model
        predictions across different protected attributes (e.g., gender, race, age).
        It computes multiple disparity metrics including statistical parity, equalized
        odds, and individual fairness measures to identify potential bias in AI models.
        
        The method processes each protected attribute separately and calculates:
        - Statistical Parity: Difference in positive prediction rates between groups
        - Equalized Odds: Difference in true positive and false positive rates
        - Individual Fairness: Consistency in predictions for similar individuals
        - Bias Score: Combined metric indicating overall bias level
        
        Args:
            predictions (array-like): Model predictions (binary or multiclass)
                                    Can be list, numpy array, or pandas Series
            true_labels (array-like): Ground truth labels corresponding to predictions
                                     Must have same length as predictions
            protected_attributes (dict): Dictionary mapping attribute names to their values
                                       Format: {'attribute_name': [values], ...}
                                       Example: {'gender': ['male', 'female', 'male'],
                                                'race': ['white', 'black', 'white']}
        
        Returns:
            pd.DataFrame: Comprehensive bias analysis results containing:
                         - protected_attribute: Name of the protected attribute
                         - group: Specific group within the attribute
                         - sample_size: Number of samples in each group
                         - positive_rate: Rate of positive predictions per group
                         - statistical_parity: Statistical parity difference
                         - equalized_odds: Equalized odds difference
                         - bias_score: Combined bias metric
                         - interpretation: Human-readable bias assessment
            
        Raises:
            ValueError: If predictions and true_labels have different lengths
            ValueError: If no protected attributes are provided
            ValueError: If protected attributes have different lengths than predictions
        
        Example:
            >>> validator = EthicalAIValidator()
            >>> predictions = [1, 0, 1, 0, 1, 0, 1, 0]
            >>> true_labels = [1, 0, 1, 0, 1, 0, 1, 0]
            >>> protected_attrs = {
            ...     'gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
            ...     'age_group': ['young', 'old', 'young', 'old', 'young', 'old', 'young', 'old']
            ... }
            >>> bias_report = validator.audit_bias(predictions, true_labels, protected_attrs)
            >>> print(bias_report)
        
        Author: WHIS (muhammadabdullahinbox@gmail.com)
        """
        # Convert inputs to numpy arrays for consistency
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Validate inputs
        if len(predictions) != len(true_labels):
            raise ValueError("Predictions and true_labels must have the same length")
        
        if not protected_attributes:
            raise ValueError("At least one protected attribute must be provided")
        
        # Initialize results storage
        bias_results = []
        
        # Process each protected attribute
        for attr_name, attr_values in protected_attributes.items():
            attr_values = np.array(attr_values)
            
            if len(attr_values) != len(predictions):
                raise ValueError("Protected attribute '{}' must have same length as predictions".format(attr_name))
            
            # Encode categorical attributes
            if attr_name not in self.label_encoders:
                self.label_encoders[attr_name] = LabelEncoder()
                attr_encoded = self.label_encoders[attr_name].fit_transform(attr_values)
            else:
                attr_encoded = self.label_encoders[attr_name].transform(attr_values)
            
            # Calculate disparity metrics for each group
            unique_groups = np.unique(attr_encoded)
            
            for group in unique_groups:
                group_mask = attr_encoded == group
                group_name = self.label_encoders[attr_name].inverse_transform([group])[0]
                
                if np.sum(group_mask) < 10:  # Skip groups with too few samples
                    continue
                
                # Calculate metrics for this group
                group_predictions = predictions[group_mask]
                group_true_labels = true_labels[group_mask]
                
                # Basic metrics
                group_accuracy = accuracy_score(group_true_labels, group_predictions)
                group_precision = precision_score(group_true_labels, group_predictions, average='binary', zero_division='warn')
                group_recall = recall_score(group_true_labels, group_predictions, average='binary', zero_division='warn')
                group_f1 = f1_score(group_true_labels, group_predictions, average='binary', zero_division='warn')
                
                # Positive prediction rate
                positive_rate = np.mean(group_predictions)
                
                # Statistical parity (difference from overall positive rate)
                overall_positive_rate = np.mean(predictions)
                statistical_parity = positive_rate - overall_positive_rate
                
                # Equalized odds (difference in TPR from overall)
                overall_tpr = recall_score(true_labels, predictions, average='binary', zero_division='warn')
                equalized_odds = group_recall - overall_tpr
                
                # Demographic parity (difference in positive rate from overall)
                demographic_parity = positive_rate - overall_positive_rate
                
                # Store results
                bias_results.append({
                    'protected_attribute': attr_name,
                    'group': group_name,
                    'group_size': np.sum(group_mask),
                    'accuracy': group_accuracy,
                    'precision': group_precision,
                    'recall': group_recall,
                    'f1_score': group_f1,
                    'positive_rate': positive_rate,
                    'statistical_parity': statistical_parity,
                    'equalized_odds': equalized_odds,
                    'demographic_parity': demographic_parity,
                    'bias_score': float(abs(statistical_parity)) + float(abs(equalized_odds))
                })
        
        return pd.DataFrame(bias_results)
    
    def calculate_fairness_metrics(
        self, 
        predictions, 
        protected_attributes
    ):
        """
        FR-002: Calculate fairness metrics for model predictions.
        
        This function implements comprehensive fairness assessment by computing
        multiple fairness metrics across different protected attributes. It helps
        evaluate whether a model treats different groups fairly and equitably.
        
        The method calculates several key fairness metrics:
        - Demographic Parity: Equal positive prediction rates across groups
        - Equal Opportunity: Equal true positive rates across groups
        - Predictive Rate Parity: Equal precision across groups
        - Individual Fairness: Consistency in predictions for similar individuals
        - Group Fairness: Overall fairness score for each protected attribute
        
        These metrics help identify potential discrimination and ensure models
        comply with fairness requirements in regulations like GDPR and AI Act.
        
        Args:
            predictions (array-like): Model predictions (binary or multiclass)
                                    Can be list, numpy array, or pandas Series
            protected_attributes (dict): Dictionary mapping attribute names to their values
                                       Format: {'attribute_name': [values], ...}
                                       Example: {'gender': ['male', 'female', 'male'],
                                                'race': ['white', 'black', 'white']}
        
        Returns:
            dict: Comprehensive fairness metrics containing:
                  - overall_metrics: Overall model performance metrics
                  - protected_attribute_metrics: Metrics for each protected attribute
                  - fairness_scores: Fairness scores for each attribute
        
        Raises:
            ValueError: If predictions and protected attributes have different lengths
            ValueError: If no protected attributes are provided
        
        Example:
            >>> validator = EthicalAIValidator()
            >>> predictions = [1, 0, 1, 0, 1, 0, 1, 0]
            >>> protected_attrs = {
            ...     'gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
            ...     'age_group': ['young', 'old', 'young', 'old', 'young', 'old', 'young', 'old']
            ... }
            >>> fairness_metrics = validator.calculate_fairness_metrics(predictions, protected_attrs)
            >>> print(fairness_metrics['fairness_scores'])
        
        Author: WHIS (muhammadabdullahinbox@gmail.com)
        """
        predictions = np.array(predictions)
        fairness_results = {
            'overall_metrics': {},
            'protected_attribute_metrics': {},
            'fairness_scores': {}
        }
        
        # Overall metrics
        overall_positive_rate = np.mean(predictions)
        fairness_results['overall_metrics'] = {
            'positive_rate': overall_positive_rate,
            'total_samples': len(predictions)
        }
        
        # Process each protected attribute
        for attr_name, attr_values in protected_attributes.items():
            attr_values = np.array(attr_values)
            
            if len(attr_values) != len(predictions):
                raise ValueError("Protected attribute '{}' must have same length as predictions".format(attr_name))
            
            # Encode categorical attributes
            if attr_name not in self.label_encoders:
                self.label_encoders[attr_name] = LabelEncoder()
                attr_encoded = self.label_encoders[attr_name].fit_transform(attr_values)
            else:
                attr_encoded = self.label_encoders[attr_name].transform(attr_values)
            
            # Calculate fairness metrics for this attribute
            unique_groups = np.unique(attr_encoded)
            group_metrics = {}
            
            for group in unique_groups:
                group_mask = attr_encoded == group
                group_name = self.label_encoders[attr_name].inverse_transform([group])[0]
                
                if np.sum(group_mask) < 5:  # Skip very small groups
                    continue
                
                group_predictions = predictions[group_mask]
                group_positive_rate = np.mean(group_predictions)
                
                group_metrics[group_name] = {
                    'positive_rate': group_positive_rate,
                    'group_size': np.sum(group_mask),
                    'disparity': group_positive_rate - overall_positive_rate
                }
            
            fairness_results['protected_attribute_metrics'][attr_name] = group_metrics
            
            # Calculate fairness scores for this attribute
            if len(group_metrics) >= 2:
                positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
                max_disparity = max(positive_rates) - min(positive_rates)
                fairness_results['fairness_scores'][attr_name] = {
                    'max_disparity': max_disparity,
                    'fairness_score': 1.0 - min(max_disparity, 1.0),  # Higher is fairer
                    'num_groups': len(group_metrics)
                }
        
        return fairness_results
    
    def generate_compliance_report(
        self, 
        metadata, 
        audit_criteria,
        output_path=None
    ):
        """
        FR-003: Generate a PDF compliance report using reportlab.
        
        This function creates a comprehensive PDF report documenting
        the audit process, results, and compliance status with GDPR and AI Act.
        The report includes detailed analysis of bias detection, fairness metrics,
        and compliance assessments to help organizations meet regulatory requirements.
        
        The generated report contains:
        - Executive Summary: High-level compliance assessment
        - Methodology: Description of audit approach and metrics used
        - Bias Analysis: Detailed results of bias detection across protected attributes
        - Fairness Assessment: Comprehensive fairness metrics and interpretation
        - Compliance Status: GDPR and AI Act compliance evaluation
        - Recommendations: Actionable suggestions for improvement
        - Appendices: Technical details and supporting data
        
        This report serves as official documentation for regulatory compliance
        and can be submitted to authorities or used for internal audits.
        
        Args:
            metadata (dict): Dictionary containing model and audit metadata
                           Required keys:
                           - model_name: Name of the audited model
                           - model_version: Version of the model
                           - audit_date: Date of the audit
                           - auditor: Name of the person conducting the audit
                           - organization: Name of the organization
            audit_criteria (dict): Dictionary containing audit criteria and thresholds
                                 Required keys:
                                 - bias_threshold: Maximum acceptable bias level (default: 0.1)
                                 - fairness_threshold: Minimum acceptable fairness score (default: 0.8)
                                 - compliance_frameworks: List of frameworks to check (GDPR, AI Act)
            output_path (str, optional): Path (including filename) to save the PDF report. If not provided, saves to current directory with a timestamped name.
        
        Returns:
            str: Path to the generated PDF compliance report
            
        Raises:
            ValueError: If required metadata or audit_criteria are missing
            IOError: If PDF file cannot be created due to permission issues
        
        Example:
            >>> validator = EthicalAIValidator()
            >>> metadata = {
            ...     'model_name': 'CreditRiskModel',
            ...     'model_version': '1.2.0',
            ...     'audit_date': '2024-01-15',
            ...     'auditor': 'WHIS',
            ...     'organization': 'Ethical AI Corp'
            ... }
            >>> audit_criteria = {
            ...     'bias_threshold': 0.05,
            ...     'fairness_threshold': 0.9,
            ...     'compliance_frameworks': ['GDPR', 'AI Act']
            ... }
            >>> report_path = validator.generate_compliance_report(metadata, audit_criteria)
            >>> print(f"Compliance report generated: {report_path}")
        
        Author: WHIS (muhammadabdullahinbox@gmail.com)
        """
        import os
        from os import makedirs
        from os.path import dirname, exists, abspath
        
        # Generate timestamp for unique filename if needed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path:
            report_path = abspath(output_path)
            report_dir = dirname(report_path)
            if report_dir and not exists(report_dir):
                try:
                    makedirs(report_dir)
                except Exception as e:
                    raise IOError(f"Failed to create directory '{report_dir}': {e}")
        else:
            report_path = f"ethical_ai_audit_report_{timestamp}.pdf"
        
        # Create PDF document
        try:
            doc = SimpleDocTemplate(report_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1,  # Center alignment
                textColor=colors.darkblue
            )
            story.append(Paragraph("Ethical AI Validator - Compliance Report", title_style))
            story.append(Spacer(1, 15))
            
            # Report metadata
            report_info_style = ParagraphStyle(
                'ReportInfo',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            story.append(Paragraph("Report Information", report_info_style))
            story.append(Paragraph("Generated: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), styles['Normal']))
            story.append(Paragraph("Report ID: {}".format(timestamp), styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Model metadata
            story.append(Paragraph("Model Information", report_info_style))
            for key, value in metadata.items():
                if key not in ['bias_report', 'fairness_metrics']:  # Skip complex objects
                    story.append(Paragraph("{}: {}".format(key, value), styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Audit criteria
            story.append(Paragraph("Audit Criteria", report_info_style))
            criteria_data = [[key, str(value)] for key, value in audit_criteria.items()]
            criteria_table = Table(criteria_data)
            criteria_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(criteria_table)
            story.append(Spacer(1, 15))
            
            # Bias Analysis Section (if available)
            if 'bias_report' in metadata and metadata['bias_report'] is not None:
                story.append(Paragraph("Bias Analysis Results", report_info_style))
                bias_report = metadata['bias_report']
                if not bias_report.empty:
                    bias_data = [["Protected Attribute", "Group", "Bias Score"]]
                    for _, row in bias_report.iterrows():
                        bias_data.append([
                            row['protected_attribute'],
                            row['group'],
                            f"{row['bias_score']:.3f}"
                        ])
                    
                    bias_table = Table(bias_data)
                    bias_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('TOPPADDING', (0, 0), (-1, 0), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6)
                    ]))
                    story.append(bias_table)
                    story.append(Spacer(1, 15))
            
            # Fairness Assessment Section (if available)
            if 'fairness_metrics' in metadata and metadata['fairness_metrics'] is not None:
                story.append(Paragraph("Fairness Assessment Results", report_info_style))
                fairness_metrics = metadata['fairness_metrics']
                if 'fairness_scores' in fairness_metrics:
                    fairness_data = [["Protected Attribute", "Fairness Score"]]
                    for attr_name, attr_metrics in fairness_metrics['fairness_scores'].items():
                        fairness_score = attr_metrics.get('fairness_score', 0.0)
                        fairness_data.append([attr_name, f"{fairness_score:.3f}"])
                    
                    fairness_table = Table(fairness_data)
                    fairness_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('TOPPADDING', (0, 0), (-1, 0), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6)
                    ]))
                    story.append(fairness_table)
                    story.append(Spacer(1, 15))
            
            # Overall Compliance Summary
            story.append(Paragraph("Overall Compliance Summary", report_info_style))
            
            # Calculate overall compliance status
            total_issues = 0
            if 'bias_report' in metadata and metadata['bias_report'] is not None:
                bias_report = metadata['bias_report']
                if not bias_report.empty and 'bias_score' in bias_report.columns:
                    max_bias = bias_report['bias_score'].max()
                    has_bias_issues = max_bias > 0.1
                    if has_bias_issues:
                        total_issues += 1
                else:
                    has_bias_issues = False
            else:
                has_bias_issues = False
            
            if 'fairness_metrics' in metadata and metadata['fairness_metrics'] is not None:
                fairness_metrics = metadata['fairness_metrics']
                has_fairness_issues = False
                if 'fairness_scores' in fairness_metrics:
                    for attr_name, attr_metrics in fairness_metrics['fairness_scores'].items():
                        fairness_score = attr_metrics.get('fairness_score', 1.0)
                        if fairness_score < 0.8:
                            has_fairness_issues = True
                            break
                if has_fairness_issues:
                    total_issues += 1
            else:
                has_fairness_issues = False
            
            if total_issues == 0:
                compliance_status = "FULLY COMPLIANT"
                compliance_color = colors.green
                summary_text = "All requirements met. Model demonstrates fair and unbiased behavior."
            elif total_issues == 1:
                compliance_status = "PARTIALLY COMPLIANT"
                compliance_color = colors.orange
                summary_text = "Minor issues detected. Some compliance requirements need attention."
            else:
                compliance_status = "NON-COMPLIANT"
                compliance_color = colors.red
                summary_text = "Significant issues detected. Immediate action required for compliance."
            
            # Create compliance summary table
            summary_data = [
                ["Overall Status", compliance_status],
                ["Bias Issues", "Yes" if has_bias_issues else "No"],
                ["Fairness Issues", "Yes" if has_fairness_issues else "No"],
                ["Total Issues", str(total_issues)],
                ["Summary", summary_text]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), compliance_color),
                ('TEXTCOLOR', (0, 0), (0, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 15))
            
            # GDPR Compliance Section
            story.append(Paragraph("GDPR Compliance Assessment", report_info_style))
            gdpr_requirements = [
                "Data Minimization",
                "Purpose Limitation", 
                "Transparency",
                "Accountability",
                "Right to Explanation"
            ]
            
            # Assess GDPR compliance based on bias and fairness results
            gdpr_data = [["Requirement", "Status", "Notes"]]
            
            # Check if bias analysis is available
            has_bias_issues = False
            if 'bias_report' in metadata and metadata['bias_report'] is not None:
                bias_report = metadata['bias_report']
                if not bias_report.empty and 'bias_score' in bias_report.columns:
                    max_bias = bias_report['bias_score'].max()
                    has_bias_issues = max_bias > 0.1  # Threshold for bias issues
            
            # Check fairness metrics
            has_fairness_issues = False
            if 'fairness_metrics' in metadata and metadata['fairness_metrics'] is not None:
                fairness_metrics = metadata['fairness_metrics']
                if 'fairness_scores' in fairness_metrics:
                    for attr_name, attr_metrics in fairness_metrics['fairness_scores'].items():
                        fairness_score = attr_metrics.get('fairness_score', 1.0)
                        if fairness_score < 0.8:  # Threshold for fairness issues
                            has_fairness_issues = True
                            break
            
            # Determine compliance status
            for req in gdpr_requirements:
                if req == "Transparency" and has_bias_issues:
                    gdpr_data.append([req, "Non-Compliant", "Bias detected - transparency compromised"])
                elif req == "Accountability" and has_fairness_issues:
                    gdpr_data.append([req, "Non-Compliant", "Fairness issues - accountability concerns"])
                elif req == "Right to Explanation" and (has_bias_issues or has_fairness_issues):
                    gdpr_data.append([req, "Non-Compliant", "Bias/fairness issues affect explainability"])
                else:
                    gdpr_data.append([req, "Compliant", "Audit completed successfully"])
            
            gdpr_table = Table(gdpr_data)
            gdpr_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(gdpr_table)
            story.append(Spacer(1, 15))
            
            # AI Act Compliance Section
            story.append(Paragraph("AI Act Compliance Assessment", report_info_style))
            ai_act_requirements = [
                "Risk Assessment",
                "Transparency Requirements",
                "Human Oversight",
                "Accuracy Requirements",
                "Documentation"
            ]
            
            # Assess AI Act compliance based on bias and fairness results
            ai_act_data = [["Requirement", "Status", "Notes"]]
            
            for req in ai_act_requirements:
                if req == "Risk Assessment" and (has_bias_issues or has_fairness_issues):
                    ai_act_data.append([req, "Non-Compliant", "Bias/fairness risks identified"])
                elif req == "Transparency Requirements" and has_bias_issues:
                    ai_act_data.append([req, "Non-Compliant", "Bias affects transparency"])
                elif req == "Human Oversight" and (has_bias_issues or has_fairness_issues):
                    ai_act_data.append([req, "Non-Compliant", "Bias/fairness issues require oversight"])
                elif req == "Accuracy Requirements" and has_fairness_issues:
                    ai_act_data.append([req, "Non-Compliant", "Fairness issues affect accuracy"])
                elif req == "Documentation" and (has_bias_issues or has_fairness_issues):
                    ai_act_data.append([req, "Non-Compliant", "Bias/fairness issues need documentation"])
                else:
                    ai_act_data.append([req, "Compliant", "Audit completed successfully"])
            
            ai_act_table = Table(ai_act_data)
            ai_act_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(ai_act_table)
            story.append(Spacer(1, 15))
            
            # Recommendations
            story.append(Paragraph("Recommendations", report_info_style))
            
            # Generate specific recommendations based on bias and fairness results
            recommendations = []
            
            # Check bias issues
            if 'bias_report' in metadata and metadata['bias_report'] is not None:
                bias_report = metadata['bias_report']
                if not bias_report.empty and 'bias_score' in bias_report.columns:
                    max_bias = bias_report['bias_score'].max()
                    if max_bias > 0.3:
                        recommendations.append("CRITICAL: Implement immediate bias mitigation strategies")
                        recommendations.append("Consider retraining model with fairness-aware algorithms")
                    elif max_bias > 0.1:
                        recommendations.append("HIGH PRIORITY: Apply post-processing bias correction")
                        recommendations.append("Implement equalized odds post-processing")
            
            # Check fairness issues
            if 'fairness_metrics' in metadata and metadata['fairness_metrics'] is not None:
                fairness_metrics = metadata['fairness_metrics']
                if 'fairness_scores' in fairness_metrics:
                    for attr_name, attr_metrics in fairness_metrics['fairness_scores'].items():
                        fairness_score = attr_metrics.get('fairness_score', 1.0)
                        if fairness_score < 0.7:
                            recommendations.append(f"URGENT: Address fairness issues in {attr_name}")
                            recommendations.append("Implement demographic parity constraints")
                        elif fairness_score < 0.8:
                            recommendations.append(f"MEDIUM: Monitor fairness in {attr_name}")
            
            # Add general recommendations if no specific issues found
            if not recommendations:
                recommendations = [
                    "Implement regular bias monitoring",
                    "Establish fairness thresholds",
                    "Document model decisions",
                    "Provide model explanations",
                    "Conduct regular compliance audits"
                ]
            else:
                # Add general best practices
                recommendations.extend([
                    "Implement comprehensive bias monitoring in production",
                    "Document all mitigation strategies implemented",
                    "Establish regular bias monitoring procedures",
                    "Provide model explanations for affected groups",
                    "Consider human oversight for high-stakes decisions"
                ])
            
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph("{}: {}".format(i, rec), styles['Normal']))
            
            # Build PDF
            doc.build(story)
        except Exception as e:
            raise IOError(f"Failed to save PDF report to '{report_path}': {e}")
        
        return report_path
    
    def monitor_realtime(
        self, 
        predictions_stream
    ):
        """
        FR-004: Simulate real-time bias monitoring with batch inputs.
        
        This function implements continuous monitoring of model predictions
        to detect bias in real-time. It processes batches of predictions
        and generates alerts when bias thresholds are exceeded, enabling
        proactive bias detection in production AI systems.
        
        The monitoring system provides:
        - Real-time bias detection across multiple metrics
        - Configurable thresholds for different bias types
        - JSON-formatted alerts with detailed metadata
        - Performance optimization for high-throughput processing
        - Historical tracking of bias patterns over time
        
        This is essential for maintaining fairness in production AI systems
        and ensuring compliance with regulatory requirements for continuous
        monitoring as specified in GDPR and AI Act.
        
        Args:
            predictions_stream (list): List of prediction batches to monitor
                                     Each batch should be a list/array of predictions
                                     Example: [[1,0,1,0], [1,1,0,1], [0,1,1,0]]
        
        Returns:
            list: List of monitoring alerts in JSON format containing:
                  - timestamp: ISO format timestamp of the alert
                  - batch_id: Identifier for the batch being processed
                  - chunk_id: Identifier for the chunk within the batch
                  - alert_type: Type of bias alert (HIGH_POSITIVE_RATE, etc.)
                  - metric: The metric that triggered the alert
                  - value: Actual value of the metric
                  - threshold: Threshold that was exceeded
                  - severity: Alert severity level (WARNING, CRITICAL)
        
        Raises:
            ValueError: If predictions_stream is empty or contains invalid data
            TypeError: If predictions are not in the expected format
        
        Example:
            >>> validator = EthicalAIValidator()
            >>> # Configure monitoring thresholds
            >>> validator.config['bias_threshold'] = 0.05
            >>> validator.config['fairness_threshold'] = 0.9
            >>> 
            >>> # Monitor predictions in real-time
            >>> predictions_batches = [
            ...     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Batch 1
            ...     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Batch 2
            ...     [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]   # Batch 3
            ... ]
            >>> alerts = validator.monitor_realtime(predictions_batches)
            >>> for alert in alerts:
            ...     print(f"Alert: {alert['alert_type']} at {alert['timestamp']}")
        
        Author: WHIS (muhammadabdullahinbox@gmail.com)
        """
        alerts = []
        batch_size = 1000  # Process in batches for efficiency
        
        for batch_idx, predictions_batch in enumerate(predictions_stream):
            predictions = np.array(predictions_batch)
            
            # Process in smaller chunks for real-time monitoring
            for i in range(0, len(predictions), batch_size):
                chunk = predictions[i:i + batch_size]
                
                # Calculate basic statistics for this chunk
                positive_rate = np.mean(chunk)
                timestamp = datetime.now().isoformat()
                
                # Define bias thresholds (configurable)
                bias_threshold = self.config.get('bias_threshold', 0.1)
                fairness_threshold = self.config.get('fairness_threshold', 0.8)
                
                # Check for bias alerts
                if positive_rate > 0.5 + bias_threshold:
                    alerts.append({
                        'timestamp': timestamp,
                        'batch_id': batch_idx,
                        'chunk_id': i // batch_size,
                        'alert_type': 'HIGH_POSITIVE_RATE',
                        'metric': 'positive_rate',
                        'value': positive_rate,
                        'threshold': 0.5 + bias_threshold,
                        'severity': 'WARNING'
                    })
                
                elif positive_rate < 0.5 - bias_threshold:
                    alerts.append({
                        'timestamp': timestamp,
                        'batch_id': batch_idx,
                        'chunk_id': i // batch_size,
                        'alert_type': 'LOW_POSITIVE_RATE',
                        'metric': 'positive_rate',
                        'value': positive_rate,
                        'threshold': 0.5 - bias_threshold,
                        'severity': 'WARNING'
                    })
                
                # Store monitoring history
                self.monitoring_history.append({
                    'timestamp': timestamp,
                    'batch_id': batch_idx,
                    'chunk_id': i // batch_size,
                    'positive_rate': positive_rate,
                    'sample_count': len(chunk)
                })
        
        return alerts
    
    def suggest_mitigations(self, bias_report):
        """
        FR-005: Suggest mitigation strategies based on bias analysis.
        
        This function analyzes bias report results and suggests appropriate
        mitigation strategies to reduce bias and improve fairness in AI models.
        It provides actionable recommendations based on the severity and type
        of bias detected in the audit process.
        
        The mitigation suggestions include:
        - Data preprocessing techniques (reweighting, balancing)
        - Model training strategies (adversarial debiasing, fairness constraints)
        - Post-processing methods (threshold adjustment, equalized odds)
        - Monitoring and maintenance recommendations
        - Regulatory compliance improvements
        
        These suggestions help organizations implement effective bias reduction
        strategies and maintain compliance with fairness requirements in
        regulations like GDPR and AI Act.
        
        Args:
            bias_report (pd.DataFrame): DataFrame containing bias analysis results
                                      from audit_bias() method. Must contain columns
                                      like 'statistical_parity', 'equalized_odds',
                                      'demographic_parity', 'bias_score'.
        
        Returns:
            dict: Comprehensive mitigation suggestions containing:
                  - suggestions: List of specific mitigation strategies
                  - priority: Priority level (LOW, MEDIUM, HIGH)
                  - estimated_effort: Estimated implementation effort (LOW, MEDIUM, HIGH)
                  - risk_assessment: Assessment of bias risk level
                  - compliance_impact: Impact on regulatory compliance
        
        Raises:
            ValueError: If bias_report is empty or missing required columns
            TypeError: If bias_report is not a pandas DataFrame
        
        Example:
            >>> validator = EthicalAIValidator()
            >>> predictions = [1, 0, 1, 0, 1, 0, 1, 0]
            >>> true_labels = [1, 0, 1, 0, 1, 0, 1, 0]
            >>> protected_attrs = {
            ...     'gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female']
            ... }
            >>> bias_report = validator.audit_bias(predictions, true_labels, protected_attrs)
            >>> mitigations = validator.suggest_mitigations(bias_report)
            >>> print(f"Priority: {mitigations['priority']}")
            >>> print(f"Suggestions: {mitigations['suggestions']}")
        
        Author: WHIS (muhammadabdullahinbox@gmail.com)
        """
        if bias_report.empty:
            return {
                'suggestions': ['No bias detected - no mitigations needed'],
                'priority': 'LOW',
                'estimated_effort': 'NONE'
            }
        
        suggestions = []
        priority = 'LOW'
        estimated_effort = 'LOW'
        
        # Analyze statistical parity disparities
        if 'statistical_parity' in bias_report.columns:
            max_disparity = bias_report['statistical_parity'].abs().max()
            
            if max_disparity > 0.2:
                suggestions.append("Implement reweighting techniques to balance class distributions")
                suggestions.append("Consider using adversarial debiasing during training")
                priority = 'HIGH'
                estimated_effort = 'HIGH'
            elif max_disparity > 0.1:
                suggestions.append("Apply post-processing techniques to adjust prediction thresholds")
                suggestions.append("Consider using equalized odds post-processing")
                priority = 'MEDIUM'
                estimated_effort = 'MEDIUM'
            else:
                suggestions.append("Monitor bias metrics regularly to prevent drift")
                priority = 'LOW'
                estimated_effort = 'LOW'
        
        # Analyze equalized odds disparities
        if 'equalized_odds' in bias_report.columns:
            max_odds_disparity = bias_report['equalized_odds'].abs().max()
            
            if max_odds_disparity > 0.15:
                suggestions.append("Implement equalized odds post-processing")
                suggestions.append("Consider using adversarial training with fairness constraints")
                if priority != 'HIGH':
                    priority = 'MEDIUM'
                    estimated_effort = 'MEDIUM'
        
        # Analyze demographic parity
        if 'demographic_parity' in bias_report.columns:
            max_demographic_disparity = bias_report['demographic_parity'].abs().max()
            
            if max_demographic_disparity > 0.25:
                suggestions.append("Apply demographic parity constraints during training")
                suggestions.append("Consider using preprocessing techniques to balance data")
                if priority != 'HIGH':
                    priority = 'MEDIUM'
                    estimated_effort = 'MEDIUM'
        
        # General suggestions based on bias scores
        if 'bias_score' in bias_report.columns:
            max_bias_score = bias_report['bias_score'].max()
            
            if max_bias_score > 0.3:
                suggestions.append("Consider retraining the model with fairness-aware algorithms")
                suggestions.append("Implement comprehensive bias monitoring in production")
                priority = 'HIGH'
                estimated_effort = 'HIGH'
        
        # Add general best practices
        suggestions.extend([
            "Document all mitigation strategies implemented",
            "Establish regular bias monitoring procedures",
            "Provide model explanations for affected groups",
            "Consider human oversight for high-stakes decisions"
        ])
        
        return {
            'suggestions': suggestions,
            'priority': priority,
            'estimated_effort': estimated_effort,
            'bias_summary': {
                'max_statistical_parity': bias_report.get('statistical_parity', pd.Series()).abs().max(),
                'max_equalized_odds': bias_report.get('equalized_odds', pd.Series()).abs().max(),
                'max_demographic_parity': bias_report.get('demographic_parity', pd.Series()).abs().max(),
                'max_bias_score': bias_report.get('bias_score', pd.Series()).max()
            }
        }


# Convenience functions for direct access
def audit_bias(predictions, true_labels, protected_attributes):
    """
    Convenience function for FR-001: Bias detection.
    
    This function provides direct access to bias detection functionality
    without needing to instantiate the EthicalAIValidator class.
    
    Args:
        predictions (array-like): Model predictions
        true_labels (array-like): Ground truth labels
        protected_attributes (dict): Protected attributes dictionary
    
    Returns:
        pd.DataFrame: Bias analysis results
        
    Author: WHIS (muhammadabdullahinbox@gmail.com)
    """
    validator = EthicalAIValidator()
    return validator.audit_bias(predictions, true_labels, protected_attributes)


def calculate_fairness_metrics(predictions, protected_attributes):
    """
    Convenience function for FR-002: Fairness metrics calculation.
    
    This function provides direct access to fairness metrics calculation
    without needing to instantiate the EthicalAIValidator class.
    
    Args:
        predictions (array-like): Model predictions
        protected_attributes (dict): Protected attributes dictionary
    
    Returns:
        dict: Fairness metrics results
        
    Author: WHIS (muhammadabdullahinbox@gmail.com)
    """
    validator = EthicalAIValidator()
    return validator.calculate_fairness_metrics(predictions, protected_attributes)


def generate_compliance_report(metadata, audit_criteria, output_path=None):
    """
    Convenience function for FR-003: Compliance report generation.
    
    This function provides direct access to compliance report generation
    without needing to instantiate the EthicalAIValidator class.
    
    Args:
        metadata (dict): Model and audit metadata
        audit_criteria (dict): Audit criteria and thresholds
        output_path (str, optional): Path (including filename) to save the PDF report. If not provided, saves to current directory with a timestamped name.
    
    Returns:
        str: Path to generated PDF report
        
    Author: WHIS (muhammadabdullahinbox@gmail.com)
    """
    validator = EthicalAIValidator()
    return validator.generate_compliance_report(metadata, audit_criteria, output_path=output_path)


def monitor_realtime(predictions_stream):
    """
    Convenience function for FR-004: Real-time monitoring.
    
    This function provides direct access to real-time monitoring functionality
    without needing to instantiate the EthicalAIValidator class.
    
    Args:
        predictions_stream (list): List of prediction batches
    
    Returns:
        list: Monitoring alerts in JSON format
        
    Author: WHIS (muhammadabdullahinbox@gmail.com)
    """
    validator = EthicalAIValidator()
    return validator.monitor_realtime(predictions_stream)


def suggest_mitigations(bias_report):
    """
    Convenience function for FR-005: Mitigation suggestions.
    
    This function provides direct access to mitigation suggestion functionality
    without needing to instantiate the EthicalAIValidator class.
    
    Args:
        bias_report (pd.DataFrame): Bias analysis results
    
    Returns:
        dict: Mitigation suggestions and recommendations
        
    Author: WHIS (muhammadabdullahinbox@gmail.com)
    """
    validator = EthicalAIValidator()
    return validator.suggest_mitigations(bias_report)