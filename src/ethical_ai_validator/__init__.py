"""
Ethical AI Validator

A comprehensive Python package for ethical AI validation and auditing,
designed to detect bias, assess fairness, and ensure compliance with
regulations like GDPR and AI Act.

Features:
- Bias detection and disparity metrics
- Fairness assessment across protected attributes
- GDPR and AI Act compliance reporting
- Real-time monitoring with alerts
- Automated mitigation suggestions

Example:
    >>> from ethical_ai_validator import EthicalAIValidator
    >>> validator = EthicalAIValidator()
    >>> bias_report = validator.audit_bias(predictions, labels, protected_attrs)
"""

__version__ = "1.0.0"
__author__ = "Ethical AI Team"
__email__ = "muhammadabdullahinbox@gmail.com"

# Import main classes and functions
from .ethical_ai_validator import (
    EthicalAIValidator,
    audit_bias,
    calculate_fairness_metrics,
    generate_compliance_report,
    monitor_realtime,
    suggest_mitigations
)

# Import convenience functions
from .ethical_ai_validator import (
    audit_bias as audit_bias_fn,
    calculate_fairness_metrics as calculate_fairness_metrics_fn,
    generate_compliance_report as generate_compliance_report_fn,
    monitor_realtime as monitor_realtime_fn,
    suggest_mitigations as suggest_mitigations_fn
)

__all__ = [
    "EthicalAIValidator",
    "audit_bias",
    "calculate_fairness_metrics", 
    "generate_compliance_report",
    "monitor_realtime",
    "suggest_mitigations",
    # Convenience functions
    "audit_bias_fn",
    "calculate_fairness_metrics_fn",
    "generate_compliance_report_fn",
    "monitor_realtime_fn",
    "suggest_mitigations_fn"
] 