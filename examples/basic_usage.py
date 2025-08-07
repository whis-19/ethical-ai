"""
Basic usage example for Ethical AI Validator.

This example demonstrates how to use the EthicalAIValidator
to audit a machine learning model for bias, fairness, and compliance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ethical_ai_validator.ethical_ai_validator import EthicalAIValidator


def main():
    """Run a basic example of ethical AI validation."""
    
    print("Ethical AI Validator - Basic Usage Example")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("\n1. Creating sample dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=15,
        n_informative=8,
        n_redundant=4,
        random_state=42
    )
    
    # Convert to DataFrame for better handling
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    # Create synthetic sensitive features
    np.random.seed(42)
    sensitive_features = pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], size=2000),
        'age_group': np.random.choice(['18-25', '26-35', '36-50', '50+'], size=2000),
        'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], size=2000)
    })
    
    print(f"Dataset created: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Sensitive features: {list(sensitive_features.columns)}")
    
    # Step 2: Split data and train model
    print("\n2. Training machine learning model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model trained successfully")
    
    # Step 3: Initialize Ethical AI Validator
    print("\n3. Initializing Ethical AI Validator...")
    validator = EthicalAIValidator()
    print("Validator initialized")
    
    # Step 4: Run comprehensive audit
    print("\n4. Running comprehensive audit...")
    
    # Get predictions
    predictions = model.predict(X_test)
    
    # Prepare sensitive features for the test set
    test_sensitive_features = sensitive_features.iloc[X_test.index]
    
    # Run bias audit
    bias_report = validator.audit_bias(
        predictions=predictions,
        true_labels=list(y_test),
        protected_attributes={
            'gender': test_sensitive_features['gender'].tolist(),
            'age_group': test_sensitive_features['age_group'].tolist(),
            'education': test_sensitive_features['education'].tolist()
        }
    )
    
    # Calculate fairness metrics
    fairness_metrics = validator.calculate_fairness_metrics(
        predictions=predictions,
        protected_attributes={
            'gender': test_sensitive_features['gender'].tolist(),
            'age_group': test_sensitive_features['age_group'].tolist(),
            'education': test_sensitive_features['education'].tolist()
        }
    )
    
    print("Audit completed successfully")
    
    # Step 5: Display results
    print("\n5. Audit Results:")
    print("-" * 30)
    
    # Bias Analysis
    if not bias_report.empty:
        print("Bias Analysis:")
        for _, row in bias_report.iterrows():
            print("  {} - {}: Bias Score {:.3f}".format(
                row['protected_attribute'], row['group'], row['bias_score']
            ))
    
    # Fairness Assessment
    print("\nFairness Assessment:")
    for attr_name, attr_metrics in fairness_metrics['fairness_scores'].items():
        print("  {}: Fairness Score {:.3f}".format(
            attr_name, attr_metrics['fairness_score']
        ))
    
    # Step 6: Generate recommendations
    print("\n6. Recommendations:")
    print("-" * 30)
    mitigations = validator.suggest_mitigations(bias_report)
    for i, suggestion in enumerate(mitigations['suggestions'], 1):
        print(f"{i}. {suggestion}")
    
    # Step 7: Generate compliance report
    print("\n7. Generating compliance report...")
    metadata = {'model_name': 'RandomForest', 'version': '1.0'}
    audit_criteria = {'bias_threshold': 0.1, 'fairness_threshold': 0.8}
    report_path = validator.generate_compliance_report(metadata, audit_criteria)
    print(f"Report saved to: {report_path}")
    
    print("\nExample completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main() 