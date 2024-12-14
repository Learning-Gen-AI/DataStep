import polars as pl
from typing import Dict, List, Callable, Any
from datetime import datetime
from enum import Enum
import re

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ValidationRule:
    def __init__(
        self,
        name: str,
        validation_func: Callable,
        columns: List[str],
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        message: str = None
    ):
        self.name = name
        self.validation_func = validation_func
        self.columns = columns
        self.severity = severity
        self.message = message or name

class BusinessValidator:
    def __init__(self, config: Dict):
        """
        Initialize business validator with configuration.
        
        Args:
            config (Dict): Configuration containing:
                date_format (str): Expected date format
                gender_values (List[str]): Valid gender values
                min_premium (float): Minimum allowed premium
                custom_rules (Dict): Custom validation rules
        """
        self.config = config
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize standard and custom validation rules."""
        rules = []
        
        # Date validation rules
        rules.append(ValidationRule(
            name="inception_date_not_null",
            validation_func=lambda df, col: df.filter(pl.col(col).is_null()),
            columns=["CoverStartDate"],
            message="CoverStartDate cannot be null"
        ))
        
        rules.append(ValidationRule(
            name="cancellation_after_inception",
            validation_func=lambda df, cols: df.filter(
                (pl.col(cols[0]).is_not_null()) & 
                (pl.col(cols[1]).is_not_null()) & 
                (pl.col(cols[0]) > pl.col(cols[1]))
            ),
            columns=["CoverEndDate", "CoverStartDate"],
            message="CoverEndDate must be after CoverStartDate"
        ))
        
        # Premium validation rules
        rules.append(ValidationRule(
            name="premium_not_zero",
            validation_func=lambda df, col: df.filter(
                (pl.col(col) <= 0) | pl.col(col).is_null()
            ),
            columns=["MonthlyPremium"],
            message="MonthlyPremium must be greater than zero"
        ))
        
        rules.append(ValidationRule(
            name="premium_exceeds_minimum",
            validation_func=lambda df, col: df.filter(
                (pl.col(col) < self.config.get('min_premium', 0))
            ),
            columns=["MonthlyPremium"],
            severity=ValidationSeverity.WARNING,
            message=f"MonthlyPremium below minimum threshold of {self.config.get('min_premium', 0)}"
        ))
        
        # Add custom rules from config
        custom_rules = self.config.get('custom_rules', {})
        for rule_name, rule_config in custom_rules.items():
            rules.append(ValidationRule(
                name=rule_name,
                validation_func=rule_config['func'],
                columns=rule_config['columns'],
                severity=ValidationSeverity(rule_config.get('severity', 'error')),
                message=rule_config.get('message', rule_name)
            ))
            
        return rules

    def _validate_rule(
        self,
        df: pl.LazyFrame,
        rule: ValidationRule
    ) -> pl.DataFrame:
        """
        Apply a single validation rule and return violations.
        """
        # Check if all required columns exist
        missing_cols = [col for col in rule.columns if col not in df.columns]
        if missing_cols:
            return pl.DataFrame({
                'rule_name': [rule.name],
                'severity': [rule.severity.value],
                'message': [f"Missing columns: {missing_cols}"],
                'violation_count': [0]
            })
        
        # Apply the validation function
        violations = rule.validation_func(df, rule.columns)
        
        # Count violations
        violation_count = violations.select(pl.count()).collect()[0, 0]
        
        if violation_count > 0:
            # Get sample of violating records for the error message
            sample_violations = violations.select(rule.columns).collect()
            sample_msg = f"{rule.message}. Sample violations: {sample_violations[0]}"
        else:
            sample_msg = rule.message
            
        return pl.DataFrame({
            'rule_name': [rule.name],
            'severity': [rule.severity.value],
            'message': [sample_msg],
            'violation_count': [violation_count]
        })

    def validate_data(self, df: pl.LazyFrame) -> Dict:
        """
        Validate data against all business rules.
        
        Returns:
            Dict containing:
                - validation_results: DataFrame with all validation results
                - error_count: Total number of errors
                - warning_count: Total number of warnings
                - failed_validations: List of failed validation names
        """
        all_results = []
        
        # Apply each validation rule
        for rule in self.rules:
            result = self._validate_rule(df, rule)
            all_results.append(result)
            
        # Combine all results
        validation_results = pl.concat(all_results)
        
        # Calculate summary statistics
        error_count = validation_results.filter(
            (pl.col('severity') == 'error') & 
            (pl.col('violation_count') > 0)
        ).select(pl.sum('violation_count')).item()
        
        warning_count = validation_results.filter(
            (pl.col('severity') == 'warning') & 
            (pl.col('violation_count') > 0)
        ).select(pl.sum('violation_count')).item()
        
        failed_validations = validation_results.filter(
            pl.col('violation_count') > 0
        ).select('rule_name').to_series().to_list()
        
        return {
            'validation_results': validation_results,
            'error_count': error_count,
            'warning_count': warning_count,
            'failed_validations': failed_validations
        }

    def get_invalid_records(
        self,
        df: pl.LazyFrame,
        include_warnings: bool = False
    ) -> pl.LazyFrame:
        """
        Return all records that fail validation rules.
        
        Args:
            df (pl.LazyFrame): Input dataset
            include_warnings (bool): Whether to include records with warnings
            
        Returns:
            pl.LazyFrame: Records that fail validation
        """
        invalid_records = None
        
        for rule in self.rules:
            if not include_warnings and rule.severity == ValidationSeverity.WARNING:
                continue
                
            violations = rule.validation_func(df, rule.columns)
            
            if invalid_records is None:
                invalid_records = violations
            else:
                invalid_records = pl.concat([invalid_records, violations])
                
        return invalid_records if invalid_records is not None else pl.LazyFrame()