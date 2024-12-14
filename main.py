import json
import polars as pl
from pathlib import Path
from typing import Dict
from data_loader import DataLoader
from outlier_detector import OutlierDetector
from data_comparison import DataComparison
from business_validator import BusinessValidator
from report_generator import ReportGenerator
from llm_validator import LLMValidator

def run_insurance_data_analysis(config: Dict) -> Dict:
    """
    Run the complete insurance data analysis workflow.
    
    Args:
        config (Dict): Main configuration dictionary containing all module configs
    
    Returns:
        Dict: Paths to generated reports
    """
    print("Starting Insurance Data Analysis Workflow...")

    # Initialize all components
    loader = DataLoader(config['loader'])
    outlier_detector = OutlierDetector(config['outlier'])
    comparator = DataComparison({
        'primary_keys': config['primary_keys'],
        **config['comparison']
    })
    validator = BusinessValidator(config['validator'])
    reporter = ReportGenerator(config['reporter'])
    llm_validator = LLMValidator({
      'primary_keys': config['primary_keys'],
        **config['llm_validator']
    })

    try:
        # 1. Load and validate data structure
        print("Loading datasets...")
        current_data, previous_data = loader.load_data(
            config['files']['current_year'],
            config['files']['previous_year']
        )
        loader.validate_columns(current_data, previous_data)
        print("Data loaded successfully.")

        # 2. Detect outliers
        print("Detecting outliers...")
        outlier_results = outlier_detector.detect_outliers(current_data)
        print(f"Found outliers in {len(outlier_results)} columns.")

        # 3. Compare datasets
        print("Comparing datasets...")
        comparison_results = comparator.compare_datasets(current_data, previous_data)
        print(f"Comparison complete. Found {comparison_results['comparison_stats']['new_records']} new records "
              f"and {comparison_results['comparison_stats']['lapsed_records']} lapsed records.")

        # 4. Validate business rules
        print("Validating business rules...")
        validation_results = validator.validate_data(current_data)
        print(f"Validation complete. Found {validation_results['error_count']} errors "
              f"and {validation_results['warning_count']} warnings.")

        # 5. Generate reports
        print("Generating reports...")
        report_files = reporter.generate_complete_report(
            validation_results,
            outlier_results,
            comparison_results
        )
        print("Reports generated successfully.")

        # Run LLM validation
        print("Running LLM validation...")
        llm_validator.validate_records(
            config['files']['current_year'],
            'reports/llm_validation_results.csv'
        )
        
        print("LLM Validation checks generated successfully.")

        return report_files

    except Exception as e:
        print(f"Error in analysis workflow: {str(e)}")
        raise

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Convert string lambda functions to actual lambda functions
    if 'validator' in config and 'custom_rules' in config['validator']:
        for rule in config['validator']['custom_rules'].values():
            if 'func' in rule and isinstance(rule['func'], str):
                # Safely evaluate lambda string to function
                # Note: Only use this with trusted config files
                rule['func'] = eval(rule['func'])
    
    return config

if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config('config.json')
        
        # Run the analysis
        report_files = run_insurance_data_analysis(config)
        
        # Print report locations
        print("\nGenerated Reports:")
        for report_type, files in report_files.items():
            if isinstance(files, dict):
                for sub_type, file_path in files.items():
                    print(f"{report_type} - {sub_type}: {file_path}")
            else:
                print(f"{report_type}: {files}")

    except Exception as e:
        print(f"Error: {str(e)}")