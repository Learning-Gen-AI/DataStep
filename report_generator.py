import polars as pl
from typing import Dict, List, Any
from pathlib import Path
import json
from datetime import datetime
import jinja2
import os

class ReportGenerator:
    def __init__(self, config: Dict):
        """
        Initialize report generator with configuration.
        
        Args:
            config (Dict): Configuration containing:
                output_dir (str): Directory for report outputs
                company_name (str): Name of insurance company
                report_formats (List[str]): List of required formats ['csv', 'json', 'html']
                include_charts (bool): Whether to include charts in HTML report
        """
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize Jinja2 for HTML templates
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates'),
            autoescape=True
        )

    def _generate_report_name(self, report_type: str, format: str) -> str:
        """Generate standardized report filename."""
        company = self.config['company_name'].lower().replace(' ', '_')
        return f"{company}_{report_type}_{self.timestamp}.{format}"

    def _write_csv_report(self, data: pl.DataFrame, report_type: str) -> str:
        """Write report data to CSV file."""
        filepath = self.output_dir / self._generate_report_name(report_type, 'csv')
        data.write_csv(filepath)
        return str(filepath)

    def _write_json_report(self, data: Dict, report_type: str) -> str:
        """Write report data to JSON file."""
        filepath = self.output_dir / self._generate_report_name(report_type, 'json')
        
        # Convert any Polars DataFrames to dictionaries
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, pl.DataFrame):
                processed_data[key] = value.to_dict(as_series=False)
            else:
                processed_data[key] = value
                
        with open(filepath, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
            
        return str(filepath)

    def _generate_html_report(
        self,
        validation_results: Dict,
        outlier_results: Dict,
        comparison_results: Dict
    ) -> str:
        """Generate comprehensive HTML report."""
        template = self.jinja_env.get_template('report_template.html')
        
        # Prepare data for the template
        context = {
            'company_name': self.config['company_name'],
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'validation_summary': {
                'total_errors': validation_results['error_count'],
                'total_warnings': validation_results['warning_count'],
                'failed_rules': validation_results['failed_validations']
            },
            'comparison_summary': {
                'total_current': comparison_results['comparison_stats']['total_records_current'],
                'total_previous': comparison_results['comparison_stats']['total_records_previous'],
                'new_records': comparison_results['comparison_stats']['new_records'],
                'lapsed_records': comparison_results['comparison_stats']['lapsed_records'],
                'retention_rate': f"{comparison_results['comparison_stats']['retention_rate']:.2f}%"
            },
            'outlier_summary': outlier_results
        }
        
        # Generate HTML
        html_content = template.render(**context)
        
        # Write to file
        filepath = self.output_dir / self._generate_report_name('complete', 'html')
        with open(filepath, 'w') as f:
            f.write(html_content)
            
        return str(filepath)

    def generate_validation_report(
        self,
        validation_results: Dict,
        output_formats: List[str] = None
    ) -> Dict[str, str]:
        """Generate validation report in specified formats."""
        if output_formats is None:
            output_formats = self.config.get('report_formats', ['csv', 'json'])
            
        report_files = {}
        
        if 'csv' in output_formats:
            report_files['csv'] = self._write_csv_report(
                validation_results['validation_results'],
                'validation'
            )
            
        if 'json' in output_formats:
            report_files['json'] = self._write_json_report(
                validation_results,
                'validation'
            )
            
        return report_files

    def generate_outlier_report(
        self,
        outlier_results: Dict,
        output_formats: List[str] = None
    ) -> Dict[str, str]:
        """Generate outlier report in specified formats."""
        if output_formats is None:
            output_formats = self.config.get('report_formats', ['csv', 'json'])
            
        report_files = {}
        
        if 'csv' in output_formats:
            # Convert outlier summary to DataFrame for CSV
            summary_df = pl.DataFrame([
                {
                    'column': col,
                    'outlier_count': len(outliers),
                    'outlier_values': str(outliers[col].unique().to_list())
                }
                for col, outliers in outlier_results.items()
            ])
            report_files['csv'] = self._write_csv_report(summary_df, 'outliers')
            
        if 'json' in output_formats:
            report_files['json'] = self._write_json_report(
                outlier_results,
                'outliers'
            )
            
        return report_files

    def generate_comparison_report(
        self,
        comparison_results: Dict,
        output_formats: List[str] = None
    ) -> Dict[str, str]:
        """Generate comparison report in specified formats."""
        if output_formats is None:
            output_formats = self.config.get('report_formats', ['csv', 'json'])
            
        report_files = {}
        
        if 'csv' in output_formats:
            # Generate separate CSV files for lapsed and new records
            lapsed_file = self._write_csv_report(
                comparison_results['table_lapsed'].collect(),
                'lapsed_records'
            )
            new_file = self._write_csv_report(
                comparison_results['table_new'].collect(),
                'new_records'
            )
            report_files['csv'] = {'lapsed': lapsed_file, 'new': new_file}
            
        if 'json' in output_formats:
            report_files['json'] = self._write_json_report(
                comparison_results,
                'comparison'
            )
            
        return report_files

    def generate_complete_report(
        self,
        validation_results: Dict,
        outlier_results: Dict,
        comparison_results: Dict
    ) -> Dict[str, str]:
        """Generate comprehensive report in all formats."""
        report_files = {
            'validation': self.generate_validation_report(validation_results),
            'outliers': self.generate_outlier_report(outlier_results),
            'comparison': self.generate_comparison_report(comparison_results)
        }
        
        # Generate HTML report if configured
        if 'html' in self.config.get('report_formats', []):
            report_files['html'] = self._generate_html_report(
                validation_results,
                outlier_results,
                comparison_results
            )
            
        return report_files