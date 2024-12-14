import polars as pl
from typing import Dict, List, Optional
from pathlib import Path
from enum import Enum

class FileFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    TSV = "tsv"

class DataLoader:
    def __init__(self, config: Dict):
        """
        Initialize the data loader with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing file paths and settings
        """
        self.config = config
        
    def _detect_file_format(self, file_path: str) -> FileFormat:
        """Detect file format from file extension."""
        extension = Path(file_path).suffix.lower()
        if extension == '.csv':
            return FileFormat.CSV
        elif extension == '.parquet':
            return FileFormat.PARQUET
        elif extension == '.tsv':
            return FileFormat.TSV
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _load_csv(self, file_path: str) -> pl.LazyFrame:
        """Load CSV or TSV file."""
        delimiter = '\t' if file_path.endswith('.tsv') else ','
        return pl.scan_csv(
            file_path,
            separator=delimiter,
            encoding=self.config.get('encoding', 'utf8'),
            try_parse_dates=True
        )

    def _load_parquet(self, file_path: str) -> pl.LazyFrame:
        """Load Parquet file."""
        return pl.scan_parquet(file_path)

    def load_data(self, current_year_path: str, previous_year_path: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Load both current and previous year datasets.
        
        Args:
            current_year_path (str): Path to current year's data file
            previous_year_path (str): Path to previous year's data file
            
        Returns:
            tuple[pl.LazyFrame, pl.LazyFrame]: Current and previous year data as LazyFrames
        """
        # Load both datasets
        current_format = self._detect_file_format(current_year_path)
        previous_format = self._detect_file_format(previous_year_path)
        
        # Load current year data
        if current_format in [FileFormat.CSV, FileFormat.TSV]:
            current_data = self._load_csv(current_year_path)
        else:
            current_data = self._load_parquet(current_year_path)
            
        # Load previous year data
        if previous_format in [FileFormat.CSV, FileFormat.TSV]:
            previous_data = self._load_csv(previous_year_path)
        else:
            previous_data = self._load_parquet(previous_year_path)
            
        return current_data, previous_data

    def validate_columns(self, current_data: pl.LazyFrame, previous_data: pl.LazyFrame) -> bool:
        """
        Validate that both datasets have the same columns in the same order.
        
        Args:
            current_data (pl.LazyFrame): Current year's data
            previous_data (pl.LazyFrame): Previous year's data
            
        Returns:
            bool: True if columns match, raises ValueError if they don't
        """
        current_cols = current_data.collect_schema().names()
        previous_cols = previous_data.collect_schema().names()
        
        # Check if columns are identical and in same order
        if current_cols != previous_cols:
            # Find differences for error message
            missing_in_current = set(previous_cols) - set(current_cols)
            missing_in_previous = set(current_cols) - set(previous_cols)
            different_order = current_cols != previous_cols and not (missing_in_current or missing_in_previous)
            
            error_msg = []
            if missing_in_current:
                error_msg.append(f"Columns missing in current year: {missing_in_current}")
            if missing_in_previous:
                error_msg.append(f"Columns missing in previous year: {missing_in_previous}")
            if different_order:
                error_msg.append("Columns are in different order")
                
            raise ValueError("\n".join(error_msg))
            
        return True