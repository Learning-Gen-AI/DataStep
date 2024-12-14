import polars as pl
from typing import Dict, List, Tuple
from enum import Enum
import numpy as np

class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"

class OutlierDetector:
    def __init__(self, config: Dict):
        """
        Initialize outlier detector with configuration.
        
        Args:
            config (Dict): Configuration containing thresholds and settings
                percentile_threshold (float): Threshold for top/bottom percentiles (default: 10.0)
                rare_category_threshold (float): Threshold for rare categories (default: 1.0)
        """
        self.percentile_threshold = config.get('percentile_threshold', 10.0)
        self.rare_category_threshold = config.get('rare_category_threshold', 1.0)

    def _determine_column_type(self, df: pl.LazyFrame, column: str) -> ColumnType:
        """Determine the type of a column."""
        # Get the first non-null value to check type
        sample = df.select(pl.col(column)).filter(pl.col(column).is_not_null()).first().collect()
        if len(sample) == 0:
            raise ValueError(f"Column {column} is empty")
            
        value = sample[0, 0]
        
        if isinstance(value, (int, float)):
            return ColumnType.NUMERIC
        elif isinstance(value, (pl.Date, pl.Datetime)):
            return ColumnType.DATE
        else:
            return ColumnType.CATEGORICAL

    def _detect_numeric_outliers(self, df: pl.LazyFrame, column: str) -> pl.DataFrame:
        """
        Detect outliers in numeric columns using percentile method and IQR.
        """
        # Calculate percentiles
        percentiles = df.select(
            pl.col(column).quantile(1 - (self.percentile_threshold/100)).alias('upper_percentile'),
            pl.col(column).quantile(self.percentile_threshold/100).alias('lower_percentile'),
            pl.col(column).quantile(0.75).alias('q3'),
            pl.col(column).quantile(0.25).alias('q1')
        ).collect()
        
        # Calculate IQR bounds
        q1 = float(percentiles['q1'][0])
        q3 = float(percentiles['q3'][0])
        iqr = q3 - q1
        iqr_lower = q1 - (1.5 * iqr)
        iqr_upper = q3 + (1.5 * iqr)
        
        # Find outliers
        outliers = df.filter(
            (pl.col(column) > pl.lit(iqr_upper)) | 
            (pl.col(column) < pl.lit(iqr_lower))
        ).select([
            pl.col(column),
            pl.lit('IQR').alias('method'),
            pl.when(pl.col(column) > iqr_upper)
            .then(pl.lit('above'))
            .otherwise(pl.lit('below'))
            .alias('direction')
        ]).collect()
        
        return outliers

    def _detect_categorical_outliers(self, df: pl.LazyFrame, column: str) -> pl.DataFrame:
        """Detect rare categories using frequency analysis."""
        try:
            # Convert to eager execution
            df_collected = df.collect()
            
            total_count = len(df_collected)
            
            # Simple frequency count
            frequencies = (
                df_collected
                .select(pl.col(column))
                .group_by(column)
                .agg(
                    pl.count().alias("count")
                )
                .with_columns(
                    (pl.col("count") * 100 / total_count).alias("percentage")
                )
                .filter(pl.col("percentage") < self.rare_category_threshold)
            )
            
            return frequencies
            
        except Exception as e:
            print(f"Error processing categorical outliers for {column}: {str(e)}")
            return pl.DataFrame()

    def _detect_date_outliers(self, df: pl.LazyFrame, column: str) -> pl.DataFrame:
        """
        Detect outliers in date columns using percentile method.
        """
        # Convert dates to unix timestamps for numerical analysis
        df_with_ts = df.with_columns(
            pl.col(column).cast(pl.Int64).alias('_temp_ts')
        )
        
        # Use numeric outlier detection on timestamps
        outliers = self._detect_numeric_outliers(df_with_ts, '_temp_ts')
        
        # Convert timestamps back to dates
        if not outliers.is_empty():
            outliers = outliers.with_columns(
                pl.from_epoch('_temp_ts').alias(column)
            ).drop('_temp_ts')
            
        return outliers

    def detect_outliers(self, df: pl.LazyFrame) -> Dict[str, pl.DataFrame]:
        """
        Detect outliers in all columns of the dataframe.
        
        Args:
            df (pl.LazyFrame): Input dataframe
            
        Returns:
            Dict[str, pl.DataFrame]: Dictionary mapping column names to their outliers
        """
        results = {}
        columns = df.collect_schema().names()
        
        for column in columns:
            try:
                # Collect the data for type determination
                sample_data = df.select(pl.col(column)).collect()
                if len(sample_data) == 0:
                    continue
                    
                # Determine column type from collected data
                first_value = sample_data[0, 0]
                
                if isinstance(first_value, (int, float)):
                    outliers = self._detect_numeric_outliers(df, column)
                elif isinstance(first_value, (pl.Date, pl.Datetime)):
                    outliers = self._detect_date_outliers(df, column)
                else:
                    outliers = self._detect_categorical_outliers(df, column)
                
                if not outliers.is_empty():
                    results[column] = outliers
                    
            except Exception as e:
                print(f"Warning: Could not process column {column}: {str(e)}")
                continue
                
        return results

    def summarize_outliers(self, outliers_dict: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        Create a summary of detected outliers.
        
        Args:
            outliers_dict (Dict[str, pl.DataFrame]): Dictionary of outliers by column
            
        Returns:
            pl.DataFrame: Summary statistics of outliers
        """
        summaries = []
        
        for column, outliers in outliers_dict.items():
            summary = {
                'column': column,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) * 100,  # Will be divided by total later
                'unique_outlier_values': outliers[column].n_unique()
            }
            summaries.append(summary)
            
        return pl.DataFrame(summaries)