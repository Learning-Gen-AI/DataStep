import polars as pl
from typing import Dict, List, Union
from datetime import datetime

class DataComparison:
    def __init__(self, config: Dict):
        """
        Initialize data comparison with configuration.
        
        Args:
            config (Dict): Configuration containing:
                primary_keys (List[str]): List of columns forming the primary key
                join_type (str): Type of join to perform (default: 'outer')
                chunk_size (int): Size of chunks for processing (default: 100000)
        """
        if 'primary_keys' not in config:
            raise ValueError("Primary keys must be specified in config")
            
        self.primary_keys = config['primary_keys']
        self.join_type = config.get('join_type', 'outer')
        self.chunk_size = config.get('chunk_size', 100000)

    def validate_primary_keys(self, df: pl.LazyFrame, year_label: str) -> bool:
        """Validate that primary keys exist and are unique in the dataset."""
        # Check if all primary keys exist in the dataset
        df_columns = df.collect_schema().names()
        missing_keys = [key for key in self.primary_keys if key not in df_columns]
        if missing_keys:
            raise ValueError(f"Primary keys {missing_keys} not found in {year_label} year dataset")

        # Collect the data and check for duplicates
        collected_df = df.collect()
        
        # Simple duplicate check
        duplicate_check = (
            collected_df
            .select(self.primary_keys)
            .group_by(self.primary_keys)
            .agg(
                pl.count().alias("count")
            )
            .filter(pl.col("count") > 1)
        )

        if len(duplicate_check) > 0:
            raise ValueError(f"Duplicate primary keys found in {year_label} year dataset")
        
        return True

    def prepare_for_comparison(
        self,
        current_data: pl.LazyFrame,
        previous_data: pl.LazyFrame
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Prepare datasets for comparison by adding year indicators.
        """
        # Add indicators to help identify source after join
        current_with_indicator = current_data.with_columns(
            pl.lit(True).alias('in_current_year')
        )
        
        previous_with_indicator = previous_data.with_columns(
            pl.lit(True).alias('in_previous_year')
        )
        
        return current_with_indicator, previous_with_indicator

    def join_datasets(
        self,
        current_data: pl.LazyFrame,
        previous_data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Join current and previous year datasets.
        """
        # Prepare datasets
        current_prepared, previous_prepared = self.prepare_for_comparison(
            current_data, previous_data
        )
        
        # Perform the join
        joined = current_prepared.join(
            previous_prepared,
            on=self.primary_keys,
            how=self.join_type,
            suffix='_prev'
        )
        
        return joined

    def identify_lapsed_and_new(
        self,
        joined_data: pl.LazyFrame
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Identify lapsed and new records from joined dataset.
        
        Returns:
            tuple[pl.LazyFrame, pl.LazyFrame]: (table_lapsed, table_new)
        """
        # Identify lapsed records (in previous year but not in current)
        table_lapsed = joined_data.filter(
            pl.col('in_previous_year') & ~pl.col('in_current_year')
        )
        
        # Identify new records (in current year but not in previous)
        table_new = joined_data.filter(
            pl.col('in_current_year') & ~pl.col('in_previous_year')
        )
        
        # Clean up the indicators and suffixes
        clean_columns = [col for col in joined_data.columns 
                        if not col.endswith('_prev') 
                        and col not in ['in_current_year', 'in_previous_year']]
        
        table_lapsed = table_lapsed.select(clean_columns)
        table_new = table_new.select(clean_columns)
        
        return table_lapsed, table_new

    def generate_comparison_stats(
        self,
        table_lapsed: pl.LazyFrame,
        table_new: pl.LazyFrame,
        joined_data: pl.LazyFrame
    ) -> Dict:
        """
        Generate statistics about the comparison.
        """
        stats = {}
        
        # Count statistics
        stats['total_records_current'] = joined_data.filter(
            pl.col('in_current_year')
        ).select(pl.count()).collect()[0, 0]
        
        stats['total_records_previous'] = joined_data.filter(
            pl.col('in_previous_year')
        ).select(pl.count()).collect()[0, 0]
        
        stats['lapsed_records'] = table_lapsed.select(pl.count()).collect()[0, 0]
        stats['new_records'] = table_new.select(pl.count()).collect()[0, 0]
        
        # Calculate retention rate
        stats['retention_rate'] = (
            (stats['total_records_current'] - stats['new_records']) /
            stats['total_records_previous'] * 100
        )
            
        return stats

    def compare_datasets(
        self,
        current_data: pl.LazyFrame,
        previous_data: pl.LazyFrame
    ) -> Dict:
        """
        Main method to compare datasets using distinct primary keys
        """
        # Validate primary keys
        self.validate_primary_keys(current_data, 'current')
        self.validate_primary_keys(previous_data, 'previous')
        
        # Collect the data
        current_collected = current_data.collect()
        previous_collected = previous_data.collect()
        
        # Get distinct primary key combinations from each dataset
        current_keys = current_collected.select(self.primary_keys).unique()
        previous_keys = previous_collected.select(self.primary_keys).unique()
        
        # Find new records (in current but not in previous)
        table_new = (
            current_collected.join(
                previous_keys,
                on=self.primary_keys,
                how='anti'  # Only keep records that don't match
            )
        )
        
        # Find lapsed records (in previous but not in current)
        table_lapsed = (
            previous_collected.join(
                current_keys,
                on=self.primary_keys,
                how='anti'  # Only keep records that don't match
            )
        )
        
        # Generate statistics
        stats = {
            'total_records_current': len(current_collected),
            'total_records_previous': len(previous_collected),
            'lapsed_records': len(table_lapsed),
            'new_records': len(table_new),
        }
        
        # Calculate retention rate
        stats['retention_rate'] = (
            (stats['total_records_current'] - stats['new_records']) /
            stats['total_records_previous'] * 100
        )
        
        return {
            'joined_data': current_collected.join(
                previous_collected, 
                on=self.primary_keys, 
                how='outer',
                suffix='_prev'
            ).lazy(),
            'table_lapsed': table_lapsed.lazy(),
            'table_new': table_new.lazy(),
            'comparison_stats': stats
        }