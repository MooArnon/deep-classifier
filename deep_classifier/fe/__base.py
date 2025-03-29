##########
# Import #
##############################################################################

import os

import polars as pl
from typing import Union, List, Optional

###########
# Classes #
##############################################################################

class BaseFE:
    def __init__(
        self,
        control_column: str = "date",
        target_column: str = "target",
        label: Optional[str] = None,
        fe_name_list: Optional[List[str]] = None,
        unused_feature: Optional[List[str]] = None
    ):
        """
        Parameters
        ----------
        control_column : str
            The name of the datetime column to control sorting.
        target_column : str
            The name of the target column.
        label : str, optional
            Another label if needed for your workflow.
        fe_name_list : list of str, optional
            Names of feature-engineering methods that this class will call.
        unused_feature : list of str, optional
            Columns that you want to drop after transformations.
        """
        self.control_column = control_column
        self.target_column = target_column
        self.label = label
        self.fe_name_list = fe_name_list or []
        self.unused_feature = unused_feature or []
        self.features = []
    
    # ------------------------------------------------
    # Decorator to parse control_column as datetime
    # and sort before calling the wrapped function
    # ------------------------------------------------

    def process_dataframe_decorator(func):
        """
        Decorator to process the DataFrame by converting
        the specified column (self.control_column) to datetime format
        and sorting the DataFrame based on that column.
        """
        def wrapper(self, df: pl.DataFrame, *args, **kwargs):
            # Convert the specified column to datetime
            # df = df.with_columns(
            #     pl.col(self.control_column)
            #     .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False)
            # )
            # Sort by the datetime column
            df = df.sort(by=self.control_column, descending=False)
            return func(self, df, *args, **kwargs)
        return wrapper
    
    @process_dataframe_decorator
    def add_binary_label(self, df: pl.DataFrame, threshold: float = 0.0) -> pl.DataFrame:
        """
        Add a binary label column (0 or 1) to the DataFrame based on price changes.

        - "LONG" → `1` (if next price >= threshold)
        - "SHORT" → `0` (otherwise)

        The first row for each asset is dropped.

        Parameters
        ----------
        df : pl.DataFrame
            The input DataFrame.
        threshold : float, optional
            Threshold to classify "LONG" (default is `0.0`).

        Returns
        -------
        pl.DataFrame
            DataFrame with a new `label` column (`0` or `1`).
        """
        if "asset" in df.columns:
            print('asset is at df')

            # Process each asset separately
            asset_values = df.select(pl.col("asset")).unique().to_series().to_list()
            asset_dfs = []

            for asset in asset_values:
                asset_df = df.filter(pl.col("asset") == asset).sort(by=self.control_column)

                asset_df = asset_df.with_columns(
                    (pl.col(self.target_column).shift(-1)).alias("shifted")
                )

                asset_df = asset_df.with_columns(
                    (pl.col('shifted') - pl.col(self.target_column)).alias("diff")
                )

                asset_df = asset_df.drop_nulls(subset=["diff"])

                # Encode labels as 0 (SHORT) and 1 (LONG)
                asset_df = asset_df.with_columns(
                    pl.when(pl.col("diff") >= threshold)
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias("label")
                )

                asset_dfs.append(asset_df)

            df = pl.concat(asset_dfs)
            return df
        else:
            df = df.with_columns(
                (pl.col(self.target_column) - pl.col(self.target_column).shift(1)).alias("diff")
            )
            df = df.drop_nulls(subset=["diff"])

            df = df.with_columns(
                pl.when(pl.col("diff") >= threshold)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("label")
            )
            return df
        
    # ------------------------------------------------
    # Example placeholder for reading in a Polars DataFrame
    # ------------------------------------------------
    def read_df(self, df_or_path: Union[str, pl.DataFrame]) -> pl.DataFrame:
        """
        If df_or_path is a string, assume it's a CSV path for demonstration.
        Otherwise, assume it's already a Polars DataFrame.
        """
        if isinstance(df_or_path, str):
            # Adjust to your real reading logic, e.g. pl.scan_csv for lazy
            return pl.read_csv(df_or_path)
        elif isinstance(df_or_path, pl.DataFrame):
            return df_or_path
        else:
            raise ValueError("Unsupported input type for df_or_path.")

    # ------------------------------------------------
    # Example placeholder function to delete unused columns
    # (Adjust as needed for your logic.)
    # ------------------------------------------------
    def delete_unused_columns(
            self,
            df: pl.DataFrame,
            target_column: str,
            control_column: str,
            label: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Drop or modify columns that you consider 'unused',
        except the ones needed: target_column, control_column, label, etc.
        """
        # For this example, we'll just return df without changes.
        # Implement your logic here if needed.
        return df

    # ------------------------------------------------
    # The transform_df method
    # ------------------------------------------------
    
    def transform_df(
            self,
            df: Union[os.PathLike, pl.DataFrame],
            keep_target: bool = True,
            keep_control: bool = False,
    ) -> pl.DataFrame:
        if isinstance(df, os.PathLike):
            df = self.read_df(df)
        elif isinstance(df, pl.DataFrame):
            df=df
        else:
            raise ValueError(f"df type is {type(df)}, it must be only os.PathLike and pl.DataFrame")
        
        if df[self.control_column].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(self.control_column)
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=True)
            )
        df = df.sort(by=self.control_column)

        df = self.delete_unused_columns(
            df=df,
            target_column=self.target_column,
            control_column=self.control_column,
            label=self.label
        )
        
        asset_values = df.select(pl.col("asset")).unique().to_series().to_list()
        asset_dfs = []
        for asset in asset_values:
            df_filtered = df.filter(pl.col("asset") == asset)
            for fe_name in self.fe_name_list:
                fe_function = getattr(self, fe_name, None)
                if fe_function is None or not callable(fe_function):
                    print(f"Warning: FE function '{fe_name}' not found or not callable.")
                    continue
                df_filtered = fe_function(df_filtered)
            df_filtered = df_filtered.drop_nulls()
            asset_dfs.append(df_filtered)
            del df_filtered
        df = pl.concat(asset_dfs)

        # Drop unused features (only if they exist)
        if self.unused_feature:
            drop_cols = [col for col in self.unused_feature if col in df.columns]
            if drop_cols:
                df = df.drop(drop_cols)
        df = self.add_binary_label(df)
        self.final_column.append('label')
        
        # Keep or drop columns based on the flags
        if not keep_control and self.control_column in df.columns:
            if keep_target:
                # keep the target, drop only control
                df = df.drop(self.control_column)
            else:
                # drop both target and control
                cols_to_drop = []
                if self.target_column in df.columns:
                    cols_to_drop.append(self.target_column)
                cols_to_drop.append(self.control_column)
                df = df.drop(cols_to_drop)
        else:
            # keep_control=True => skip dropping the control column
            if not keep_target and self.target_column in df.columns:
                df = df.drop(self.target_column)

        self.features = df.columns

        # Example: remove known columns if they exist
        for col_to_remove in ("price", "signal"):
            if col_to_remove in self.features:
                self.features.remove(col_to_remove)
        if 'asset' in df.columns:
            df = df.drop('asset')
        if 'asset' in self.final_column:
            self.final_column.remove('asset')
        return df.select(list(set(self.final_column)))
    
    # ------------------------------------------------
    # Example placeholders for feature-engineering methods
    # (Implement your actual logic here.)
    # ------------------------------------------------
    def lag_df(self, df: pl.DataFrame) -> pl.DataFrame:
        # Example: create a lag of the target_column
        return df.with_columns(
            (pl.col(self.target_column).shift(1).alias(f"{self.target_column}_lag1"))
        )

    def rolling_df(self, df: pl.DataFrame) -> pl.DataFrame:
        # Example: rolling mean on the target column
        # (For demonstration, using an expression approach)
        window_size = 3
        return df.with_columns([
            pl.col(self.target_column).rolling_mean(window_size).alias(f"{self.target_column}_rm{window_size}")
        ])

    def percent_change_df(self, df: pl.DataFrame) -> pl.DataFrame:
        # Example: simple percentage change
        return df.with_columns(
            (
                (pl.col(self.target_column) - pl.col(self.target_column).shift(1)) 
                / pl.col(self.target_column).shift(1)
            ).alias(f"{self.target_column}_pct_change")
        )

    def rsi_df(self, df: pl.DataFrame) -> pl.DataFrame:
        # Placeholder for RSI logic
        return df  # implement your RSI calculation here
