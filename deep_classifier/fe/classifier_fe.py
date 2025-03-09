##########
# Import #
##############################################################################

from typing import Optional, List, Union
import polars as pl

from .__base import BaseFE

###########
# Classes #
##############################################################################

class ClassifierFE(BaseFE):
    def __init__(
        self,
        control_column: str = "date",
        target_column: str = "target",
        label: Optional[str] = None,
        fe_name_list: Optional[List[str]] = None,
        unused_feature: Optional[List[str]] = None,
        percent_change_window: Union[int, List[int]] = [1, 2, 3, 4, 5, 9, 22, 30],
        percent_price_ema_window: Union[int, List[int]] = [7, 22],
        rolling_window: int = 3,
        rsi_period: int = 14
    ):
        """
        A specialized feature engineering class for classification tasks.
        All features are computed as relative values.

        Parameters
        ----------
        control_column : str
            The name of the datetime column to sort by and parse as datetime.
        target_column : str
            The name of the target column. For OHLC data, you can set this to "close".
        label : str, optional
            Another label if needed.
        fe_name_list : list of str, optional
            A list of feature-engineering method names to apply 
            (e.g., ["lag_df", "rolling_df", "percent_change_df", "rsi_df", "macd_df", "percent_price_ema_df"]).
        unused_feature : list of str, optional
            Columns you want to drop after feature engineering.
        percent_change_window : int or list of int, optional
            The window or list of windows to use for the percent change calculation.
        percent_price_ema_window : int or list of int, optional
            The EMA periods to use for the percentage difference between price and EMA.
        rolling_window : int, optional
            Window size for the rolling mean calculation.
        rsi_period : int, optional
            The period to use for the RSI calculation.
        """
        super().__init__(
            control_column=control_column,
            target_column=target_column,
            label=label,
            fe_name_list=fe_name_list,
            unused_feature=unused_feature
        )
        self.percent_change_window = percent_change_window
        self.percent_price_ema_window = percent_price_ema_window
        self.rolling_window = rolling_window
        self.rsi_period = rsi_period
        
        # To store names of final feature columns
        self.final_column = ['asset']  

    #############
    # Imdicator #
    ##############################################################################
    
    def lag_df(self, df: pl.DataFrame, periods: int = 1) -> pl.DataFrame:
        """
        Create a relative lag feature.
        Computes (price - price.shift(periods)) / price.shift(periods).
        """
        return df.with_columns(
            (
                (pl.col(self.target_column) - pl.col(self.target_column).shift(periods))
                / pl.col(self.target_column).shift(periods)
            ).alias(f"{self.target_column}_rel_lag{periods}")
        )
    
    ##############################################################################

    def rolling_df(self, df: pl.DataFrame, window_size: Optional[int] = None) -> pl.DataFrame:
        """
        Create a relative rolling mean feature.
        Computes (price / rolling_mean - 1), i.e. the relative deviation from the rolling mean.
        If window_size is not provided, uses self.rolling_window.
        """
        if window_size is None:
            window_size = self.rolling_window
        return df.with_columns(
            (
                pl.col(self.target_column) / pl.col(self.target_column).rolling_mean(window_size) - 1
            ).alias(f"{self.target_column}_rel_rm{window_size}")
        )

    ##############################################################################
    
    def percent_change_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate the percent change of the target column for one or more windows.
        For each window, computes:
            (price - price.shift(window)) / price.shift(window)
        A new column is added with the name:
            f"{target_column}_percent_change_{window}"
        """
        if isinstance(self.percent_change_window, list):
            windows = self.percent_change_window
        else:
            windows = [self.percent_change_window]
        
        for window in windows:
            col_name = f"{self.target_column}_percent_change_{window}"
            df = df.with_columns(
                (
                    (pl.col(self.target_column) - pl.col(self.target_column).shift(window))
                    / pl.col(self.target_column).shift(window)
                ).alias(col_name)
            )
            self.final_column.append(col_name)
        return df

    ##############################################################################
    
    def rsi_df(self, df: pl.DataFrame, period: Optional[int] = None) -> pl.DataFrame:
        """
        Calculate the RSI (Relative Strength Index) using a simple rolling mean approach.
        RSI is inherently relative (0-100).
        
        Steps:
            1) Compute price deltas.
            2) Separate gains and losses.
            3) Compute rolling averages with min_samples=period 
                so that the first (period-1) rows are null.
            4) Compute RSI = 100 - [100 / (1 + (avg_gain / avg_loss))].
        """
        if period is None:
            period = self.rsi_period
        price_col = self.target_column
        delta_col = "delta"
        gain_col = "gain"
        loss_col = "loss"
        avg_gain_col = "avg_gain"
        avg_loss_col = "avg_loss"
        rsi_col = "rsi"

        df = df.with_columns(
            (pl.col(price_col) - pl.col(price_col).shift(1)).alias(delta_col)
        )

        df = df.with_columns([
            pl.when(pl.col(delta_col) > 0).then(pl.col(delta_col)).otherwise(0.0).alias(gain_col),
            pl.when(pl.col(delta_col) < 0).then(-pl.col(delta_col)).otherwise(0.0).alias(loss_col)
        ])

        df = df.with_columns([
            pl.col(gain_col)
            .rolling_mean(window_size=period, min_samples=period)
            .alias(avg_gain_col),
            pl.col(loss_col)
            .rolling_mean(window_size=period, min_samples=period)
            .alias(avg_loss_col)
        ])

        df = df.with_columns(
            (
                100.0 - (100.0 / (1.0 + (pl.col(avg_gain_col) / pl.col(avg_loss_col))))
            ).alias(rsi_col)
        )
        
        df = df.drop([delta_col, gain_col, loss_col, avg_gain_col, avg_loss_col])
        self.final_column.append(rsi_col)
        return df

    ##############################################################################
    
    def ema_df(self, df: pl.DataFrame, period: int) -> pl.DataFrame:
        """
        Compute the Exponential Moving Average (EMA) for the target column over the specified period.
        This helper returns the absolute EMA, used in other calculations.
        """
        alpha = 2 / (period + 1)
        return df.with_columns(
            pl.col(self.target_column)
            .ewm_mean(alpha=alpha)
            .alias(f"ema_{period}")
        )

    ##############################################################################
    
    def macd_df(
            self, 
            df: pl.DataFrame, 
            short_period: int = 7, 
            long_period: int = 22
    ) -> pl.DataFrame:
        """
        Compute the MACD (Moving Average Convergence Divergence) indicator as a relative feature.
        First, compute short and long EMAs using ema_df.
        Then, compute absolute MACD = (short EMA - long EMA) and convert to relative MACD
        by dividing by the long EMA.
        """
        df = self.ema_df(df, period=short_period)
        df = self.ema_df(df, period=long_period)
        df = df.with_columns(
            (pl.col(f"ema_{short_period}") - pl.col(f"ema_{long_period}")).alias("macd")
        )
        self.final_column.append("rel_macd")
        return df.with_columns(
            (pl.col("macd") / pl.col(f"ema_{long_period}")).alias("rel_macd")
        )

    ##############################################################################
    
    def percent_price_ema_df(
            self, 
            df: pl.DataFrame,
            period: Optional[Union[int, List[int]]] = None
    ) -> pl.DataFrame:
        """
        Compute the percentage difference between the target price 
        and its EMA over the given period(s).
        Calculated as: ((price - EMA) / EMA) * 100.
        If period is a list, compute this for each period.
        The intermediate EMA column(s) are dropped so only the relative feature remains.
        """
        if period is None:
            period = self.percent_price_ema_window
        if not isinstance(period, list):
            periods = [period]
        else:
            periods = period
            
        for p in periods:
            df = self.ema_df(df, period=p)
            ema_col = f"ema_{p}"
            df = df.with_columns(
                (
                    (pl.col(self.target_column) - pl.col(ema_col)
                ) / pl.col(ema_col) * 100)
                .alias(f"percent_price_ema_{p}")
            )
            df = df.drop(ema_col)
            self.final_column.append(f"percent_price_ema_{p}")
            
        return df
    
    ##############################################################################
    
##################################################################################
