##########
# Import #
##############################################################################

import pytest
import polars as pl
import datetime

from deep_classifier.fe.classifier_fe import ClassifierFE

############
# Generate #
##############################################################################

def generate_timestamps(start_str: str, count: int, delta_minutes: int = 1):
    start = datetime.datetime.fromisoformat(start_str)
    return [
        (start + datetime.timedelta(minutes=i)).isoformat() \
            for i in range(count)
    ]

##############################################################################

def build_df(data, timestamps):
    """
    Build a simple DataFrame with control column "date" and target column "target".
    """
    return pl.DataFrame({
        "date": timestamps,
        "target": data
    })

##############################################################################

def generate_multi_asset_df():
    """
    Generate a simple DataFrame with data for multiple assets.
    Each asset has 5 rows. The DataFrame includes:
        - "date": ISO datetime string.
        - "target": A numeric value that varies by asset.
        - "asset": Asset identifier.
        - "price" and "signal": Dummy columns that should be dropped.
    """
    rows = []
    base_date = datetime.datetime(2023, 1, 1, 0, 0, 0)
    assets = ["Asset_A", "Asset_B", "Asset_C"]
    for asset in assets:
        for i in range(5):
            # Create a date offset by i minutes.
            date_str = (base_date + datetime.timedelta(minutes=i)).isoformat()
            # Define target values that differ by asset.
            if asset == "Asset_A":
                target = 100 + i * 10
            elif asset == "Asset_B":
                target = 200 + i * 5
            else:
                target = 50 + i * 20
            rows.append({
                "date": date_str,
                "target": target,
                "asset": asset,
                "price": target,    # Dummy column to be dropped.
                "signal": 1         # Dummy column to be dropped.
            })
    return pl.DataFrame(rows)

#############
# Test-case #
##############################################################################

def test_lag_df():
    # Create a simple series: 100, 105, 110, 120
    timestamps = generate_timestamps("2023-01-01T00:00:00", 4)
    data = [100, 105, 110, 120]
    df = build_df(data, timestamps)
    
    fe = ClassifierFE(control_column="date", target_column="target", fe_name_list=[])
    df_lag = fe.lag_df(df, periods=1)
    
    # Expected relative lag: row1: (105-100)/100 = 0.05, row2: (110-105)/105 ≈ 0.047619, row3: (120-110)/110 ≈ 0.090909
    expected = [None, 0.05, 0.047619047619047616, 0.09090909090909091]
    result = df_lag["target_rel_lag1"].to_list()
    
    assert result[0] is None, "Expected first lag value to be null."
    for i in range(1, len(expected)):
        assert abs(result[i] - expected[i]) < 1e-6, f"Relative lag at row {i} is incorrect."

##############################################################################
# Test relative rolling mean feature

def test_rolling_df():
    # Create a simple series: 100, 110, 120, 130, 140.
    timestamps = generate_timestamps("2023-01-01T00:00:00", 5)
    data = [100, 110, 120, 130, 140]
    df = build_df(data, timestamps)
    
    fe = ClassifierFE(control_column="date", target_column="target", fe_name_list=[])
    df_roll = fe.rolling_df(df, window_size=3)
    # Relative deviation = (price / rolling_mean - 1)
    result = df_roll["target_rel_rm3"].to_list()
    
    # For a 3-window, the first two rows should be null.
    assert result[0] is None, "First rolling relative value should be null."
    assert result[1] is None, "Second rolling relative value should be null."
    
    # For row index 2: rolling mean of [100,110,120] is 110 → (120/110 -1) ≈ 0.090909
    expected_row2 = (120/110) - 1
    expected_row3 = (130/((110+120+130)/3)) - 1
    expected_row4 = (140/((120+130+140)/3)) - 1
    
    assert abs(result[2] - expected_row2) < 1e-6, "Rolling relative value at row 2 is incorrect."
    assert abs(result[3] - expected_row3) < 1e-6, "Rolling relative value at row 3 is incorrect."
    assert abs(result[4] - expected_row4) < 1e-6, "Rolling relative value at row 4 is incorrect."

##############################################################################
# Test percent change feature for multiple windows

@pytest.mark.parametrize("windows, expected_columns", [
    ([1, 2], ["target_percent_change_1", "target_percent_change_2"]),
    (3, ["target_percent_change_3"])
])
def test_percent_change_df(windows, expected_columns):
    # Create a simple increasing series: 100, 105, 110, 115, 120.
    timestamps = generate_timestamps("2023-01-01T00:00:00", 5)
    data = [100, 105, 110, 115, 120]
    df = build_df(data, timestamps)
    
    fe = ClassifierFE(control_column="date", target_column="target", fe_name_list=[], percent_change_window=windows)
    df_pct = fe.percent_change_df(df)
    
    for col in expected_columns:
        assert col in df_pct.columns, f"Expected column {col} not found."
    
    # For window=1, expected relative changes: row1: (105-100)/100=0.05, etc.
    if "target_percent_change_1" in df_pct.columns:
        vals = df_pct["target_percent_change_1"].to_list()
        expected = [None, 0.05, 0.047619047619047616, 0.045454545454545456, 0.043478260869565216]
        for i, exp in enumerate(expected):
            if exp is None:
                assert vals[i] is None, f"Row {i} for window 1 should be null."
            else:
                assert abs(vals[i] - exp) < 1e-6, f"Window 1 percent change at row {i} is incorrect."

##############################################################################
# Test RSI feature

def test_rsi_df():
    # Create a dataset with some oscillation: [100, 90, 95, 100, 98, 102].
    timestamps = generate_timestamps("2023-01-01T00:00:00", 6)
    data = [100, 90, 95, 100, 98, 102]
    df = build_df(data, timestamps)
    
    fe = ClassifierFE(control_column="date", target_column="target", fe_name_list=[], rsi_period=3)
    df_rsi = fe.rsi_df(df, period=3)
    res = df_rsi["rsi"].to_list()
    
    # For period=3, first 2 rows are null.
    assert res[0] is None and res[1] is None, "First two RSI values should be null."
    # Remaining RSI values should be between 0 and 100.
    for v in res[2:]:
        assert 0 <= v <= 100, f"RSI value {v} out of bounds."

##############################################################################
# Test MACD feature

def test_macd_df():
    # Create a simple series: 100, 102, 104, 106, 108, 110.
    timestamps = generate_timestamps("2023-01-01T00:00:00", 6)
    data = [100, 102, 104, 106, 108, 110]
    df = build_df(data, timestamps)
    
    fe = ClassifierFE(control_column="date", target_column="target", fe_name_list=[])
    df_macd = fe.macd_df(df, short_period=3, long_period=5)
    # Check that the relative MACD column exists.
    assert "rel_macd" in df_macd.columns, "Relative MACD column 'rel_macd' not found."
    
    # Check that MACD values are floats (and not None).
    res = df_macd["rel_macd"].to_list()
    for val in res:
        if val is not None:
            assert isinstance(val, float), "MACD value is not a float."


##############################################################################
# Test Percent Price-EMA feature for multiple periods

def test_percent_price_ema_df():
    # Create a simple series: 100, 102, 104, 106, 108, 110, 112, 114, 116, 118.
    timestamps = generate_timestamps("2023-01-01T00:00:00", 10)
    data = [100 + 2 * i for i in range(10)]
    df = build_df(data, timestamps)
    
    # Use multiple EMA periods: [3, 5]
    fe = ClassifierFE(
        control_column="date", 
        target_column="target", 
        fe_name_list=[], 
        percent_price_ema_window=[3, 5]
    )
    df_pct_ema = fe.percent_price_ema_df(df)
    
    # Check that both expected columns are present.
    assert "percent_price_ema_3" in df_pct_ema.columns, "Column 'percent_price_ema_3' missing."
    assert "percent_price_ema_5" in df_pct_ema.columns, "Column 'percent_price_ema_5' missing."
    
    # Check that no intermediate EMA columns remain.
    for col in df_pct_ema.columns:
        assert not col.startswith("ema_"), "Intermediate EMA column found; should be dropped."
    
    # Optionally, check that computed percentage differences are floats.
    for col in ["percent_price_ema_3", "percent_price_ema_5"]:
        vals = df_pct_ema[col].to_list()
        for v in vals:
            if v is not None:
                assert isinstance(v, float), f"Value in {col} is not a float."
                
##############################################################################

def test_transform_df_multi_asset():
    df = generate_multi_asset_df()
    
    # Instantiate the feature engineering class with multiple asset support.
    fe = ClassifierFE(
        control_column="date",
        target_column="target",
        fe_name_list=[
            "lag_df",
            "rolling_df",
            "percent_change_df",
            "rsi_df",
            "macd_df",
            "percent_price_ema_df"
        ],
        unused_feature=["price", "signal"],
        percent_change_window=[1, 2],
        percent_price_ema_window=[7, 22],
        rolling_window=3,
        rsi_period=3
    )
    
    # Transform the DataFrame. The transform_df method processes each asset separately 
    # and concatenates the results.
    df_transformed = fe.transform_df(df, keep_target=True, keep_control=True)
    
    # The final DataFrame should contain only the columns stored in fe.final_column.
    final_features = set(fe.final_column)
    output_columns = set(df_transformed.columns)
    assert output_columns == final_features, f"Output columns {output_columns} do not match expected features {final_features}"
    
    # Check that the transformed DataFrame is not empty.
    assert df_transformed.height > 0, "Transformed DataFrame is empty."
    
    # Verify that there are no nulls in the final feature columns.
    for col in df_transformed.columns:
        null_count = df_transformed.select(pl.col(col).is_null().sum())[col][0]
        assert null_count == 0, f"Column '{col}' has {null_count} null values."

##############################################################################

def test_add_label():
    """
    Test that the add_label method correctly computes labels based on the change in 'target'
    and drops the first row.
    
    For example, given target = [100, 102, 101, 105]:
        - Row 1: 102-100 = 2 (>=0) → "LONG"
        - Row 2: 101-102 = -1 (<0) → "SHORT"
        - Row 3: 105-101 = 4 (>=0) → "LONG"
    
    The first row is dropped, so we expect the resulting label column to be:
        ["LONG", "SHORT", "LONG"]
    """
    # Create a DataFrame with 4 rows.
    timestamps = generate_timestamps("2023-01-01T00:00:00", 4)
    data = [100, 102, 101, 105]
    df = build_df(data, timestamps)
    
    # Instantiate the feature engineering class.
    fe = ClassifierFE(control_column="date", target_column="target", fe_name_list=[])
    
    # Add label using threshold 0.
    df_with_label = fe.add_binary_label(df, threshold=0.0)
    
    # The resulting DataFrame should have 3 rows (first row dropped).
    assert df_with_label.height == 3, f"Expected 6 rows after dropping first row, got {df_with_label.height} | {df_with_label}."
    
    # Check that the 'label' column exists.
    assert "label" in df_with_label.columns, "The 'label' column is missing."
    
    # Expected labels: Row1: LONG, Row2: SHORT, Row3: LONG.
    expected_labels = ["LONG", "SHORT", "LONG"]
    result_labels = df_with_label["label"].to_list()
    assert result_labels == expected_labels, f"Expected labels {expected_labels}, got {result_labels}."

##############################################################################

def build_single_asset_df():
    """
    Build a simple single-asset DataFrame without an "asset" column.
    For target values: [100, 102, 101, 105]
        - shift(-1) gives: [102, 101, 105, null]
        - diff: [2, -1, 4, null]
        - Expected labels (with threshold 0.0): ["LONG", "SHORT", "LONG"]
    """
    timestamps = generate_timestamps("2023-01-01T00:00:00", 4)
    data = [100, 102, 101, 105]
    return pl.DataFrame({
        "date": timestamps,
        "target": data
    })

def build_multi_asset_df():
    """
    Build a multi-asset DataFrame with two assets: "X" and "Y".
    Each asset has 4 rows.
    
    Asset X: target = [100, 102, 101, 105]
        - shift(-1) yields diff: [2, -1, 4, null] → Expected labels: ["LONG", "SHORT", "LONG"]
    
    Asset Y: target = [200, 195, 205, 210]
        - shift(-1) yields diff: [-5, 10, 5, null] → Expected labels: ["SHORT", "LONG", "LONG"]
    """
    rows = []
    # Asset X
    timestamps_X = generate_timestamps("2023-01-01T00:00:00", 4)
    targets_X = [100, 102, 101, 105]
    for t, target in zip(timestamps_X, targets_X):
        rows.append({
            "date": t,
            "target": target,
            "asset": "X"
        })
    # Asset Y
    timestamps_Y = generate_timestamps("2023-01-01T00:05:00", 4)
    targets_Y = [200, 195, 205, 210]
    for t, target in zip(timestamps_Y, targets_Y):
        rows.append({
            "date": t,
            "target": target,
            "asset": "Y"
        })
    return pl.DataFrame(rows)

##############################################################################

def test_add_binary_label_single_asset():
    """
    Test add_binary_label on a single-asset DataFrame (without an "asset" column).
    Expected: For target [100, 102, 101, 105]:
        - diff: [2, -1, 4, null]
        - Labels: ["LONG", "SHORT", "LONG"]
    """
    df = build_single_asset_df()
    fe = ClassifierFE(control_column="date", target_column="target", fe_name_list=[])
    # When no asset column is present, the else branch is executed.
    df_labeled = fe.add_binary_label(df, threshold=0.0)
    # We explicitly drop the first row inside add_binary_label.
    result_labels = df_labeled.sort("date")["label"].to_list()
    expected_labels = ["LONG", "SHORT", "LONG"]
    assert result_labels == expected_labels, f"Expected {expected_labels}, got {result_labels}"

##############################################################################

def test_add_binary_label_multi_asset():
    """
    Test add_binary_label on a multi-asset DataFrame.
    For each asset, the last row is dropped due to shift(-1) producing null diff.
    Expected:
        - Asset X: labels = ["LONG", "SHORT", "LONG"]
        - Asset Y: labels = ["SHORT", "LONG", "LONG"]
    """
    df = build_multi_asset_df()
    fe = ClassifierFE(control_column="date", target_column="target", fe_name_list=["add_binary_label"])
    # Since multiple assets exist, the method should process each asset separately.
    df_labeled = fe.add_binary_label(df, threshold=0.0)
    # Sort the result by asset and date for predictable order.
    df_labeled = df_labeled.sort(["asset", "date"])
    
    pdf = df_labeled.to_pandas()
    expected = {
        "X": ["LONG", "SHORT", "LONG"],
        "Y": ["SHORT", "LONG", "LONG"]
    }
    
    for asset, exp_labels in expected.items():
        asset_df = pdf[pdf["asset"] == asset]
        result_labels = asset_df["label"].tolist()
        assert result_labels == exp_labels, f"For asset {asset}, expected {exp_labels}, got {result_labels}"

##############################################################################
