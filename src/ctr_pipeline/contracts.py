"""Column names and types for the raw CSV, the hourly tables and the features.

Every module that names a column imports it from here, so a rename happens once.
"""

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

_STRING_COLUMNS = {
    "id",
    "hour",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
}

# Listed in CSV file order. Nulls are not rejected here: the quality checks count
# them, so a bad day is reported rather than failing the read.
_RAW_COLUMN_NAMES = [
    "id",
    "click",
    "hour",
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
]

RAW_SCHEMA = StructType(
    [
        StructField(name, StringType() if name in _STRING_COLUMNS else IntegerType(), True)
        for name in _RAW_COLUMN_NAMES
    ]
)

RAW_COLUMNS: list[str] = list(_RAW_COLUMN_NAMES)

DERIVED_COLUMNS = ["hour_ts", "dt", "hr"]
PARTITION_COLUMNS = ["dt", "hr"]
HOURLY_COLUMNS = RAW_COLUMNS + DERIVED_COLUMNS

# Click history is kept per key. Device history uses the pair of IP and model
# because device_id is a placeholder on most rows.
HISTORY_KEYS: dict[str, list[str]] = {
    "app": ["app_id"],
    "site": ["site_id"],
    "device": ["device_ip", "device_model"],
}

FEATURE_COLUMNS: list[str] = [
    column
    for key in ("app", "site", "device")
    for column in (f"{key}_prior_impr", f"{key}_prior_clicks", f"{key}_ctr_smooth")
] + ["global_prior_ctr"]

MODEL_INPUT_COLUMNS: list[str] = [
    column
    for key in ("app", "site", "device")
    for column in (f"{key}_prior_impr", f"{key}_prior_clicks")
]

# Fields the hashed logistic model reads as categories. Not included: id (unique per row),
# device_id (a placeholder on 82.5% of rows), device_ip (millions of values, its history is
# already in the device features), hour and hour_ts (hr carries the hour of the day).
CATEGORICAL_FIELDS: list[str] = [
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "hr",
]

NUMERIC_FEATURES: list[str] = [
    "app_ctr_smooth",
    "site_ctr_smooth",
    "device_ctr_smooth",
    "app_prior_impr",
    "site_prior_impr",
    "device_prior_impr",
]

# Only the smoothed rates are compared day to day. The prior_impr counts grow by design
# (they are cumulative since FIRST_DAY), so a drift score on them would always fire.
DRIFT_FEATURES: list[str] = ["app_ctr_smooth", "site_ctr_smooth", "device_ctr_smooth"]

# Every column any model may read. Promotion reads these for every model type.
PREDICT_COLUMNS: list[str] = CATEGORICAL_FIELDS + [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FIELDS
]
