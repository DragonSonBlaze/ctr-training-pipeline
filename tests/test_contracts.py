from pyspark.sql.types import IntegerType, StringType

from ctr_pipeline.contracts import (
    FEATURE_COLUMNS,
    HOURLY_COLUMNS,
    MODEL_INPUT_COLUMNS,
    RAW_COLUMNS,
    RAW_SCHEMA,
)

EXPECTED_RAW_HEADER = (
    "id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,"
    "app_category,device_id,device_ip,device_model,device_type,device_conn_type,"
    "C14,C15,C16,C17,C18,C19,C20,C21"
)


def test_raw_schema_has_24_columns_in_file_order():
    assert RAW_COLUMNS == EXPECTED_RAW_HEADER.split(",")
    assert [field.name for field in RAW_SCHEMA.fields] == EXPECTED_RAW_HEADER.split(",")
    assert len(RAW_COLUMNS) == 24


def test_id_and_hour_are_strings():
    fields = {field.name: field.dataType for field in RAW_SCHEMA.fields}
    assert isinstance(fields["id"], StringType)
    assert isinstance(fields["hour"], StringType)
    assert isinstance(fields["click"], IntegerType)


def test_hourly_columns_add_derived_columns():
    assert HOURLY_COLUMNS == RAW_COLUMNS + ["hour_ts", "dt", "hr"]


def test_feature_columns():
    assert FEATURE_COLUMNS == [
        "app_prior_impr",
        "app_prior_clicks",
        "app_ctr_smooth",
        "site_prior_impr",
        "site_prior_clicks",
        "site_ctr_smooth",
        "device_prior_impr",
        "device_prior_clicks",
        "device_ctr_smooth",
        "global_prior_ctr",
    ]


def test_model_input_columns_are_the_prior_counts():
    assert MODEL_INPUT_COLUMNS == [
        "app_prior_impr",
        "app_prior_clicks",
        "site_prior_impr",
        "site_prior_clicks",
        "device_prior_impr",
        "device_prior_clicks",
    ]
