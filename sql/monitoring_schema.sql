CREATE TABLE IF NOT EXISTS pipeline_step_runs (
    id          BIGSERIAL PRIMARY KEY,
    step        TEXT NOT NULL,
    data_date   DATE,
    started_at  TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    duration_s  DOUBLE PRECISION NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('success', 'failed', 'skipped')),
    rows_in     BIGINT,
    rows_out    BIGINT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS data_quality_results (
    data_date  DATE NOT NULL,
    check_name TEXT NOT NULL,
    passed     BOOLEAN NOT NULL,
    observed   DOUBLE PRECISION,
    threshold  TEXT,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (data_date, check_name)
);

CREATE TABLE IF NOT EXISTS model_runs (
    mlflow_run_id     TEXT PRIMARY KEY,
    data_date         DATE NOT NULL,
    model_type        TEXT NOT NULL,
    model_version     INTEGER,
    dataset_version   TEXT NOT NULL,
    train_rows        BIGINT,
    eval_rows         BIGINT,
    auc               DOUBLE PRECISION,
    log_loss          DOUBLE PRECISION,
    mean_pred         DOUBLE PRECISION,
    actual_ctr        DOUBLE PRECISION,
    champion_log_loss DOUBLE PRECISION,
    promoted          BOOLEAN,
    recorded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feature_daily_stats (
    data_date     DATE NOT NULL,
    feature       TEXT NOT NULL,
    mean          DOUBLE PRECISION,
    p50           DOUBLE PRECISION,
    p95           DOUBLE PRECISION,
    null_fraction DOUBLE PRECISION,
    PRIMARY KEY (data_date, feature)
);

CREATE TABLE IF NOT EXISTS feature_drift (
    data_date      DATE NOT NULL,
    reference_date DATE NOT NULL,
    feature        TEXT NOT NULL,
    distance       DOUBLE PRECISION NOT NULL,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (data_date, feature)
);
