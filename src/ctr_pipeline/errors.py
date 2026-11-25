"""Exceptions the pipeline raises on purpose. Each one maps to a clear exit code."""


class PipelineError(Exception):
    """Base class for errors the pipeline raises on purpose."""


class SkipStep(PipelineError):
    """The step has nothing to do. The CLI exits with code 99, which Airflow reads as skipped."""


class QualityCheckFailed(PipelineError):
    """At least one data quality check failed for the day."""


class LoadVerificationError(PipelineError):
    """The hourly tables read back do not match the input they were written from."""


class MissingInputError(PipelineError):
    """An input the step needs is absent, such as the raw CSV or a day with no rows."""
