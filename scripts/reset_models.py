"""Delete the registered model ctr-model: every version and the champion alias.

Runs stay in MLflow, so a replay can start from an empty registry without losing the
experiment record. A missing registered model is not an error.
"""

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ctr_pipeline.config import REGISTERED_MODEL, get_settings


def main() -> None:
    tracking_uri = get_settings().mlflow_tracking_uri
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.delete_registered_model(REGISTERED_MODEL)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        print("no registered model")
        return
    print(f"deleted registered model {REGISTERED_MODEL}")


if __name__ == "__main__":
    main()
