# Данный модуль это дополненная и расширенная вресия примера, который находится по адресу https://github.com/mlflow/mlflow-example

# В этом модуле приведен упрощенный код, созданный исключительно в целях демонстрации узкого фунционала инструмента
# и не предлагается к использованию в каких-либо учебных или профессиональных задачах, за исключением той, которая
# решалась в рамках конкретного открытого урока.

# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import settings
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    logger.info(f"Mlflow tracking address: {settings.mlflow_tracking_url}")
    logger.info(f"Region for S3 storage: {settings.mlflow_s3_region}")
    logger.info(f"Mlflow Default Artifact Storage: {settings.mlflow_s3_endpoint_url}")

    # Read the wine-quality csv file from the URL
    # csv_url = (
    #     "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/data/winequality-red.csv"
    # )

    # Read the first half of the data from S3 bucket
    # csv_url = (
    #     "s3://mlflow-test/data/wine-quality_1st_half.csv"
    # )

     # Read the wine-quality csv file from S3 bucket
    csv_url = (
        "s3://mlflow-test/data/wine-quality.csv"
    )

    try:
        # data = pd.read_csv(csv_url, sep=";")
        data = pd.read_csv(
                    csv_url,
                    storage_options={
                        "client_kwargs": {
                            "endpoint_url": "https://storage.yandexcloud.net"
                        }
                    }
        )
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # define a new experiment
    experiment_name = 'Mlflow_Demo_2023' 
    # return experiment ID
    try:
        # try create a new experiment
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        # or use an existing
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # define the run name
    run_name="Test_with_complete_set_of_data"

    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("data loc", csv_url) # log traning data url / location info
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel_Demo",
            conda_env='conda_environment.yaml')
        else:
            mlflow.sklearn.log_model(lr, "model")
