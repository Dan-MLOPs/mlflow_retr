# Модуль загрузки конфигурационной информации
# 
# В этом модуле приведен упрощенный код, созданный исключительно в целях демонстрации узкого фунционала инструмента
# и не предлагается к использованию в каких-либо учебных или профессиональных задачах, за исключением той, которая
# решалась в рамках конкретного открытого урока.

import os
from dotenv import load_dotenv
load_dotenv()

mlflow_tracking_url = os.getenv('MLFLOW_TRACKING_URI')
mlflow_s3_endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
mlflow_s3_region=os.getenv('AWS_DEFAULT_REGION')