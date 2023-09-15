import logging
from datetime import datetime

from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator

from airml.k8s import set_exec_config
from ********** import runner, settings

logger = logging.getLogger(__name__)

worker_image = '************'
worker_tag = '*****'

conn_info = BaseHook.get_connection(conn_id='********')
dwh_conn = 'oracle+cx_oracle://{login}:{password}@{dsn}/?service_name=dwh&{encoding}'.format(
    login=conn_info.login,
    password=conn_info.password,
    dsn='********:1521',
    encoding='encoding=UTF-8&nencoding=UTF-8',
)


config = settings.load()
config["conn"] = dwh_conn

# declare default arguments for the dag
default_args = {
    'owner': '*********',
    'depends_on_past': False,
    'start_date': datetime.strptime('2022-05-03T10:00:00', '%Y-%m-%dT%H:%M:%S'),
    'provide_context': False,
    'retries': 0,
}

# create a new dag
dag = DAG(
    dag_id='**********',
    default_args=default_args,
    schedule_interval='0 3 * * *',
    max_active_runs=1,
    catchup=False,
)


predict = PythonOperator(
    task_id='predict',
    python_callable=runner.predict,
    op_kwargs={'config': config},
    executor_config=set_exec_config(mem='4Gi', cpu='1', image=f'{worker_image}:{worker_tag}'),
    dag=dag,
)

# Set the task sequence
predict
