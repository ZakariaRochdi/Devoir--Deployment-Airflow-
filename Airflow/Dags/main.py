import os
from datetime import datetime, timedelta
from airflow import DAG, Dataset
from airflow.operators.python import PythonOperator

from Dags.scraper import scrap_main
from Dags.data_cleaning import preprocess_main
from modeling import modeling_main
from Dags.exploratory_analysis import plots_main


base_dir = os.path.dirname(os.path.abspath(__file__))


video_dataset = Dataset(f"{base_dir}/data/video_data.csv")
comment_dataset = Dataset(f"{base_dir}/data/comment_data.csv")


pp_video_dataset = Dataset(f"{base_dir}/data/pp_video_data.csv")
pp_comment_dataset = Dataset(f"{base_dir}/data/pp_comment_data.csv")


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="1_youtube_scraper",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["youtube", "scraping"],
) as scrap_dag:

    t1_scrap = PythonOperator(
        task_id="scrap_youtube_data",
        python_callable=scrap_main,
        outlets=[video_dataset, comment_dataset],
    )


with DAG(
    dag_id="2_youtube_processor",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=[video_dataset, comment_dataset],
    catchup=False,
    tags=["youtube", "processing"],
) as process_dag:

    t1_process = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_main,
        outlets=[pp_video_dataset, pp_comment_dataset],
    )


with DAG(
    dag_id="3_youtube_analytics",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=[pp_video_dataset, pp_comment_dataset],
    catchup=False,
    tags=["youtube", "ml", "viz"],
) as analysis_dag:

    t1_model = PythonOperator(
        task_id="train_model",
        python_callable=modeling_main,
    )

    t2_plot = PythonOperator(
        task_id="generate_plots",
        python_callable=plots_main,
    )
