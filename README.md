# CronJob для классификации звонков

Основные параметры, пороги, названия таблиц скрыты с помощью ****

В models/config.json находятся пороги для модели, а также дополнительные параметры

В src представлен основной код:

1) обработка данных (call_preprocessing.py)
2) основной скрипт с загрузкой, скорингом и загрузкой в dwh (predict.py)
3) dag для запуска в airflow по cron (dag.py)
4) дополнительные файлы   

