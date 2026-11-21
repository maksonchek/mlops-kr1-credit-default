# Кредитный риск: CatBoost, MLflow и Docker
Привалов Максим Александрович, МПИ-25-1-1. MLOps, контрольная №1, задания 46–49.

Прогнозирование дефолта по кредитной карте на данных [Default of Credit Card Clients с Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset). В работе сравниваются LogisticRegression и CatBoost, эксперименты регистрируются в MLflow, выбранная модель доступна через FastAPI в Docker. Загрузка данных, подготовка признаков, обучение и оценка качества находятся в ноутбуке.

## Запуск API
Нужны Git и Docker. Обучать модель перед запуском не требуется.
```bash
git clone https://github.com/maksonchek/mlops-kr1-credit-default.git
cd mlops-kr1-credit-default
docker build -t mlops-credit-default .
docker run --rm -d --name mlops-credit -p 127.0.0.1:18081:8000 mlops-credit-default
```
Документация: http://127.0.0.1:18081/docs. Проверка из PowerShell:
```powershell
Invoke-RestMethod http://127.0.0.1:18081/health
Invoke-RestMethod http://127.0.0.1:18081/predict -Method Post -ContentType 'application/json' -Body (Get-Content example.json -Raw)
docker stop mlops-credit
```
`POST /predict` принимает 23 именованных поля, возвращает `default_probability`, `prediction` и `threshold`. Формат есть в `example.json` и `/docs`. ID и целевая метка не передаются. Коды категорий, возраст и статусы платежа передаются целыми числами. Пропуски и лишние поля отклоняются. `GET /health` подтверждает загрузку модели. Приложение работает от UID 10001; порт опубликован только на localhost.

## Данные и результаты
30 000 клиентов. Тест: 6001, development: 23999. Одинаковые векторы признаков объединены в группы и не пересекаются между train/test или частями фолда. Пять фолдов StratifiedGroupKFold; выбор модели по CV ROC-AUC. Разбиение позволяет оценить качество на отложенных клиентах той же исторической выборки; перенос на более поздний период отдельно не оценивался.

Выбрана `catboost_d6_lr0.05`. CV ROC-AUC: **0.78386**. Независимый тест: ROC-AUC **0.78176**, Gini **0.56353**, AP **0.56416**, Brier **0.13445**. Порог **0.26276** выбран по F1 на OOF development; тест при выборе не использовался. Тестовые precision **0.51813**, recall **0.58526**, F1 **0.54965**. Порог не выражает реальные банковские издержки. AP вычисляется как average precision.

`КР_1_Привалов_МПИ-25-1-1.ipynb` содержит выполненные расчёты, таблицы, графики и выводы. `cv_results.csv` — сравнение; `split.csv` — разбиение; `model_context.json` — признаки, порог и метрики; `model.joblib` — модель; `verification.json` — проверка восстановления; `container_verification.json` — фактическая проверка контейнера.

## Повторение обучения и MLflow
Python 3.10, CPU. Выполнить из корня проекта:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Выбрать созданное окружение в Jupyter и выполнить ноутбук сверху вниз. Датасет включён в репозиторий; если CSV отсутствует, ноутбук скачивает его с Kaggle. Эксперимент называется `MLOps_KR1_Credit_Default`. Локальные `mlflow.db`, `mlartifacts` и `.venv` не публикуются. Каждый повтор создаёт новые runs.
```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5051
```
После выполнения ноутбука результаты экспериментов доступны в интерфейсе MLflow по адресу http://127.0.0.1:5051. API загружает сохранённую модель из `model.joblib` и работает независимо от сервера MLflow.

## Источники и ограничения
Пример работы с Tracking API и сохранения моделей: [MLflow tutorial](https://github.com/mtpatter/mlflow-tutorial).

Данные: I-Cheng Yeh, [Default of Credit Card Clients, UCI](https://doi.org/10.24432/C55S3H), CC BY 4.0; CSV получен из Kaggle. Источник и SHA-256 — в `data_provenance.json`. Выборка содержит исторические данные клиентов из Тайваня. Перед использованием модели на другой популяции или за другой период необходима повторная оценка качества. Порог, выбранный по F1, требует пересмотра с учётом стоимости ошибок в конкретной задаче.

Версии зависимостей API закреплены в `requirements-api-lock-linux.txt`. Базовый образ `python:3.10-slim` указан тегом и может обновляться, поэтому будущая сборка контейнера может отличаться от проверенной.
