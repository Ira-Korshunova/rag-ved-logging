# RAG-ассистент по ВЭД с логированием (ДЗ мод 8.9)
FROM python:3.11-slim

WORKDIR /app

# Сначала зависимости — слой кешируется при пересборках
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY assistant_api/ .

# Хранилища (вектора, кеш, логи) вынесены в /app/state — там монтируется Docker-том
RUN mkdir -p /app/state

EXPOSE 8002

# 1 воркер: экономия RAM на VPS 2ГБ + предсказуемое поведение кеша/логов
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8002", "--timeout", "180", "webapp:app"]