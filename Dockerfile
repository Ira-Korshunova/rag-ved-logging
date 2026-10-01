# RAG-ассистент по ВЭД с логированием (ДЗ мод 8.9)
FROM python:3.11-slim

WORKDIR /app

# Сначала CPU-версия torch: sentence-transformers тянет полноценный torch
# с CUDA-библиотеками (~2,3 ГБ), которые на сервере без GPU бесполезны и
# рвали сборку по месту ("no space left on device"). CPU-вёрл ~200 МБ.
# Индекс pytorch — официальный (download.pytorch.org/whl/cpu).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Затем продуктовые зависимости (torch уже стоит — pip его не перекачает)
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY assistant_api/ .

# Хранилища (вектора, кеш, логи) вынесены в /app/state — там монтируется Docker-том
RUN mkdir -p /app/state

EXPOSE 8002

# 1 воркер: экономия RAM на VPS 2ГБ + предсказуемое поведение кеша/логов
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8002", "--timeout", "180", "webapp:app"]