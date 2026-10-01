#!/usr/bin/env bash
# Обновление rag-ved на VPS из свежего GitHub: pull → сборка → пересоздание контейнеров.
# Запуск: sudo bash deploy/redeploy.sh
set -euo pipefail
cd /opt/rag-ved

git pull

docker build -t rag-ved .

docker rm -f rag-ved rag-ved-bot 2>/dev/null || true

docker run -d --name rag-ved --restart unless-stopped --network n8n_default \
  --env-file /opt/rag-ved/deploy/.env \
  -v /opt/rag-ved/state:/app/state \
  -l traefik.enable=true \
  -l "traefik.http.routers.ragved.rule=Host(\`ask.cygnusweb.ru\`)" \
  -l traefik.http.routers.ragved.entrypoints=websecure \
  -l traefik.http.routers.ragved.tls=true \
  -l traefik.http.routers.ragved.tls.certresolver=mytlschallenge \
  -l traefik.http.services.ragved.loadbalancer.server.port=8002 \
  rag-ved

docker run -d --name rag-ved-bot --restart unless-stopped --network n8n_default \
  --env-file /opt/rag-ved/deploy/.env \
  -v /opt/rag-ved/state:/app/state \
  rag-ved python telegram_bot.py

echo "Готово. Проверка: sudo docker ps | grep rag-ved"