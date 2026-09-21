# Деплой rag-ved-logging на VPS (rag.cygnusweb.ru)

Сервер: `root@5.129.214.194` (Ubuntu 24.04, 2 ГБ RAM + swap 2G).
Схема — та же, что у Porta: контейнер в сети `n8n_default`, наружу через **Traefik**
(лейблы, certresolver `mytlschallenge`), порты наружу не открываются.
Субдомен: **rag.cygnusweb.ru**.

---

## Шаг 0 (Ирина, один раз): DNS-запись

В панели управления DNS `cygnusweb.ru` добавить **A-запись**: `rag` → `5.129.214.194`
(как делалось для `porta`). Без неё Traefik не сможет выпустить сертификат.

## Шаг 1 (Claude): залить код на сервер

```bash
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  --exclude '.DS_Store' --exclude '.env' --exclude '*.db' --exclude '*.jsonl' \
  --exclude 'КЛЮЧ_*' \
  -e ssh "/Users/irina/Desktop/ДЗ/ДЗмод8_9/rag-ved-logging/" \
  root@5.129.214.194:/opt/rag-ved/
```

Хранилища (`*.db`, `*.jsonl`) не тащим — на сервере они появятся в томе.
`chroma_db/` (готовый индекс 165 чанков) тащим — он нужен как стартовый сид.

## Шаг 2 (Ирина, руками): ключ на сервере

На сервере создать `/opt/rag-ved/deploy/.env` из шаблона и вписать ключ:

```bash
ssh root@5.129.214.194
cp /opt/rag-ved/deploy/.env.server.example /opt/rag-ved/deploy/.env
nano /opt/rag-ved/deploy/.env        # вписать OPENAI_API_KEY и STATS_PASSWORD
chmod 600 /opt/rag-ved/deploy/.env
```

*Ключ DashScope вставляет Ирина сама — в чат и в git он не пишется.*

## Шаг 3 (Claude): собрать и запустить контейнер

```bash
ssh root@5.129.214.194 'set -e
# 1. сид готового векторного индекса в том
mkdir -p /opt/rag-ved/state
rm -rf /opt/rag-ved/state/chroma_db
cp -r /opt/rag-ved/assistant_api/chroma_db /opt/rag-ved/state/chroma_db
# 2. образ + контейнер (лейблы Traefik — как у Porta)
cd /opt/rag-ved && docker build -t rag-ved .
docker rm -f rag-ved 2>/dev/null || true
docker run -d --name rag-ved --restart unless-stopped --network n8n_default \
  --env-file /opt/rag-ved/deploy/.env \
  -v /opt/rag-ved/state:/app/state \
  -l traefik.enable=true \
  -l "traefik.http.routers.ragved.rule=Host(\`rag.cygnusweb.ru\`)" \
  -l traefik.http.routers.ragved.entrypoints=websecure \
  -l traefik.http.routers.ragved.tls=true \
  -l traefik.http.routers.ragved.tls.certresolver=mytlschallenge \
  -l traefik.http.services.ragved.loadbalancer.server.port=8002 \
  rag-ved'
```

Проверка:
```bash
curl -s https://rag.cygnusweb.ru/ | head -5
docker logs rag-ved --tail 20
```

## Шаг 4 (проверка ДЗ-сценария)

1. Открыть `https://rag.cygnusweb.ru` → задать вопрос («Что такое аккредитив?»).
2. Сделать тот же вопрос ещё раз → в ответе источник «кеш».
3. Отправить пустой запрос (или >2000 символов) → сообщение об отклонении.
4. Открыть `https://rag.cygnusweb.ru/stats?key=ПАРОЛЬ` → статистика:
   запросы/принято/отклонено/кеш/длительность/токены.
5. Скриншот страницы статистики — для формы.

## Обновление кода (повторный деплой)

```bash
# Шаг 1 rsync (см. выше), затем:
ssh root@5.129.214.194 'cd /opt/rag-ved && docker build -t rag-ved . && docker rm -f rag-ved && docker run -d ... '
```
(команда запуска — та же из шага 3; том `/opt/rag-ved/state` сохраняет вектора/логи)

## Откат / удаление

```bash
docker rm -f rag-ved && docker rmi rag-ved   # контейнер и образ
rm -rf /opt/rag-ved                          # код + том со статистикой (аккуратно!)
```

---

## Отзыв доступа (позже)

SSH-ключ `porta-deploy@irina-mac` в `/root/.ssh/authorized_keys` — тот же, что у Porta
(см. док `ДОСТУП_И_ДЕПЛОЙ.md` вне git). Отзыв одной командой:
`ssh root@5.129.214.194 'sed -i.bak "/porta-deploy@irina-mac/d" /root/.ssh/authorized_keys'`