# Деплой rag-ved-logging на VPS (rag.cygnusweb.ru)

Сервер: `irina@72.56.94.48` (Ubuntu, тот же стек, что у Porta: Traefik + n8n).
Схема — та же, что у Porta: контейнер в сети `n8n_default`, наружу через **Traefik**
(лейблы, certresolver `mytlschallenge`), порты наружу не открываются.
Субдомен: **rag.cygnusweb.ru**.

Все команды на сервере выполняет Ирина (SSH-ключ с passphrase, Claude ключа не имеет).
Claude даёт команды по одной, смотрит на вывод и ведёт дальше.

---

## Шаг 0 (Ирина): DNS-запись

В панели управления DNS `cygnusweb.ru` добавить **A-запись**: `rag` → `72.56.94.48`
(как сделано для `porta` и `white`). Без неё Traefik не сможет выпустить сертификат.

## Шаг 1 (Ирина, с Mac): залить код на сервер

Сначала один раз создать папку и отдать её себе:

```bash
ssh irina@72.56.94.48 'sudo mkdir -p /opt/rag-ved && sudo chown irina /opt/rag-ved'
```

Потом залить:

```bash
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  --exclude '.DS_Store' --exclude '.env' --exclude '.venv/' --exclude '*.db' \
  --exclude '*.jsonl' --exclude 'КЛЮЧ_*' \
  -e ssh "/Users/irina/Desktop/ДЗ/ДЗмод8_9/rag-ved-logging/" \
  irina@72.56.94.48:/opt/rag-ved/
```

Хранилища (`*.db`, `*.jsonl`) не тащим — на сервере они появятся в томе.
`chroma_db/` (готовый индекс 165 чанков) тащим — он нужен как стартовый сид.

## Шаг 2 (Ирина, руками на сервере): ключи

```bash
ssh irina@72.56.94.48
cp /opt/rag-ved/deploy/.env.server.example /opt/rag-ved/deploy/.env
nano /opt/rag-ved/deploy/.env        # вписать OPENAI_API_KEY (DashScope) и придумать STATS_PASSWORD
chmod 600 /opt/rag-ved/deploy/.env
exit
```

*Ключ DashScope вставляет Ирина сама — в чат и в git он не пишется.*

## Шаг 3 (Ирина, на сервере): собрать и запустить контейнер

```bash
ssh irina@72.56.94.48 'set -e
# 1. сид готового векторного индекса в том
sudo mkdir -p /opt/rag-ved/state
sudo rm -rf /opt/rag-ved/state/chroma_db
sudo cp -r /opt/rag-ved/assistant_api/chroma_db /opt/rag-ved/state/chroma_db
# 2. образ + контейнер (лейблы Traefik — как у Porta)
cd /opt/rag-ved && sudo docker build -t rag-ved .
sudo docker rm -f rag-ved 2>/dev/null || true
sudo docker run -d --name rag-ved --restart unless-stopped --network n8n_default \
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

> Название Traefik-сети и certresolver проверяем на сервере:
> `sudo docker network ls | grep n8n` — сеть должна называться `n8n_default`
> (так же, как на старом сервере; если нет — подставить фактическое имя).
> Certresolver — тот же, что в лейблах работающих контейнеров Porta/n8n:
> `sudo docker inspect porta --format '{{json .Config.Labels}}' | grep -o 'certresolver=[a-z]*'`

Проверка (подождать ~30 с на выпуск сертификата):

```bash
curl -s https://rag.cygnusweb.ru/ | head -5
sudo docker logs rag-ved --tail 20
```

## Шаг 4 (проверка ДЗ-сценария)

1. Открыть `https://rag.cygnusweb.ru` → задать вопрос («Что такое аккредитив?»).
2. Сделать тот же вопрос ещё раз → в ответе источник «кеш».
3. Отправить пустой запрос (или >2000 символов) → сообщение об отклонении.
4. Открыть `https://rag.cygnusweb.ru/stats?key=ПАРОЛЬ` → статистика:
   запросы/принято/отклонено/кеш/длительность/токены.
5. Скриншот страницы статистики — для формы (руками, не MCP).

## Обновление кода (повторный деплой)

```bash
# Шаг 1 rsync (см. выше), затем:
ssh irina@72.56.94.48 'cd /opt/rag-ved && sudo docker build -t rag-ved . \
  && sudo docker rm -f rag-ved && sudo docker run -d ... '
```
(команда запуска — та же из шага 3; том `/opt/rag-ved/state` сохраняет вектора/логи)

## Откат / удаление

```bash
sudo docker rm -f rag-ved && sudo docker rmi rag-ved   # контейнер и образ
sudo rm -rf /opt/rag-ved                               # код + том со статистикой (аккуратно!)
```