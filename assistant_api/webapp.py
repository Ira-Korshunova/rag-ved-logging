"""
Веб-интерфейс RAG-ассистента по ВЭД — для запуска на сервере.

Поверхности (роли разделены):

    GET  /            — витрина: вопрос → ответ с источниками (без служебного)
    GET  /about       — как устроен ассистент: документы базы, конвейер,
                        анонимность логов
    GET  /admin       — панель оператора (вход по паролю формой, сессия-cookie):
                        загрузка документов, статистика, лента событий
    POST /ask         — обработка вопроса
    POST /admin/upload — загрузка документа в базу (инкрементальная индексация)
    GET  /stats?key=  — прежняя статистика под паролем (для скриншотов/ДЗ:
                        пароль STATS_PASSWORD)

Запуск локально:  python webapp.py
Запуск на сервере: gunicorn -w 1 -b 127.0.0.1:8002 webapp:app
"""

import os
import time
from collections import deque
from functools import wraps

from dotenv import load_dotenv
from flask import (Flask, request, render_template_string, redirect,
                   session, make_response)
from werkzeug.utils import secure_filename

from rag_pipeline import RAGPipeline

load_dotenv()

app = Flask(__name__)

# Секрет для cookie-сессии /admin: берём FLASK_SECRET, при его отсутствии
# выводим из STATS_PASSWORD (стабилен, пока стабилен пароль)
app.secret_key = os.getenv("FLASK_SECRET") or (
    "ragved-admin-" + os.getenv("STATS_PASSWORD", "local-dev"))
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # до 4 МБ на загрузку

DATA_DIR = os.getenv("DATA_DIR", "data")
ALLOWED_EXTENSIONS = (".txt", ".md")
MAX_UPLOAD_BYTES = 2 * 1024 * 1024

_pipeline = None
_ingesting = False   # замок: одна индексация одновременно (1 воркер gunicorn)


def get_pipeline() -> RAGPipeline:
    """Ленивая инициализация pipeline (один раз на процесс gunicorn)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path=os.getenv("CACHE_DB_PATH", "api_rag_cache.db"),
            data_file=DATA_DIR,
        )
    return _pipeline


# --------------------------------------------------------------- rate-limit
# Простая защита публичного входа в LLM: словарь счётчиков в памяти
# (1 воркер gunicorn — корректно; Redis под это не нужен)

_rate_minute = {}  # ip -> deque(timestamps)
_rate_day = {}     # ip -> deque(timestamps)
MAX_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "6"))
MAX_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "150"))


def _rate_hits(ip, store, window_sec, cap):
    now = time.time()
    dq = store.setdefault(ip, deque())
    while dq and dq[0] <= now - window_sec:
        dq.popleft()
    if len(dq) >= cap:
        return False
    dq.append(now)
    return True


def rate_limited(ip: str) -> bool:
    return (not _rate_hits(ip, _rate_minute, 60, MAX_PER_MINUTE)
            or not _rate_hits(ip, _rate_day, 86400, MAX_PER_DAY))


# --------------------------------------------------------------- auth /admin
def admin_ok() -> bool:
    return bool(session.get("admin_ok")) and bool(os.getenv("STATS_PASSWORD"))


def require_stats_password(view):
    """Простейший доступ к статистике по паролю (правило урока:
    разграничение доступа). Оставлен для /stats?key= (скриншоты/ДЗ)."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        password = os.getenv("STATS_PASSWORD")
        if not password:
            return "Статистика отключена (не задан STATS_PASSWORD).", 403
        entered = request.args.get("key", "")
        if entered != password:
            return ("Доступ закрыт. Добавьте ?key=пароль "
                    "(пароль в переменной STATS_PASSWORD на сервере)."), 401
        return view(*args, **kwargs)
    return wrapped


# --------------------------------------------------------------- примеры вопросов
EXAMPLES = [
    "Чем отличается CIF от FOB?",
    "Какие документы нужны для импорта?",
    "Что такое ИМ40?",
    "Что такое УНК и когда контракт ставится на учёт?",
]


def _safe_filename(name):
    """Имя файла для сохранения в data/: без путей и служебных символов."""
    name = os.path.basename(name or "")
    name = name.replace("..", "_").replace("/", "_").replace("\\", "_").strip()
    return name or ""


# --------------------------------------------------------------- шаблоны
# Единая стилистика — «тёплая карта Porta» (макеты предпросмотр_дизайна:
# главная.html и статистика.html, фон porta-map-warm-v3.png)
BASE_CSS = """
  :root {
    --ink: #3A2A18; --muted: #7A6A55; --line: rgba(139, 90, 43, .25);
    --card: rgba(251, 248, 241, .92); --card-line: rgba(255, 255, 255, .7);
    --shadow: 0 2px 16px rgba(60, 40, 20, .12);
    --brown: #77573A; --brown-soft: rgba(119, 87, 58, .35);
    --good: #4A6B2A; --danger: #A63226; --chip: rgba(251, 248, 241, .8);
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", sans-serif;
    max-width: 46rem; margin: 0 auto; padding: 2.2rem 1rem 3rem;
    color: var(--ink); line-height: 1.55; position: relative; min-height: 100vh;
  }
  body::before {
    content: ""; position: fixed; inset: 0; z-index: -1;
    background: url("/static/porta-map-warm-v3.png") center / cover no-repeat;
    opacity: .34; filter: saturate(95%) contrast(1.05);
  }
  h1 { font-size: 1.5rem; margin: 0; letter-spacing: -.01em; }
  h2 { font-size: 1.05rem; margin: 1.6rem 0 .5rem; }
  button, .btn {
    font: inherit; font-weight: 600; cursor: pointer; border: 0; border-radius: 6px;
    background: #DACDBE; color: #3A2A18; padding: .55rem 1.3rem; text-decoration: none;
  }
  button:hover, .btn:hover { background: #CDBFB0; }
  .badge {
    font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--brown); border: 1px solid var(--brown-soft);
    padding: .18rem .55rem; border-radius: 6px; background: rgba(251,248,241,.7);
  }
  .chip {
    font-size: .78rem; color: var(--muted); background: var(--chip);
    border: 1px solid var(--card-line); border-radius: 6px; padding: .2rem .65rem;
    text-decoration: none;
  }
  .chip b { color: var(--ink); font-weight: 600; }
  .card {
    background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 8px;
    padding: 1rem; box-shadow: var(--shadow);
  }
  .muted { color: var(--muted); }
  .good { color: var(--good); font-weight: 600; }
  a { color: var(--brown); text-underline-offset: 3px; }
"""
TABLE_CSS = """
  .tiles {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
    gap: .7rem; margin: 1.2rem 0 .4rem;
  }
  .tile {
    background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 8px; padding: .8rem .95rem;
    box-shadow: var(--shadow);
  }
  .tile .n { font-size: 1.5rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .tile .l { color: var(--muted); font-size: .78rem; margin-top: .15rem; }
  table {
    border-collapse: collapse; width: 100%; margin: .4rem 0 1.1rem;
    background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 8px; overflow: hidden;
    box-shadow: var(--shadow); font-variant-numeric: tabular-nums;
  }
  td, th { border-bottom: 1px solid rgba(139, 90, 43, .14); padding: .5rem .75rem; text-align: left; }
  tr:last-child td { border-bottom: 0; }
  th { background: rgba(139, 90, 43, .1); font-size: .82rem; font-weight: 600; color: var(--muted); }
  .reason {
    display: inline-block; font-size: .78rem; background: rgba(139, 90, 43, .1);
    color: var(--muted); border-radius: 6px; padding: .15rem .6rem; margin: .1rem .25rem .1rem 0;
  }
  .err-row td { background: rgba(166, 50, 38, .06); }
"""

PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ассистент по ВЭД</title>
<style>""" + BASE_CSS + """
  .lead { color: var(--muted); margin: .45rem 0 1.4rem; }
  .brand { display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap; }
  textarea {
    width: 100%; min-height: 5.2rem; resize: vertical; font: inherit;
    border: 1px solid var(--brown-soft); border-radius: 6px; padding: .7rem .85rem;
    background: rgba(255,255,255,.75); color: var(--ink);
  }
  textarea:focus { outline: 2px solid var(--brown); outline-offset: 1px; }
  .row { display: flex; align-items: center; gap: .8rem; margin-top: .7rem; flex-wrap: wrap; }
  .hint { color: var(--muted); font-size: .82rem; }
  .examples { display: flex; align-items: center; flex-wrap: wrap; gap: .45rem; margin-top: .9rem; }
  .examples .chip { cursor: pointer; font: inherit; font-size: .78rem; border: 1px solid var(--card-line); background: var(--chip); color: var(--muted); }
  .examples .chip:hover { background: #EFE9DE; }
  .examples .label { font-size: .82rem; color: var(--muted); }
  .examples form { display: inline; margin: 0; }
  .answer { background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 8px;
    padding: 1rem 1.1rem; white-space: pre-wrap; box-shadow: var(--shadow); }
  .meta { display: flex; gap: .45rem; margin-top: .55rem; flex-wrap: wrap; }
  .error { color: var(--danger); font-size: .95rem; }
  footer { margin-top: 2.2rem; color: var(--muted); font-size: .85rem; }
</style>
</head>
<body>
<div class="brand">
  <h1>ИИ-ассистент для ВЭД</h1>
  <a class="chip" href="/about">как работает</a>
</div>
<p class="lead">Таможня, Incoterms, ТН ВЭД, формы расчётов, валютный контроль —
  ответы по базе документов с указанием источников.</p>
<form class="card" method="post" action="/ask">
  <textarea name="query" placeholder="Ваш вопрос по ВЭД…" required>{{ query_text }}</textarea>
  <div class="row">
    <button type="submit">Спросить</button>
    <span class="hint">Ответ собирается строго по базе документов — под ответом видно, по каким именно</span>
  </div>
</form>

<div class="examples"><span class="label">Примеры:</span>
  {% for ex in examples %}
  <form method="post" action="/ask">
    <input type="hidden" name="query" value="{{ ex }}">
    <button type="submit" class="chip">{{ ex }}</button>
  </form>
  {% endfor %}
</div>

{% if answer %}
  <h2>Ответ</h2>
  {% if from_cache %}<div class="meta"><span class="chip good">из кеша — ответ мгновенный, без расхода токенов</span></div>{% endif %}
  <div class="answer">{{ answer }}</div>
  {% if sources %}
    <div class="meta"><span class="chip">источники: <b>{% for s in sources %}{{ s }}{% if not loop.last %}, {% endif %}{% endfor %}</b></span></div>
  {% endif %}
{% endif %}
{% if error %}
  <div class="meta" style="margin-top: 1rem"><p class="error">{{ error }}</p></div>
{% endif %}

<footer>
  Ассистент отвечает по документам своей базы; точность норм проверяйте
  по актуальной редакции НПА. <a href="/about">Как он работает →</a>
</footer>
</body>
</html>
"""

ABOUT_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>О ассистенте — как работает</title>
<style>""" + BASE_CSS + """
  .section { margin-top: 1.1rem; }
  ol, ul { margin: .3rem 0; padding-left: 1.3rem; }
  li { margin-bottom: .35rem; }
  h2 { font-size: 1.02rem; margin: 1.5rem 0 .45rem; }
</style>
</head>
<body>
<div class="brand">
  <h1>О ассистенте</h1>
</div>
<p class="lead">Отвечает на вопросы по внешнеэкономической деятельности:
таможенные процедуры ЕАЭС, Incoterms 2020, ТН ВЭД, формы международных
расчётов, валютный контроль (ФЗ-173), ТК ЕАЭС, ФЗ-289, решения ЕЭК.</p>

<h2>Как получается ответ</h2>
<div class="card section">
<ol>
  <li>вопрос превращается в вектор и ищет ближайшие фрагменты базы знаний
      (ChromaDB, локальные эмбеддинги BAAI/bge-m3);</li>
  <li>найденные фрагменты ({{ top_k }} шт.) передаются языковой модели как
      контекст;</li>
  <li>модель отвечает строго по переданному контексту.</li>
</ol>
</div>

<h2>Как доверять результату</h2>
<div class="card section">
<ul>
  <li><b>Источники под ответом</b> — видно, по каким документам базы он собран;</li>
  <li><b>Кеш</b> — одинаковые вопросы отвечаются мгновенно и одинаково
      (без повторной генерации);</li>
  <li>база — конспекты действующих норм; перед применением сверяйте с
      актуальной редакцией НПА.</li>
</ul>
</div>

<h2>Анонимность и логи</h2>
<div class="card section">
<ul>
  <li>каждый запрос проходит 5 контрольных точек: получен → принят/отклонён →
      обработка → ответ готов → отправлен;</li>
  <li>персональные данные (телефоны, email, номера карт и документов)
      маскируются <i>до</i> записи в лог;</li>
  <li>логи хранятся {{ retention_days }} дней и дальше чистятся автоматически;</li>
  <li>операторская статистика ({% if has_stats %}страница под паролем{% else %}отключена — не задан пароль{% endif %}).</li>
</ul>
</div>

<p><a href="/">← К ассистенту</a></p>
</body>
</html>
"""

LOGIN_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Панель оператора — вход</title>
<style>""" + BASE_CSS + """
  .wrap { max-width: 22rem; margin: 8vh auto 0; }
  input[type=password] {
    width: 100%; font: inherit; padding: .6rem .8rem; margin-top: .7rem;
    border: 1px solid var(--brown-soft); border-radius: 6px;
    background: rgba(255,255,255,.75); color: var(--ink);
  }
  input[type=password]:focus { outline: 2px solid var(--brown); outline-offset: 1px; }
  button { margin-top: .9rem; width: 100%; }
</style></head>
<body>
<div class="wrap">
<div class="card">
<h1>Панель оператора</h1>
{% if no_password %}<p class="muted">Панель отключена: не задан STATS_PASSWORD на сервере.</p>
{% else %}<p class="muted" style="font-size: .9rem">Вход по паролю (переменная STATS_PASSWORD на сервере).
{% if wrong %}<span style="color: var(--danger)">Неверный пароль.</span>{% endif %}</p>
<form method="post" action="/admin">
  <input type="password" name="password" placeholder="Пароль" required autofocus>
  <button type="submit">Войти</button>
</form>{% endif %}
<p style="margin-top: 1.2rem"><a href="/">← К ассистенту</a></p>
</div>
</div>
</body></html>
"""

ADMIN_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Панель оператора</title>
<style>""" + BASE_CSS + TABLE_CSS + """
  .brand { display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap; }
  .links { margin: .4rem 0 0; font-size: .88rem; }
  .msg { background: rgba(74, 107, 42, .12); border: 1px solid rgba(74, 107, 42, .3);
    color: var(--good); padding: .6rem .8rem; border-radius: 6px; margin: .7rem 0; }
  .bad-msg { background: rgba(166, 50, 38, .08); border: 1px solid rgba(166, 50, 38, .3);
    color: var(--danger); padding: .6rem .8rem; border-radius: 6px; margin: .7rem 0; }
  input[type=file] { font: inherit; font-size: .9rem; color: var(--ink); }
  h2 { font-size: 1.02rem; margin: 1.7rem 0 .55rem; }
</style></head>
<body>
<div class="brand">
  <h1>Панель оператора</h1>
  <span class="badge">база знаний · логи конвейера</span>
</div>
<p class="links"><a href="/">← витрина</a> · <a href="/admin?logout=1">выйти</a></p>

<h2>Пополнение базы</h2>
{% if ingest_msg %}<div class="msg">{{ ingest_msg }}</div>{% endif %}
{% if ingest_err %}<div class="bad-msg">{{ ingest_err }}</div>{% endif %}
<div class="card">
<p style="margin: .2rem 0 .7rem">Файл <b>.txt</b> или <b>.md</b> (до 2 МБ). Документ будет добавлен в базу
и доступен сразу после индексации; уже загруженные с тем же именем файлы
не дублируются.</p>
{% if ingesting %}
  <p class="muted">Сейчас идёт индексация предыдущего файла — обновите страницу
  через минуту.</p>
{% else %}
<form method="post" action="/admin/upload" enctype="multipart/form-data">
  <input type="file" name="doc" accept=".txt,.md" required>
  <p><button type="submit">Загрузить и индексировать</button></p>
</form>
{% endif %}
<p class="muted" style="margin: .2rem 0 .2rem">Всего в базе: {{ chunk_count }} чанков.</p>
</div>

<h2>Статистика за 7 дней</h2>
<div class="tiles">
  <div class="tile"><div class="n">{{ stats.total_requests }}</div><div class="l">запросов получено</div></div>
  <div class="tile"><div class="n good">{{ stats.accepted }}</div><div class="l">принято</div></div>
  <div class="tile"><div class="n">{{ stats.cache_share_pct }}%</div><div class="l">ответов из кеша</div></div>
  <div class="tile"><div class="n">{{ stats.avg_duration_ms or '—' }}<span style="font-size:.6em{% if stats.avg_duration_ms %}; margin-left:.2em{% endif %}">{% if stats.avg_duration_ms %}мс{% endif %}</span></div><div class="l">средняя длительность</div></div>
</div>
<table>
  <tr><th>Показатель</th><th>Значение</th></tr>
  <tr><td>Запросов получено</td><td>{{ stats.total_requests }}</td></tr>
  <tr><td>Принято</td><td class="good">{{ stats.accepted }}</td></tr>
  <tr><td>Отклонено</td><td>{{ stats.rejected }}
      {% for reason, n in stats.rejected_by_reason.items() %}<span class="reason">{{ reason }}: {{ n }}</span>{% endfor %}</td></tr>
  <tr><td>Запросов по источникам</td><td>
      {% for src, n in stats.by_source.items() %}{{ src }}: {{ n }}{% if not loop.last %}, {% endif %}{% endfor %}</td></tr>
  <tr><td>Ответов подготовлено</td><td>{{ stats.answered }}</td></tr>
  <tr><td>Из кеша</td><td>{{ stats.cache_hits }} ({{ stats.cache_share_pct }}%)</td></tr>
  <tr><td>Средняя длительность</td><td>{{ stats.avg_duration_ms or '—' }} мс</td></tr>
  <tr><td>Ошибок</td><td>{{ stats.errors }}</td></tr>
</table>

<h2>Последние события конвейера</h2>
{% if feed %}
<table>
  <tr><th>Время</th><th>Событие</th><th>Источник</th><th>Вопрос</th><th>Кеш</th><th>мс</th><th>Заметка</th></tr>
  {% for e in feed %}
  <tr {% if e.error %}class="err-row"{% endif %}>
    <td>{{ e.time }}</td>
    <td>{{ e.event }}</td>
    <td>{{ e.source }}</td>
    <td>{{ e.query or '—' }}</td>
    <td>{% if e.from_cache is not none %}{{ 'да' if e.from_cache else '—' }}{% else %}—{% endif %}</td>
    <td>{% if e.duration_ms %}{{ e.duration_ms }}{% else %}—{% endif %}</td>
    <td>{% if e.error %}⚠ {{ e.error }}{% elif e.reason %}{{ e.reason }}{% else %}—{% endif %}</td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p class="muted">События пока не записывались.</p>
{% endif %}

</body></html>
"""

STATS_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Статистика логов</title>
<style>""" + BASE_CSS + TABLE_CSS + """
  .brand { display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap; }
  h1 { font-size: 1.35rem; }
  h2 { font-size: 1.02rem; margin: 1.7rem 0 .55rem; }
</style>
</head>
<body>
<div class="brand">
  <h1>Статистика логов</h1>
  {% if stats.period_start %}<span class="badge">с {{ stats.period_start }} по {{ stats.period_end }}</span>
  {% else %}<span class="badge">за {{ stats.period_days }} дн.</span>{% endif %}
</div>

<div class="tiles">
  <div class="tile"><div class="n">{{ stats.total_requests }}</div><div class="l">запросов получено</div></div>
  <div class="tile"><div class="n good">{{ stats.accepted }}</div><div class="l">принято</div></div>
  <div class="tile"><div class="n">{{ stats.cache_share_pct }}%</div><div class="l">ответов из кеша</div></div>
  <div class="tile"><div class="n">{{ stats.avg_duration_ms or '—' }}<span style="font-size:.6em{% if stats.avg_duration_ms %}; margin-left:.2em{% endif %}">{% if stats.avg_duration_ms %}мс{% endif %}</span></div><div class="l">средняя длительность</div></div>
</div>

<h2>Конвейер запросов (5 событий урока)</h2>
<table>
  <tr><th>Показатель</th><th>Значение</th></tr>
  <tr><td>Запросов получено</td><td>{{ stats.total_requests }}</td></tr>
  <tr><td>Принято</td><td class="good">{{ stats.accepted }}</td></tr>
  <tr><td>Отклонено</td><td>{{ stats.rejected }}
      {% for reason, n in stats.rejected_by_reason.items() %}<span class="reason">{{ reason }}: {{ n }}</span>{% endfor %}</td></tr>
  <tr><td>Запросов по источникам</td><td>
      {% for src, n in stats.by_source.items() %}{{ src }}: {{ n }}{% if not loop.last %}, {% endif %}{% endfor %}</td></tr>
  <tr><td>Ответов подготовлено</td><td>{{ stats.answered }}</td></tr>
  <tr><td>Из кеша</td><td>{{ stats.cache_hits }} ({{ stats.cache_share_pct }}%)</td></tr>
  <tr><td>Средняя длительность</td><td>{{ stats.avg_duration_ms or '—' }} мс</td></tr>
  <tr><td>Ошибок</td><td>{{ stats.errors }}</td></tr>
</table>

<h2>Токены по моделям</h2>
<table>
  <tr><th>Модель</th><th>Prompt</th><th>Completion</th></tr>
  {% for model, tok in stats.tokens_by_model.items() %}
  <tr><td>{{ model }}</td><td>{{ tok.prompt }}</td><td>{{ tok.completion }}</td></tr>
  {% else %}
  <tr><td colspan="3">Нет данных за период</td></tr>
  {% endfor %}
</table>

<footer><a class="btn" href="/">← К ассистенту</a></footer>
</body>
</html>
"""


# --------------------------------------------------------------- витрина
@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE, examples=EXAMPLES,
                                  query_text="", answer=None, sources=None,
                                  from_cache=False, error=None)


@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("query", "")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()

    if rate_limited(ip):
        return render_template_string(
            PAGE, examples=EXAMPLES, query_text=query, answer=None,
            sources=None, from_cache=False,
            error="Слишком много вопросов подряд — подождите немного и попробуйте ещё."), 429

    ctx = {"examples": EXAMPLES, "query_text": query,
           "answer": None, "sources": None, "from_cache": False, "error": None}
    try:
        result = get_pipeline().query(query)
        from_cache = result.get("from_cache", False)
        # Источники показываются только для свежего ответа: в кеш попадает
        # текст документа без метаданных, у него source нет
        sources = None
        if not from_cache:
            sources = list(dict.fromkeys(
                d.get("source") for d in (result.get("context_docs") or [])
                if isinstance(d, dict) and d.get("source")))
        ctx.update(answer=result["answer"], sources=sources, from_cache=from_cache)
    except ValueError as ve:
        # Отклонение с причиной — уже залогировано в pipeline
        ctx["error"] = str(ve)
        return render_template_string(PAGE, **ctx)
    except Exception:
        ctx["error"] = "Не получилось получить ответ по базе — попробуйте другой вопрос."
        return render_template_string(PAGE, **ctx)
    return render_template_string(PAGE, **ctx)


@app.route("/about", methods=["GET"])
def about():
    pipeline = get_pipeline()
    return render_template_string(
        ABOUT_PAGE,
        top_k=pipeline.top_k,
        retention_days=os.getenv("LOG_RETENTION_DAYS", "90"),
        has_stats=bool(os.getenv("STATS_PASSWORD")),
    )


# --------------------------------------------------------------- панель оператора
@app.route("/admin", methods=["GET", "POST"])
def admin():
    if not os.getenv("STATS_PASSWORD"):
        return render_template_string(LOGIN_PAGE, no_password=True, wrong=False), 403

    # выход из панели
    if request.method == "GET" and request.args.get("logout"):
        session.pop("admin_ok", None)
        return redirect("/admin")

    if request.method == "POST":
        entered = request.form.get("password") or ""
        if entered == os.getenv("STATS_PASSWORD"):
            session["admin_ok"] = True
            return redirect("/admin")
        return render_template_string(LOGIN_PAGE, no_password=False, wrong=True), 401

    if not admin_ok():
        return render_template_string(LOGIN_PAGE, no_password=False, wrong=False)

    pipeline = get_pipeline()
    return render_template_string(
        ADMIN_PAGE,
        stats=pipeline.logger.get_stats(period_days=7),
        feed=pipeline.logger.get_recent(limit=25)["events"],
        chunk_count=pipeline.vector_store.get_collection_stats()["count"],
        ingest_msg=request.args.get("upload") or None,
        ingest_err=request.args.get("upload_err") or None,
        ingesting=_ingesting,
    )


@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    if not admin_ok():
        return redirect("/admin")

    global _ingesting
    upload = request.files.get("doc")
    name = _safe_filename(upload.filename) if upload else ""

    if upload and name:
        upload.seek(0, os.SEEK_END)
        size = upload.tell()
        upload.seek(0)
        if not name.lower().endswith(ALLOWED_EXTENSIONS):
            return redirect("/admin?upload_err=" +
                            "Загружаются файлы .txt и .md — этот формат для базы не подходит.")
        if size == 0 or size > MAX_UPLOAD_BYTES:
            return redirect("/admin?upload_err=" +
                            "Файл пустой либо больше 2 МБ (для конспекта этого больше, чем нужно).")

        target = os.path.join(DATA_DIR, name)
        if os.path.exists(target):
            return redirect("/admin?upload_err=" +
                            f"Файл «{name}» уже в базе. Если нужно заменить — сначала удалите старый с сервера.")
        upload.save(target)

        if _ingesting:
            return redirect("/admin?upload_err=" +
                            "Файл сохранён, но индексация занята — повторите через минуту.")

        _ingesting = True
        try:
            result = get_pipeline().vector_store.add_documents_from_folder(DATA_DIR)
            if result["added_files"] == 0:
                return redirect("/admin?upload=" +
                                f"Файл «{name}» уже был в базе — новых чанков не добавлено ({result['total']} чанков всего).")
            msg = (f"✅ База пополнена: {result['added_files']} новый(ых) файл(ов), "
                   f"+{result['added_chunks']} чанков. Всего в коллекции: {result['total']}. "
                   f"Задайте вопрос в форме «Спросить» — ответ придёт уже по новой базе.")
        except Exception as e:
            os.remove(target)  # неудачная индексация — файл в базу не вписался
            return redirect("/admin?upload_err=" +
                            f"Индексация «{name}» не удалась: {e}. Файл удалён.")
        finally:
            _ingesting = False
        return redirect("/admin?upload=" + msg)

    return redirect("/admin?upload_err=" + "Файл не выбран.")


# --------------------------------------------------------------- прежняя статистика
@app.route("/stats")
@require_stats_password
def stats():
    stats_data = get_pipeline().logger.get_stats(period_days=7)
    return render_template_string(STATS_PAGE, stats=stats_data)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8002, debug=False)