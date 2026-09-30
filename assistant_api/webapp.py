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
PAGE = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ассистент по ВЭД</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #222; }
  header { display: flex; justify-content: space-between; align-items: baseline; }
  h1 { font-size: 1.4rem; margin-bottom: .2rem; }
  .sub { color: #666; margin-top: 0; }
  textarea, button { font-size: 1rem; }
  textarea { width: 100%; box-sizing: border-box; min-height: 90px; }
  button { padding: .5rem 1.2rem; cursor: pointer; }
  .examples { margin-top: .6rem; display: flex; flex-wrap: wrap; gap: .5rem; }
  .examples button { padding: .35rem .8rem; font-size: .9rem; border-radius: 16px; background: #eee; border: 1px solid #ddd; }
  .answer { background: #f5f5f0; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }
  .sources { color: #555; font-size: .88rem; margin-top: .6rem; }
  .sources code { background: #eee; padding: .1rem .4rem; border-radius: 4px; }
  .meta { color: #777; font-size: .85rem; margin-top: .5rem; }
  .badge { display: inline-block; background: #e7f0e7; color: #2a6b2a; font-size: .78rem; padding: .1rem .55rem; border-radius: 10px; }
  .error { color: #b00; font-size: .95rem; }
  a { color: #06c; }
  footer { margin-top: 2rem; color: #999; font-size: .85rem; }
</style>
</head>
<body>
<header>
  <h1>Ассистент по ВЭД</h1>
  <a href="/about">О ассистенте</a>
</header>
<p class="sub">Таможенные процедуры, Incoterms, ТН ВЭД, формы расчётов, валютный
контроль. Отвечает по базе документов — под ответом видно, по каким именно.</p>

<form method="post" action="/ask">
  <textarea name="query" placeholder="Ваш вопрос по ВЭД..." required>{{ query_text }}</textarea>
  <p><button type="submit">Спросить</button></p>
</form>

<div class="examples">Примеры:
  {% for ex in examples %}
  <form method="post" action="/ask" style="display:inline">
    <input type="hidden" name="query" value="{{ ex }}">
    <button type="submit">{{ ex }}</button>
  </form>
  {% endfor %}
</div>

{% if answer %}
  <h2>Ответ</h2>
  {% if from_cache %}<p class="meta"><span class="badge">из кеша — ответ быстрее и без расходов токенов</span></p>{% endif %}
  <div class="answer">{{ answer }}</div>
  {% if sources %}
    <p class="sources">Ответ собран по документам базы:
      {% for s in sources %}<code>{{ s }}</code>{% if not loop.last %}, {% endif %}{% endfor %}</p>
  {% endif %}
{% endif %}
{% if error %}<p class="error">{{ error }}</p>{% endif %}

<footer>
  Ассистент отвечает по документам своей базы; точность норм проверяйте
  по актуальной редакции НПА. <a href="/about">Как он работает →</a>
</footer>
</body>
</html>
"""

ABOUT_PAGE = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>О ассистенте</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #222; }
  h2 { font-size: 1.1rem; margin-top: 1.4rem; }
  li { margin-bottom: .35rem; }
  .muted { color: #666; }
  a { color: #06c; }
</style>
</head>
<body>
<h1>О ассистенте</h1>
<p>Ассистент отвечает на вопросы по внешнеэкономической деятельности —
таможенные процедуры ЕАЭС, Incoterms 2020, ТН ВЭД, формы международных
расчётов, валютный контроль (ФЗ-173), ТК ЕАЭС, ФЗ-289, решения ЕЭК.</p>

<h2>Как получается ответ</h2>
<ol>
  <li>вопрос превращается в вектор и ищет ближайшие фрагменты базы знаний
      (ChromaDB, локальные эмбеддинги BAAI/bge-m3);</li>
  <li>найденные фрагменты ({{ top_k }} шт.) передаются языковой модели как
      контекст;</li>
  <li>модель отвечает строго по переданному контексту.</li>
</ol>

<h2>Как доверять результату</h2>
<ul>
  <li><b>Источники под ответом</b> — видно, по каким документам базы он собран;</li>
  <li><b>Кеш</b> — одинаковые вопросы отвечаются мгновенно и одинаково
      (без повторной генерации);</li>
  <li>база — конспекты действующих норм; перед применением сверяйте с
      актуальной редакцией НПА.</li>
</ul>

<h2>Анонимность и логи</h2>
<ul>
  <li>каждый запрос проходит 5 контрольных точек: получен → принят/отклонён →
      обработка → ответ готов → отправлен;</li>
  <li>персональные данные (телефоны, email, номера карт и документов)
      маскируются <i>до</i> записи в лог;</li>
  <li>логи хранятся {{ retention_days }} дней и дальше чистятся автоматически;</li>
  <li>операторская статистика ({% if has_stats %}страница под паролем{% else %}отключена — не задан пароль{% endif %}).</li>
</ul>

<p class="muted"><a href="/">← К ассистенту</a></p>
</body>
</html>
"""

LOGIN_PAGE = """
<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Панель оператора — вход</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 420px; margin: 3rem auto; padding: 0 1rem; }
  input { padding: .5rem; font-size: 1rem; width: 100%; box-sizing: border-box; }
  button { font-size: 1rem; padding: .5rem 1.2rem; margin-top: .6rem; cursor: pointer; }
  .err { color: #b00; }
</style></head>
<body>
<h2>Панель оператора</h2>
{% if no_password %}<p class="err">Панель отключена: не задан STATS_PASSWORD на сервере.</p>
{% else %}Вход по паролю (переменная STATS_PASSWORD на сервере).
{% if wrong %}<p class="err">Неверный пароль.</p>{% endif %}
<form method="post" action="/admin">
  <input type="password" name="password" placeholder="Пароль" required autofocus>
  <p><button type="submit">Войти</button></p>
</form>{% endif %}
<p><a href="/">← К ассистенту</a></p>
</body></html>
"""

ADMIN_PAGE = """
<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Панель оператора</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 780px; margin: 2rem auto; padding: 0 1rem; }
  h2 { font-size: 1.1rem; margin-top: 1.6rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }
  table { border-collapse: collapse; width: 100%; margin: .6rem 0 1rem; }
  td, th { border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; font-size: .9rem; }
  th { background: #f0f0ea; }
  input[type=file] { font-size: .9rem; }
  button { padding: .5rem 1.2rem; cursor: pointer; font-size: .95rem; }
  .msg { background: #eef5ee; padding: .6rem .8rem; border-radius: 6px; margin-bottom: .6rem; }
  .bad-msg { background: #f9eaea; color: #a00; padding: .6rem .8rem; border-radius: 6px; margin-bottom: .6rem; }
  .err-row { background: #fdf2f2; }
  .muted { color: #888; font-size: .85rem; }
</style></head>
<body>
<h1>Панель оператора</h1>
<p class="muted"><a href="/">← витрина</a> · <a href="/admin?logout=1">выйти</a></p>

<h2>Пополнение базы</h2>
{% if ingest_msg %}<div class="msg">{{ ingest_msg }}</div>{% endif %}
{% if ingest_err %}<div class="bad-msg">{{ ingest_err }}</div>{% endif %}
<p>Файл <b>.txt</b> или <b>.md</b> (до 2 МБ). Документ будет добавлен в базу
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
<p class="muted">Всего в базе: {{ chunk_count }} чанков.</p>

<h2>Статистика за 7 дней</h2>
<table>
  <tr><th>Показатель</th><th>Значение</th></tr>
  <tr><td>Запросов получено</td><td>{{ stats.total_requests }}</td></tr>
  <tr><td>Принято</td><td>{{ stats.accepted }}</td></tr>
  <tr><td>Отклонено</td><td>{{ stats.rejected }}
      {% for reason, n in stats.rejected_by_reason.items() %} — {{ reason }}: {{ n }}{% endfor %}</td></tr>
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

STATS_PAGE = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Статистика</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  table { border-collapse: collapse; width: 100%; margin: .6rem 0 1.2rem; }
  td, th { border: 1px solid #ddd; padding: .45rem .7rem; text-align: left; }
  th { background: #f0f0ea; }
  h2 { font-size: 1.1rem; }
</style>
</head>
<body>
<h1>Статистика логов {% if stats.period_start %}(с {{ stats.period_start }} по {{ stats.period_end }}){% else %}(за {{ stats.period_days }} дн.){% endif %}</h1>

<h2>Конвейер запросов (5 событий)</h2>
<table>
  <tr><th>Показатель</th><th>Значение</th></tr>
  <tr><td>Запросов получено</td><td>{{ stats.total_requests }}</td></tr>
  <tr><td>Принято</td><td>{{ stats.accepted }}</td></tr>
  <tr><td>Отклонено</td><td>{{ stats.rejected }}
      {% for reason, n in stats.rejected_by_reason.items() %} — {{ reason }}: {{ n }}{% endfor %}</td></tr>
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

<p><a href="/">← К ассистенту</a></p>
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