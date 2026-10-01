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
    overflow-x: hidden; overflow-wrap: break-word;
  }
  body::before {
    content: ""; position: fixed; inset: 0; z-index: -1;
    background: url("/static/porta-map-warm-v3.png") center / cover no-repeat;
    opacity: .34; filter: saturate(95%) contrast(1.05);
  }
  h1 { font-size: 1.5rem; margin: 0; letter-spacing: -.01em; }
  h2 { font-size: 1.05rem; margin: 1.6rem 0 .5rem; }
  p { margin: .45rem 0; }
  button, .btn {
    font: inherit; font-weight: 600; cursor: pointer; border: 0; border-radius: 6px;
    background: #DACDBE; color: #3A2A18; padding: .55rem 1.3rem; text-decoration: none;
  }
  button:hover, .btn:hover { background: #CDBFB0; }
  button[disabled] { opacity: .65; cursor: default; }
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
  .chip b { color: var(--ink); font-weight: 600; overflow-wrap: anywhere; }
  .card {
    background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 4px;
    padding: 1rem; box-shadow: var(--shadow);
  }
  .muted { color: var(--muted); }
  .good { color: var(--good); font-weight: 600; }
  .lead { color: var(--muted); }
  a { color: var(--brown); text-underline-offset: 3px; max-width: 100%; }
"""
TABLE_CSS = """
  .tiles {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
    gap: .7rem; margin: 1.2rem 0 .4rem;
  }
  .tile {
    background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 4px; padding: .8rem .95rem;
    box-shadow: var(--shadow);
  }
  .tile .n { font-size: 1.5rem; font-weight: 700; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  .tile .l { color: var(--muted); font-size: .78rem; margin-top: .15rem; }
  .tablewrap { overflow-x: auto; margin: .4rem 0 1.1rem; -webkit-overflow-scrolling: touch; }
  table {
    border-collapse: collapse; width: 100%; margin: 0;
    background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 4px; overflow: hidden;
    box-shadow: var(--shadow); font-variant-numeric: tabular-nums;
  }
  td, th { border-bottom: 1px solid rgba(139, 90, 43, .14); padding: .5rem .75rem; text-align: left; overflow-wrap: anywhere; }
  tr:last-child td { border-bottom: 0; }
  th { background: rgba(139, 90, 43, .1); font-size: .82rem; font-weight: 600; color: var(--muted); white-space: nowrap; }
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
<title>ИИ-ассистент для ВЭД</title>
<style>""" + BASE_CSS + """
  /* витрина в паттерне Perplexity: шапка, форма по центру, ответ под ней */
  html, body { height: 100%; }
  body {
    max-width: none; margin: 0; padding: 0; overflow: hidden; /* рамка приложения: скролл только внутри main */
    display: flex; flex-direction: column;
    background: #EFE7DA;
  }
  .bar {
    position: sticky; top: 0; z-index: 10; flex: 0 0 auto;
    background: rgba(239, 231, 218, .55); backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--card-line);
  }
  .bar-in {
    max-width: 62rem; margin: 0 auto; padding: .65rem 1.2rem;
    display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap;
  }
  .bar h1 { font-size: 1.1rem; margin: 0; letter-spacing: -.01em; margin-right: auto; }
  a { color: var(--brown); text-underline-offset: 3px; }
  .bar-in a, .backlink {
    text-decoration: none; font-weight: 600; font-size: .8rem; color: var(--brown);
    background: transparent; padding: .2rem .2rem;
    white-space: nowrap; flex: 0 0 auto;  /* кнопка в шапке всегда в одну строку */
  }
  .bar-in a + a { margin-left: 2rem; }  /* ссылки в шапке не слипаются */
  .bar-in a:hover, .backlink:hover { text-decoration: underline; text-underline-offset: 3px; }
  main {
    width: 100%; max-width: 62rem; margin: 0 auto; padding: 3rem 1.2rem 1.2rem; /* окно вопроса опущено ниже; вопрос закреплён */
    display: flex; flex-direction: column; align-items: center;
    flex: 1 1 auto; min-height: 0;
  }
  .scroller { /* прокрутка только в зоне ответа: бары, футер и вопрос всегда на виду */
    width: 100%; max-width: 40rem; margin: 0 auto;
    flex: 1 1 auto; min-height: 0; overflow-y: auto; -webkit-overflow-scrolling: touch;
  }
  .ask {
    width: 100%; max-width: 40rem;
    background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 4px;
    padding: .9rem 1rem; box-shadow: var(--shadow);
  }
  textarea {
    display: block; width: 100%; min-height: 5.4rem; resize: vertical;
    font: inherit; border: 1px solid var(--brown-soft); border-radius: 0;
    padding: .7rem .85rem; background: rgba(255,255,255,.75); color: var(--ink);
  }
  textarea:focus { outline: 2px solid var(--brown); outline-offset: 1px; }
  .row { display: flex; align-items: center; gap: .9rem; margin-top: .7rem; flex-wrap: wrap; }
  button {
    font: inherit; font-weight: 600; cursor: pointer; border: 0; border-radius: 8px;
    background: #DACDBE; color: #3A2A18; padding: .55rem 1.5rem; margin-left: auto;
  }
  button:hover { background: #CDBFB0; }
  button[disabled] { opacity: .6; cursor: default; }
  #wait {
    width: 100%; max-width: 40rem; margin-top: .9rem;
    padding: .75rem 1rem; border-radius: 8px; font-size: .88rem;
    background: rgba(119, 87, 58, .1); border: 1px solid var(--brown-soft);
  }
  #wait .dot { display: inline-block; width: .55rem; height: .55rem; border-radius: 50%;
    background: var(--brown); margin-right: .5rem; vertical-align: 1px; }
  .answer-zone { width: 100%; max-width: 40rem; margin-top: 1.6rem; }
  #answer { display: none; }
  .answer {
    background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 4px;
    padding: 1.05rem 1.2rem; white-space: pre-wrap; overflow-wrap: anywhere;
    box-shadow: var(--shadow); font-size: .95rem;
  }
  .meta { margin-top: .55rem; font-size: .8rem; color: var(--muted); text-align: right; }
  .errorbox { width: 100%; max-width: 40rem; margin-top: .9rem; color: var(--danger); font-size: .92rem; }
  footer { height: 3rem; box-sizing: border-box; margin-top: auto; display: flex;
    align-items: center; justify-content: space-between; /* копирайт слева, дисклеймер справа */
    background: rgba(239, 231, 218, .55); backdrop-filter: blur(10px);
    border-top: 1px solid var(--card-line); overflow: hidden;
    padding: 0 1.2rem; color: var(--muted); font-size: .8rem; }
  footer .r { text-align: right; }
  @media (max-width: 640px) { main { padding: 1rem; } }
</style>
</head>
<body>
<div class="bar">
  <div class="bar-in">
    <h1>ИИ-ассистент для ВЭД</h1>
    <a href="/about">Как работает</a>
    <a href="/admin">Панель оператора</a>
  </div>
</div>

<main>
  <form class="ask" id="askform">
    <textarea id="qbox" placeholder="Ваш вопрос по ВЭД…" required></textarea>
    <div class="row">
      <button type="submit" id="askbtn" aria-label="Спросить" title="Спросить">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" stroke-width="2" stroke-linecap="round"
          stroke-linejoin="round" style="display:block" aria-hidden="true">
          <path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.2 2.2m8.4 8.4 2.2 2.2m0-12.8-2.2 2.2M7.8 16.2l-2.2 2.2"/>
        </svg>
      </button>
    </div>
  </form>
  <div class="scroller">
    <div id="wait" hidden><span class="dot"></span>Ищу ответ в базе — обычно 20–60 секунд</div>
    <div id="errbox" class="errorbox" hidden></div>

    <div class="answer-zone">
      <div id="answer">
        <div class="answer" id="answertext"></div>
        <div class="meta" id="sourceschip"></div>
      </div>
    </div>
  </div>
</main>

<footer><span>© 2026 Irina Korshunova</span><span class="r">Точность норм проверяйте по актуальной редакции НПА</span></footer>

<script>
// запрос к /ask без перезагрузки страницы; на время генерации — искра крутится
var form = document.getElementById('askform'),
    btn = document.getElementById('askbtn'),
    wait = document.getElementById('wait'),
    errbox = document.getElementById('errbox'),
    zone = document.getElementById('answer'),
    textEl = document.getElementById('answertext'),
    srcEl = document.getElementById('sourceschip');
var ICON = btn.innerHTML;
var SPIN = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:block" aria-hidden="true">' +
  '<circle cx="12" cy="12" r="9" stroke-dasharray="42" stroke-dashoffset="14"/></svg>';

form.addEventListener('submit', function (e) {
  e.preventDefault();
  var q = document.getElementById('qbox').value.trim();
  if (!q) return;
  errbox.hidden = true; zone.style.display = 'none';
  btn.disabled = true; btn.innerHTML = SPIN; wait.hidden = false;
  fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded',
               'Accept': 'application/json' },
    body: 'query=' + encodeURIComponent(q)
  }).then(function (r) { return r.json(); }).then(function (d) {
    wait.hidden = true; btn.disabled = false; btn.innerHTML = ICON;
    if (d.error) { errbox.textContent = d.error; errbox.hidden = false; return; }
    textEl.textContent = d.answer || '';
    var s = (d.sources && d.sources.length) ? 'Источники: ' + d.sources.join(', ')
      : 'Источники: в базе не нашлось документов по этому вопросу';
    if (d.from_cache) s += ' · из кеша';
    srcEl.textContent = s;
    zone.style.display = 'block';
    document.getElementById('qbox').value = '';  // вопрос ушёл в ответ — поле готово к следующему запросу
    btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }).catch(function () {
    wait.hidden = true; btn.disabled = false; btn.innerHTML = ICON;
    errbox.textContent = 'Сеть недоступна — попробуйте ещё раз позже.';
    errbox.hidden = false;
  });
});
</script>
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

<p><a class="backlink" href="/">К ассистенту</a></p>
</body>
</html>
"""

LOGIN_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Панель оператора — вход</title>
<style>""" + BASE_CSS + """
  .wrap { max-width: 22rem; margin: 8vh auto 0; }
  h1 { font-size: 1.25rem; }  /* заголовок входа скромнее витринного */
  input[type=password] {
    width: 100%; font: inherit; padding: .6rem .8rem; margin-top: .7rem;
    border: 1px solid var(--brown-soft); border-radius: 0; /* прямоугольное, как поле на /stats */
    background: transparent; color: var(--ink); /* без подвыделенного окошка */
  }
  input[type=password]:focus { outline: 2px solid var(--brown); outline-offset: 1px; }
  .action-row { display: flex; align-items: center; justify-content: space-between; margin-top: .9rem; }
  .action-row button, .action-row a {
    font: inherit; font-weight: 500; cursor: pointer; border: 0; border-radius: 3px;
    background: transparent; color: var(--brown); padding: .3rem .55rem; text-decoration: none;
  }
  .action-row button:hover, .action-row a:hover {
    text-decoration: underline; text-underline-offset: 3px;
  }
</style></head>
<body>
<div class="wrap">
<h1 style="margin-bottom: .6rem">Панель оператора</h1>
{% if no_password %}<div class="card"><p class="muted">Панель отключена: не задан STATS_PASSWORD на сервере.</p></div>
{% else %}<form method="post" action="/admin" class="card">
  <input type="password" name="password" placeholder="Пароль" required autofocus>
  <div class="action-row">
    <button type="submit">Войти</button>
    <a href="/">К ассистенту</a>
  </div>
</form>{% endif %}
</div>
</body></html>
"""

ADMIN_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Панель оператора</title>
<style>""" + BASE_CSS + """
  /* шапка на всю ширину окна */
  .bar {
    position: sticky; top: 0; z-index: 10;
    background: rgba(239, 231, 218, .55); backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--card-line);
    margin: -2.2rem calc(50% - 50vw) .6rem; padding: .65rem 2.4rem;
  }
  .bar-in {
    max-width: none; margin: 0 auto;
    display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap;
  }
  .bar h1 { font-size: 1.1rem; margin: 0; letter-spacing: -.01em; margin-right: auto; }
  .bar-in a, .backlink {
    text-decoration: none; font-weight: 600; font-size: .8rem; color: var(--brown);
    background: transparent; padding: .2rem .2rem;
    white-space: nowrap; flex: 0 0 auto;  /* кнопка в шапке всегда в одну строку */
  }
  .bar-in a + a { margin-left: 2rem; }  /* ссылки в шапке не слипаются */
  .bar-in a:hover, .backlink:hover { text-decoration: underline; text-underline-offset: 3px; }
  .bstats { font-size: .78rem; color: var(--muted); }
  .bstats b { font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
  .bar-in .bstats { margin-right: 2rem; }
  /* переключатель периода — управляет всей страницей */
  .periods { display: inline-flex; gap: .25rem; }
  .p {
    font-size: .72rem; font-weight: 600; color: var(--brown);
    border: 1px solid rgba(119, 87, 58, .12); border-radius: 3px;
    padding: .1rem .45rem; text-decoration: none;
  }
  .p:hover { border-color: rgba(119, 87, 58, .25); }
  .p.active { background: #DACDBE; color: var(--ink); border-color: transparent; }
  /* карточки и тихие кнопки */
  h2 { font-size: .92rem; font-weight: 600; margin: 1.05rem 0 .5rem; }
  .msg { background: rgba(74, 107, 42, .12); border: 1px solid rgba(74, 107, 42, .3);
    color: var(--good); padding: .6rem .8rem; border-radius: 4px; margin: 0 0 .7rem; font-size: .8rem; }
  .bad-msg { background: rgba(166, 50, 38, .08); border: 1px solid rgba(166, 50, 38, .3);
    color: var(--danger); padding: .6rem .8rem; border-radius: 4px; margin: 0 0 .7rem; font-size: .8rem; }
  .btn-quiet {
    font-size: .8rem; font-weight: 600; background: transparent; color: var(--brown);
    border: 1px solid rgba(119, 87, 58, .12); border-radius: 3px;
    padding: .3rem .8rem; cursor: pointer;
  }
  .btn-quiet:hover { background: transparent; border-color: rgba(119, 87, 58, .25); }
  .btn-quiet.light { font-weight: 400; }
  .fname { font-size: .78rem; color: var(--muted); background: var(--card);
    border: 1px solid var(--card-line); border-radius: 3px; padding: .25rem .65rem; }
  .uprow { display: flex; align-items: center; gap: .8rem; flex-wrap: wrap; margin: 1.6rem 0 .55rem; }
  .uprow button { margin-left: auto; font-weight: 600; font-size: .8rem; background: transparent;
    color: var(--brown); border: 1px solid rgba(119, 87, 58, .12); border-radius: 3px; padding: .3rem .8rem; }
  input[type=file] { position: absolute; width: 1px; height: 1px; opacity: 0; }
  label.btn-quiet { display: inline-block; cursor: pointer; }
  /* таблицы: separate — чтобы прилипающая шапка работала */
  table {
    border-collapse: separate; border-spacing: 0; width: 100%; margin: 0;
    background: var(--card); backdrop-filter: blur(12px);
    border: 1px solid var(--card-line); border-radius: 4px;
    box-shadow: var(--shadow); font-variant-numeric: tabular-nums;
  }
  td, th { border-bottom: 1px solid rgba(139, 90, 43, .14); padding: .5rem .75rem; text-align: left; overflow-wrap: anywhere; }
  tr:last-child td { border-bottom: 0; }
  th { background-color: #F5EFE4; font-size: .82rem; font-weight: 600; color: var(--muted); white-space: nowrap;
    position: sticky; top: 0; z-index: 2; }
  thead th { box-shadow: inset 0 -1px 0 rgba(139, 90, 43, .14); }
  .reason-chip { display: inline-block; font-size: .75rem; background: rgba(139, 90, 43, .1);
    color: var(--muted); border-radius: 3px; padding: .12rem .5rem; margin-left: .15rem; }
  .err-row td { background: rgba(166, 50, 38, .06); }
  /* дашборд-ряд: пополнение базы слева, статистика справа;
     растянут на ширину ленты — края стыкуются ровно */
  .dash {
    display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem;
    align-items: stretch;
    margin: .2rem -18rem 0;
  }
  .dash > div { display: flex; flex-direction: column; }
  .dash .card { flex: 1; display: flex; flex-direction: column; }
  @media (max-width: 44rem) { .dash { grid-template-columns: 1fr; margin-left: -1rem; margin-right: -1rem; } }
  /* компактная таблица статистики + рамка с прокруткой */
  table.compact td, table.compact th { font-size: .8rem; padding: .22rem .7rem; }
  table.compact th { font-size: .75rem; }
  .statwrap { max-height: 11.5rem; overflow: auto; }
  /* широкая лента событий: выходит за шину контента,
     остальную высоту экрана отдаём ей — всё внутри одного экрана */
  .tablewrap { overflow-x: auto; margin: .4rem 0 1.1rem; -webkit-overflow-scrolling: touch; }
  .tablewrap.wide { margin: .4rem -18rem 0; flex: 1; min-height: 8rem; }
  .wide-head { margin-left: -18rem !important; text-align: left; }
  /* шрифт ленты — как в таблице статистики */
  .tablewrap.wide td, .tablewrap.wide th { font-size: .8rem; padding: .42rem .7rem; }
  /* строка расхода LLM под лентой */
  .totals {
    margin: -1px -18rem 0; padding: .32rem .75rem;
    background: #F5EFE4; font-size: .78rem; color: var(--muted);
    border: 1px solid var(--card-line); border-radius: 0 0 4px 4px;
  }
  .totals b { color: var(--ink); font-variant-numeric: tabular-nums; }
  html, body { height: 100vh; }
  body { display: flex; flex-direction: column; padding: 2.2rem 1rem 1.2rem; }
  @media (max-width: 44rem) {
    .dash { margin-left: -1rem; margin-right: -1rem; }
    .tablewrap.wide, .totals { margin-left: -1rem; margin-right: -1rem; }
    .wide-head { margin-left: -1rem !important; }
    html, body { height: auto; body-scroll: auto; }
    body { height: auto; min-height: 100vh; }
  }
</style></head>
<body>
<div class="bar">
  <div class="bar-in">
    <h1>Панель оператора</h1>
    <span class="bstats"><b>{{ stats.total_requests }}</b> запросов · <b>{{ stats.accepted }}</b> принято · <b>{{ stats.cache_share_pct }}%</b> из кеша · <b>{{ avg_sec }}</b> с среднее</span>
    <span class="periods">{% for p, label in period_options %}<a class="p {{ 'active' if p == period else '' }}" href="/admin?period={{ p }}">{{ label }}</a>{% endfor %}</span>
    <a href="/">ассистент</a>
    <a href="/admin?logout=1">выйти</a>
  </div>
</div>

<div class="dash">
  <div>
    <h2>Пополнение базы</h2>
    <div class="card">
      {% if ingest_msg %}<div class="msg">{{ ingest_msg }}</div>{% endif %}
      {% if ingest_err %}<div class="bad-msg">{{ ingest_err }}</div>{% endif %}
      <p style="margin: .2rem 0 .7rem; font-size: .8rem">Файл <b>.txt</b> или <b>.md</b> (до 2 МБ). Документ будет добавлен
      в базу и доступен сразу после индексации; уже загруженные с тем же именем файлы
      не дублируются.</p>
      {% if ingesting %}
        <p class="muted" style="font-size: .8rem">Сейчас идёт индексация предыдущего файла — обновите
        страницу через минуту.</p>
      {% else %}
      <form method="post" action="/admin/upload" enctype="multipart/form-data">
        <div class="uprow">
          <input type="file" id="doc" name="doc" accept=".txt,.md" required>
          <label class="btn-quiet light" for="doc">Выбрать файл</label>
          <span class="fname" id="fname">файл не выбран</span>
          <button type="submit" class="btn-quiet">Загрузить и индексировать</button>
        </div>
      </form>
      {% endif %}
      <p class="muted" style="font-size: .72rem; margin: auto 0 0; text-align: right">Всего в базе: {{ chunk_count }} чанков.</p>
    </div>
  </div>
  <div>
    <h2>Статистика</h2>
    <div class="statwrap"><table class="compact">
      <tr><th>Показатель</th><th>Значение</th></tr>
      <tr><td>Запросов получено</td><td>{{ stats.total_requests }}</td></tr>
      <tr><td>Принято</td><td class="good">{{ stats.accepted }}</td></tr>
      <tr><td>Отклонено</td><td>{{ stats.rejected }}
          {% for reason, n in stats.rejected_by_reason.items() %}<span class="reason-chip">{{ reason }}: {{ n }}</span>{% endfor %}</td></tr>
      <tr><td>Источники</td><td>
          {% for src, n in stats.by_source.items() %}{{ src }}: {{ n }}{% if not loop.last %} · {% endif %}{% endfor %}</td></tr>
      <tr><td>Пользователи (telegram)</td><td>
          {{ stats.unique_users }} уникальных
          {% for uid, n in stats.by_user.items() %}<span class="reason-chip">{{ uid }}: {{ n }}</span>{% endfor %}</td></tr>
      <tr><td>Ответов подготовлено</td><td>{{ stats.answered }}</td></tr>
      <tr><td>Из кеша</td><td>{{ stats.cache_hits }} ({{ stats.cache_share_pct }}%)</td></tr>
      <tr><td>Средняя длительность</td><td>{{ stats.avg_duration_ms or '—' }} мс</td></tr>
      <tr><td>Ошибок</td><td>{{ stats.errors }}</td></tr>
    </table></div>
  </div>
</div>

<h2 class="wide-head">События конвейера <span class="muted" style="font-weight:400; font-size:.8rem">за выбранный период</span></h2>
{% if feed %}
<div class="tablewrap wide"><table>
  <tr><th>Время</th><th>Событие</th><th>Источник</th><th>Вопрос</th><th>Кеш</th><th>мс</th><th>Модель</th><th>Токены (p/c)</th><th>Заметка</th></tr>
  {% for e in feed %}
  <tr {% if e.error %}class="err-row"{% endif %}>
    <td>{{ e.time }}</td>
    <td>{{ e.event }}</td>
    <td>{{ e.source }}</td>
    <td>{{ e.query or '—' }}</td>
    <td>{% if e.from_cache is not none %}{{ 'да' if e.from_cache else '—' }}{% else %}—{% endif %}</td>
    <td>{% if e.duration_ms %}{{ e.duration_ms }}{% else %}—{% endif %}</td>
    <td>{{ e.model or '—' }}</td>
    <td>{% if e.prompt_tokens is not none %}{{ e.prompt_tokens }}/{{ e.completion_tokens }}{% else %}—{% endif %}</td>
    <td>{% if e.error %}⚠ {{ e.error }}{% elif e.reason %}{{ e.reason }}{% else %}—{% endif %}</td>
  </tr>
  {% endfor %}
</table></div>
{% else %}
<p class="muted" style="margin: .4rem -18rem 0">Событий за выбранный период нет.</p>
{% endif %}
<div class="totals">Расход LLM ({{ model_name }}): вход — {{ stats.total_prompt_tokens }} токенов (вопрос + найденное в базе),
выход — {{ stats.total_completion_tokens }} (ответ). Стоимость за выбранный период ≈ {{ stats.cost_usd }} $</div>

<script>
// показать имя выбранного файла вместо «файл не выбран»
var doc = document.getElementById('doc');
if (doc) doc.addEventListener('change', function () {
  document.getElementById('fname').textContent =
    doc.files.length ? doc.files[0].name : 'файл не выбран';
});
</script>
</body></html>
"""

PERIOD_OPTIONS = [(1, "24 ч"), (7, "7 дн"), (30, "30 дн")]

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
<div class="tablewrap"><table>
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
</table></div>

<h2>Токены по моделям</h2>
<div class="tablewrap"><table>
  <tr><th>Модель</th><th>Prompt</th><th>Completion</th></tr>
  {% for model, tok in stats.tokens_by_model.items() %}
  <tr><td>{{ model }}</td><td>{{ tok.prompt }}</td><td>{{ tok.completion }}</td></tr>
  {% else %}
  <tr><td colspan="3">Нет данных за период</td></tr>
  {% endfor %}
</table></div>

<footer><a class="backlink" href="/">К ассистенту</a></footer>
</body>
</html>
"""


# --------------------------------------------------------------- витрина
@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE)


# Человекочитаемые причины отклонения (причины — из конвейера логов)
REJECT_TEXT = {
    "empty_query": "Пустой вопрос — напишите, что хотите узнать.",
    "too_long_query": "Вопрос слишком длинный — сократите и повторите.",
    "internal_error": "Внутренняя ошибка при обработке — попробуйте позже.",
}


def _friendly_reject(exc: ValueError) -> str:
    raw = str(exc)
    if raw.startswith("Запрос отклонён: "):
        return REJECT_TEXT.get(raw.split(": ", 1)[1], raw)
    return raw


@app.route("/ask", methods=["POST"])
def ask():
    """JSON-ответ для витрины: {answer, sources, from_cache} или {error}."""
    query = request.form.get("query", "").strip()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()

    if rate_limited(ip):
        return jsonify_wrap({"error": "Слишком много вопросов подряд — подождите немного и попробуйте ещё."}, 429)

    try:
        result = get_pipeline().query(query)
    except ValueError as ve:
        # Отклонение с причиной — уже залогировано в pipeline
        return jsonify_wrap({"error": _friendly_reject(ve)}, 400)
    except Exception:
        return jsonify_wrap({"error": "Не получилось получить ответ по базе — попробуйте другой вопрос."}, 500)

    from_cache = result.get("from_cache", False)
    # Источники показываются только для свежего ответа: в кеш попадает
    # текст документа без метаданных, у него source нет
    sources = None
    if not from_cache:
        sources = list(dict.fromkeys(
            d.get("source") for d in (result.get("context_docs") or [])
            if isinstance(d, dict) and d.get("source")))
    return jsonify_wrap({"answer": result["answer"],
                         "sources": sources or [], "from_cache": from_cache})


def jsonify_wrap(payload: dict, code: int = 200):
    """Мини-jsonify: без импорта flask.jsonify — тот же контракт."""
    import json as _json
    from flask import Response
    return Response(_json.dumps(payload, ensure_ascii=False),
                    status=code, mimetype="application/json")


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

    try:
        period = int(request.args.get("period", "7"))
    except ValueError:
        period = 7
    if period not in (1, 7, 30):
        period = 7

    pipeline = get_pipeline()
    stats = pipeline.logger.get_stats(period_days=period)
    avg_sec = round(stats["avg_duration_ms"] / 1000) if stats["avg_duration_ms"] else None
    return render_template_string(
        ADMIN_PAGE,
        stats=stats,
        period=period,
        period_options=PERIOD_OPTIONS,
        avg_sec=avg_sec if avg_sec is not None else "—",
        model_name=os.getenv("MODEL_NAME", "—"),
        feed=pipeline.logger.get_recent(limit=25, days=period)["events"],
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