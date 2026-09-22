"""
Веб-интерфейс RAG-ассистента по ВЭД — для запуска на сервере.

Страницы:
    GET  /      — форма вопроса
    POST /ask   — обработка вопроса через RAG pipeline
    GET  /stats — статистика логов (конвейер урока PEcf09), доступ по паролю
                  из переменной окружения STATS_PASSWORD (разграничение доступа)

Запуск локально:  python webapp.py
Запуск на сервере: gunicorn -w 1 -b 127.0.0.1:8002 webapp:app
"""

import os
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, request, render_template_string, redirect, url_for

from rag_pipeline import RAGPipeline

load_dotenv()

app = Flask(__name__)

_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Ленивая инициализация pipeline (один раз на процесс gunicorn)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path=os.getenv("CACHE_DB_PATH", "api_rag_cache.db"),
            data_file="data",
        )
    return _pipeline


def require_stats_password(view):
    """Простой доступ к статистике по паролю (правило урока: разграничение доступа)."""
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


PAGE = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG-ассистент по ВЭД</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #222; }
  h1 { font-size: 1.4rem; }
  textarea, button { font-size: 1rem; }
  textarea { width: 100%; box-sizing: border-box; min-height: 90px; }
  button { padding: .5rem 1.2rem; cursor: pointer; }
  .answer { background: #f5f5f0; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }
  .meta { color: #777; font-size: .85rem; margin-top: .5rem; }
  .error { color: #b00; }
  a { color: #06c; }
</style>
</head>
<body>
<h1>RAG-ассистент по ВЭД</h1>
<p>Таможня, Incoterms, ТН ВЭД, расчёты, валютный контроль. <a href="/stats">Статистика</a></p>
<form method="post" action="/ask">
  <textarea name="query" placeholder="Ваш вопрос по ВЭД..." required></textarea>
  <p><button type="submit">Спросить</button></p>
</form>
{% if answer %}
  <h2>Ответ</h2>
  <div class="answer">{{ answer }}</div>
  <p class="meta">
    Источник: {{ src }}{% if docs %} · документов в контексте: {{ docs }}{% endif %}
    {% if request_id %} · id запроса: {{ request_id }}{% endif %}
  </p>
{% endif %}
{% if error %}<p class="error">{{ error }}</p>{% endif %}
</body>
</html>
"""

STATS_PAGE = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Статистика — логирование (урок PEcf09)</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  table { border-collapse: collapse; width: 100%; margin: .6rem 0 1.2rem; }
  td, th { border: 1px solid #ddd; padding: .45rem .7rem; text-align: left; }
  th { background: #f0f0ea; }
  h2 { font-size: 1.1rem; }
</style>
</head>
<body>
<h1>Статистика логов (за {{ stats.period_days }} дн.)</h1>

<h2>Конвейер запросов (5 событий урока)</h2>
<table>
  <tr><th>Показатель</th><th>Значение</th></tr>
  <tr><td>Запросов получено</td><td>{{ stats.total_requests }}</td></tr>
  <tr><td>Принято</td><td>{{ stats.accepted }}</td></tr>
  <tr><td>Отклонено</td><td>{{ stats.rejected }}
      {% for reason, n in stats.rejected_by_reason.items() %} — {{ reason }}: {{ n }}{% endfor %}</td></tr>
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


@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE)


@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("query", "")
    try:
        result = get_pipeline().query(query)
        return render_template_string(
            PAGE,
            answer=result["answer"],
            src="кеш" if result["from_cache"] else f"LLM ({result.get('model', '')})",
            docs=len(result.get("context_docs") or []),
            request_id=result.get("request_id"),
        )
    except ValueError as ve:
        # Отклонение с причиной — уже залогировано в pipeline
        return render_template_string(PAGE, error=str(ve))
    except Exception as e:
        return render_template_string(PAGE, error=f"Ошибка обработки: {e}")


@app.route("/stats")
@require_stats_password
def stats():
    stats_data = get_pipeline().logger.get_stats(period_days=7)
    return render_template_string(STATS_PAGE, stats=stats_data)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8002, debug=False)