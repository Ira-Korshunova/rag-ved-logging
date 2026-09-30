"""
Телеграм-бот RAG-ассистента по ВЭД — как в уроке PEcf09 (быстрый бот поверх движка).

Код преподавателя: обработчик сообщений Telegram → pipeline → лог с
source="telegram", user_id. Здесь то же поверх нашего конвейера логирования:
    - источник запроса: source="telegram" (поле в логе, как в схеме урока)
    - user_id пишется в лог (как у преподавателя); username НЕ пишем —
      анонимность по таблице урока
    - логи пишутся в ту же базу request_logs.db, что у web и console

Запуск: python telegram_bot.py   (токен — TELEGRAM_BOT_TOKEN из .env)
На сервере — второй контейнер того же образа (см. deploy/DEPLOY_RAG.md).
"""

import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("tg")

_pipeline = None


def get_pipeline():
    """Ленивая инициализация движка (source=telegram)."""
    global _pipeline
    if _pipeline is None:
        from rag_pipeline import RAGPipeline
        _pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path=os.getenv("CACHE_DB_PATH", "api_rag_cache.db"),
            data_file="data",
            source="telegram",
        )
    return _pipeline


HELP_TEXT = (
    "🤖 <b>RAG-ассистент по ВЭД</b>\n\n"
    "Спрашивай про таможню, Incoterms, ТН ВЭД, формы расчётов, валютный "
    "контроль — просто отправь вопрос текстом.\n\n"
    "Команды:\n"
    "/start — это сообщение\n"
    "/stats — статистика запросов за 7 дней\n"
    "/ingest — пополнить базу новыми файлами из data/ (для админа)"
)


def _admin_ids():
    """Список Telegram user_id, которым разрешён /ingest (ADMIN_USER_IDS,
    через запятую). Если переменная не задана — команда допускается всем
    (режим разработки); на сервере админа стоит вписать."""
    raw = os.getenv("ADMIN_USER_IDS", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


ADMIN_HINT = (
    "Впиши в ADMIN_USER_IDS (файл .env, через запятую) свой Telegram user_id, "
    "и команда откроется. Узнать свой user_id можно у @userinfobot."
)


async def ingest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пополнение базы знаний (админ-команда): инкрементальная индексация
    папки data/ — в коллекцию добавляются только ещё не индексированные файлы,
    существующие вектора не пересоздаются."""
    user_id = str(update.effective_user.id)
    admins = _admin_ids()
    if admins and user_id not in admins:
        await update.message.reply_text(
            "Команда доступна только администратору базы знаний.")
        return

    pipeline = get_pipeline()
    await update.message.chat.send_action(action=ChatAction.TYPING)
    try:
        result = pipeline.vector_store.add_documents_from_folder(
            os.getenv("INGEST_DATA_PATH", "data"))
    except FileNotFoundError as e:
        await update.message.reply_text(f"Папка с документами не найдена: {e}")
        return
    except Exception as e:
        log.error("Ошибка индексации: %s", e)
        await update.message.reply_text(
            "Индексация не удалась — ошибка пошла в лог.")
        return

    if result["added_files"] == 0:
        await update.message.reply_text(
            f"Новых документов нет — база актуальна "
            f"({result['total']} чанков).")
    else:
        await update.message.reply_html(
            f"✅ База пополнена: <b>+{result['added_chunks']} чанков</b> "
            f"из {result['added_files']} новых файлов.\n"
            f"Всего в коллекции: {result['total']} чанков.")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(HELP_TEXT)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статистика из логов (как /stats у преподавателя, без пароля —
    доступен только автору команды в личном чате с ботом)."""
    pipeline = get_pipeline()
    stats = pipeline.logger.get_stats(period_days=7)
    lines = [
        "<b>📊 Статистика за 7 дней</b>",
        f"Запросов: {stats['total_requests']}",
        f"Принято: {stats['accepted']}, отклонено: {stats['rejected']}",
    ]
    if stats["by_source"]:
        src = ", ".join(f"{k}: {v}" for k, v in stats["by_source"].items())
        lines.append(f"По источникам: {src}")
    lines.append(f"Из кеша: {stats['cache_hits']} ({stats['cache_share_pct']}%)")
    if stats["avg_duration_ms"]:
        lines.append(f"Средняя длительность: {stats['avg_duration_ms']} мс")
    if stats["tokens_by_model"]:
        for model, tok in stats["tokens_by_model"].items():
            lines.append(
                f"{model}: {tok['prompt']}/{tok['completion']} ток. (prompt/completion)")
    if stats["errors"]:
        lines.append(f"Ошибок: {stats['errors']}")
    await update.message.reply_html("\n".join(lines))


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Вопрос пользователя → конвейер → лог(source=telegram, user_id) → ответ."""
    user_message = update.message.text
    user_id = str(update.effective_user.id)   # как в коде урока

    pipeline = get_pipeline()
    await update.message.chat.send_action(action=ChatAction.TYPING)
    try:
        result = pipeline.query(user_message, user_id=user_id)
        suffix = "\n\n— по базе знаний ВЭД" + (" · из кеша" if result.get("from_cache") else "")
        answer = result["answer"] + suffix
    except ValueError as e:
        # Отклонение по конвейеру (отклонение уже записано в лог с причиной)
        await update.message.reply_text(
            f"Запрос отклонён: {str(e).replace('Запрос отклонён: ', '')}"
        )
        return
    except Exception as e:
        log.error("Ошибка обработки: %s", e)
        await update.message.reply_text(
            "Не получилось обработать запрос — ошибка зафиксирована в логах.")
        return

    # Telegram ограничивает сообщение 4096 символами — режем с запасом
    await update.message.reply_text(answer[:4000], disable_web_page_preview=True)
    log.info("Ответ отправлен (source=telegram, user_id=%s)", user_id)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error("Ошибка Telegram: %s", context.error)


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Ошибка: TELEGRAM_BOT_TOKEN не установлен (впиши в .env)")
        sys.exit(1)

    log.info("Запуск бота через Long Poll API...")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("ingest", ingest_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,
                                   handle_question))
    app.add_error_handler(on_error)
    log.info("Бот ждёт сообщения (%s)", datetime.now().isoformat(timespec="seconds"))
    app.run_polling(drop_pending_updates=True, allowed_updates=["message"])


if __name__ == "__main__":
    main()