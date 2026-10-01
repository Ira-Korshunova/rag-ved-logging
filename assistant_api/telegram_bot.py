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

import csv
import io
import logging
import os
import re
import sys
from typing import Dict, List
from datetime import datetime

from dotenv import load_dotenv
from telegram import BotCommand, InputFile, Update
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


USER_HELP_TEXT = (
    "🤖 <b>Ассистент по ВЭД</b>\n\n"
    "Отвечаю по базе документов ЕАЭС и ТК РФ: таможенные процедуры, "
    "Incoterms, ТН ВЭД, формы расчётов, валютный контроль.\n\n"
    "Просто напишите вопрос. Под каждым ответом — документы, "
    "по которым он собран."
)


def _admin_ids():
    """Список Telegram user_id оператора (ADMIN_USER_IDS, через запятую).
    Пока переменная не задана — служебные команды допускаются всем
    (режим разработки); на сервере стоит вписать свой user_id."""
    raw = os.getenv("ADMIN_USER_IDS", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def is_admin(user_id: str) -> bool:
    """Служебные команды (/stats, /ingest) — только оператору."""
    admins = _admin_ids()
    if not admins:
        return True
    return user_id in admins


# Кнопки-команды (ReplyKeyboard, как у преподавателя в уроке)
BTN_HELP = "Помощь"
BTN_STATS = "📊 Статистика"
BTN_LOGS = "📁 Логи"
_COMMAND_LABELS = {BTN_HELP, BTN_STATS, BTN_LOGS, "Помощь", "Статистика", "Логи"}

# Память диалога: последние обмены по каждому чату — для вопросов-
# продолжений («а про FOB подробнее?»). Держим 3 последних пары;
# в лог эти данные не попадают (там только маскированный вопрос).
DIALOG_MEMORY: Dict[int, List[tuple]] = {}
MAX_DIALOG_TURNS = 3


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
            os.getenv("DATA_DIR", "data"))
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
    """Приветствие: суть ассистента + кнопки (служебные — только оператору)."""
    user_id = str(update.effective_user.id)
    # /start — новый диалог: память обменов этого чата сбрасываем
    DIALOG_MEMORY.pop(update.effective_chat.id, None)
    kb = None
    if update.effective_chat.type == "private":
        from telegram import KeyboardButton, ReplyKeyboardMarkup
        rows = [
            [KeyboardButton(BTN_HELP), KeyboardButton(BTN_STATS)],
            [KeyboardButton(BTN_LOGS)],
        ] if is_admin(user_id) else [
            [KeyboardButton(BTN_HELP)],
        ]
        kb = ReplyKeyboardMarkup(rows, resize_keyboard=True)
    if kb:
        await update.message.reply_html(USER_HELP_TEXT,
                                        reply_markup=kb)
    else:
        await update.message.reply_html(USER_HELP_TEXT)


async def stats_response(update: Update):
    """Статистика из логов — только оператору (кнопка /статистика или /stats)."""
    if not is_admin(str(update.effective_user.id)):
        await update.message.reply_text(
            "Статистика запросов доступна только оператору ассистента.")
        return
    pipeline = get_pipeline()
    stats = pipeline.logger.get_stats(period_days=7)
    lines = [
        "<b>📊 Статистика за 7 дней</b>",
        f"Запросов: {stats['total_requests']}",
        f"Принято: {stats['accepted']}, отклонено: {stats['rejected']}",
        f"⚙️ Модель ответа: <b>{os.getenv('MODEL_NAME', '—')}</b>",
        f"Эмбеддинги: {os.getenv('EMBEDDING_PROVIDER', 'api')} / "
        f"<b>{os.getenv('EMBEDDING_MODEL', '—')}</b>",
        f"Фрагментов в контексте: топ-{pipeline.top_k}",
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


async def logs_response(update: Update):
    """Лог конвейера CSV-файлом — только оператору (кнопка /логи).
    В файле маскированные данные (то же правило, что у страницы статистики)."""
    if not is_admin(str(update.effective_user.id)):
        await update.message.reply_text(
            "Логи запросов доступны только оператору ассистента.")
        return
    pipeline = get_pipeline()
    events = pipeline.logger.get_recent(limit=200)["events"]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["время", "событие", "источник", "вопрос (маскирован)",
                     "из кеша", "мс", "причина", "ошибка"])
    for e in events:
        writer.writerow([
            e.get("time", ""), e.get("event", ""), e.get("source", ""),
            e.get("query", ""), e.get("from_cache", ""),
            e.get("duration_ms", ""), e.get("reason", ""), e.get("error", ""),
        ])
    filename = "логи_conвейера_" + datetime.now().strftime("%Y%m%d_%H%M") + ".csv"
    await update.message.reply_document(
        document=InputFile(buf.getvalue().encode("utf-8-sig"), filename=filename),
        caption=f"Лог конвейера — последние {len(events)} событий "
                "(перс. данные маскированы до записи).")


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Вопрос пользователя → конвейер → лог(source=telegram, user_id) → ответ.
    Текст кнопок (Помощь/Статистика/Логи) маршрутизируется в свои ответы."""
    user_message = update.message.text
    user_id = str(update.effective_user.id)   # как в коде урока

    # кнопки-команды — не вопросы, в конвейер и лог не идут
    if user_message in _COMMAND_LABELS:
        if user_message in (BTN_HELP, "Помощь"):
            await cmd_start(update, context)
        elif user_message in (BTN_STATS, "Статистика"):
            await stats_response(update)
        else:
            await logs_response(update)
        return

    # Приветствия и болтовня — не вопросы к базе: отвечаем дружелюбно,
    # конвейер не гоняем и в лог не пишем
    low = user_message.lower().strip()
    if len(low) <= 20 and re.search(
            r"^(привет|здравств|добрый (утро|день|вечер)|hi|hello|ку)",
            low):
        await update.message.reply_text(
            "Привет! Я ассистент по ВЭД — задайте вопрос по базе "
            "документов ЕАЭС и ТК РФ, например про Incoterms или ТН ВЭД.")
        return

    # Светская болтовня («как дела?» и т.п.) — не вопрос к базе:
    # конвейер не нужен, отвечаем по-человечески
    if len(low) <= 25 and re.search(
            r"^(как дела|как ты|как жизнь|что нового|что делаешь|чем займ|"
            r"спасибо|благодар|пока|хорошо|ладно|ок)", low):
        await update.message.reply_text(
            "У меня всё хорошо — я готов отвечать по базе знаний ВЭД. "
            "Спросите, например: какие документы нужны для импорта, "
            "что такое ИМ40, чем CIF отличается от FOB.")
        return

    pipeline = get_pipeline()
    # История диалога этого чата: последние пары (вопрос, ответ)
    chat_id = update.effective_chat.id
    history = DIALOG_MEMORY.get(chat_id, [])
    await update.message.chat.send_action(action=ChatAction.TYPING)
    try:
        result = pipeline.query(user_message, user_id=user_id, history=history)
        # Модель честно признаёт, что в базе ответа нет, — но пользователю
        # это звучит как техническая простыня; отвечаем по-человечески
        if result["answer"].lstrip().lower().startswith(
                "в предоставленном контексте нет информации"):
            await update.message.reply_text(
                "Это не по моей части — я отвечаю по базе знаний ВЭД: "
                "таможенные процедуры, Incoterms, ТН ВЭД, документы, "
                "валютный контроль. Задайте вопрос из этой темы.")
            return
        if result.get("from_cache"):
            answer = result["answer"] + "\n\n— из кеша"
        else:
            # Источники собираются из метаданных найденных документов
            # (как в веб-витрине); показываются только свежим ответам
            sources = list(dict.fromkeys(
                d.get("source") for d in (result.get("context_docs") or [])
                if isinstance(d, dict) and d.get("source")))
            tail = "Источники: " + (", ".join(sources) if sources else "нет")
            answer = result["answer"] + "\n\n" + tail
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

    # Запоминаем обмен в историю диалога (чистый текст, без подписи
    # источников — она интерфейсная); держим не больше 3 последних
    turns = DIALOG_MEMORY.setdefault(chat_id, [])
    turns.append((user_message, result["answer"][:600]))
    if len(turns) > MAX_DIALOG_TURNS:
        del turns[:len(turns) - MAX_DIALOG_TURNS]


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error("Ошибка Telegram: %s", context.error)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await stats_response(update)


async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await logs_response(update)


async def post_init(app):
    """Меню команд (кнопка с ≡ слева от поля ввода) — регистрируется
    после старта, когда цикл уже крутится и await работает."""
    await app.bot.set_my_commands([
        BotCommand("start", "Старт — приветствие и кнопки"),
        BotCommand("help", "Помощь"),
        BotCommand("stats", "Статистика запросов"),
        BotCommand("logs", "Выгрузка логов (CSV)"),
    ])
    log.info("Меню команд зарегистрировано")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Ошибка: TELEGRAM_BOT_TOKEN не установлен (впиши в .env)")
        sys.exit(1)

    log.info("Запуск бота через Long Poll API...")
    app = Application.builder().token(token).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("logs", logs_command))
    app.add_handler(CommandHandler("ingest", ingest_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,
                                   handle_question))
    app.add_error_handler(on_error)
    log.info("Бот ждёт сообщения (%s)", datetime.now().isoformat(timespec="seconds"))
    app.run_polling(drop_pending_updates=True, allowed_updates=["message"])


if __name__ == "__main__":
    main()