"""
Модуль логирования запросов по конвейеру урока PEcf09 «Портфолио: метрики и логирование».

Конвейер из 5 событий на каждый запрос:
    1. REQUEST_RECEIVED  — пользователь отправил запрос
    2. REQUEST_ACCEPTED / REQUEST_REJECTED (с причиной) — запрос принят/отклонён
    3. PROCESSING_STARTED — ассистент приступил к выполнению запроса
    4. ANSWER_PREPARED   — ассистент подготовил ответ (модель, токены, длительность)
    5. ANSWER_SENT       — система отправила ответ

Хранение: SQLite (request_logs.db) + JSONL-файл (по строке на событие).
Безопасность (таблица урока):
    - анонимность: PII-фильтр маскирует телефоны, email, карты, паспорта до записи
    - шифрование/доступ: ключи в .env вне репозитория, БД и JSONL вне репозитория
    - срок хранения: ретеншн (LOG_RETENTION_DAYS, по умолчанию 90 дней), очистка при старте
"""

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

# События конвейера урока
EVENT_RECEIVED = "request_received"        # 1. пользователь отправил запрос
EVENT_ACCEPTED = "request_accepted"        # 2а. запрос принят
EVENT_REJECTED = "request_rejected"        # 2б. запрос отклонён (с причиной)
EVENT_STARTED = "processing_started"       # 3. ассистент приступил
EVENT_ANSWER_READY = "answer_prepared"     # 4. ассистент подготовил ответ
EVENT_SENT = "answer_sent"                 # 5. система отправила ответ

# Причины отказа (шаг 2 конвейера)
REASON_EMPTY = "empty_query"               # пустой ввод
REASON_TOO_LONG = "too_long_query"         # превышение длины
REASON_ERROR = "internal_error"            # ошибка при обработке


class RequestLogger:
    """Логирование запросов по конвейеру урока с учётом правил безопасности."""

    def __init__(self,
                 db_path: str = "request_logs.db",
                 jsonl_path: str = "request_logs.jsonl",
                 retention_days: int = None,
                 max_query_len: int = 2000,
                 source: str = "web"):
        """
        Args:
            db_path: путь к SQLite-базе логов
            jsonl_path: путь к JSONL-файлу (выгрузка для анализа)
            retention_days: срок хранения логов в днях (None — взять LOG_RETENTION_DAYS
                            из окружения, по умолчанию 90)
            max_query_len: максимальная длина запроса (длиннее — отклонение)
            source: канал запросов по умолчанию (web / console / telegram — как источник
                    запроса в схеме урока; у веб-канала нет юзеров, поэтому
                    персональных данных нет, только канал)
        """
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self.max_query_len = max_query_len
        self.source = source

        if retention_days is None:
            retention_days = int(os.getenv("LOG_RETENTION_DAYS", "90"))
        self.retention_days = retention_days

        self._init_db()
        self.cleanup()

    # ------------------------------------------------------------------ база

    def _init_db(self):
        """Создание таблицы логов и индексов."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                event TEXT NOT NULL,
                reason TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                duration_ms INTEGER,
                from_cache INTEGER DEFAULT 0,
                error TEXT,
                query_masked TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_event ON request_log(event)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_created ON request_log(created_at)")
        conn.commit()
        conn.close()
        # Миграция старых баз: колонки source (канал) и user_id появились позже
        conn = sqlite3.connect(self.db_path)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(request_log)")]
        if columns and "source" not in columns:
            conn.execute("ALTER TABLE request_log ADD COLUMN source TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_source ON request_log(source)")
        if columns and "user_id" not in columns:
            conn.execute("ALTER TABLE request_log ADD COLUMN user_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_user ON request_log(user_id)")
        if columns and ("source" not in columns or "user_id" not in columns):
            conn.commit()
        conn.close()

    # ------------------------------------------------------------------ API

    def new_request(self) -> str:
        """Новый request_id для цепочки событий."""
        return uuid.uuid4().hex[:12]

    def log(self, request_id: str, event: str, query: str = None,
            reason: str = None, model: str = None,
            prompt_tokens: int = None, completion_tokens: int = None,
            duration_ms: int = None, from_cache: bool = False,
            error: str = None, source: str = None, user_id: str = None):
        """
        Запись события конвейера. Текст запроса перед записью проходит
        PII-фильтр (анонимность по уроку). Канал запроса (source) берётся
        из конструктора, отдельное событие может переопределить.
        user_id — идентификатор пользователя канала (только telegram, как в
        коде урока; web-канал пользователей не имеет, поле остаётся пустым).
        """
        if source is None:
            source = self.source
        query_masked = self.mask_pii(query)[:500] if query else None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO request_log
                (request_id, event, reason, model, prompt_tokens,
                 completion_tokens, duration_ms, from_cache, error, query_masked,
                 source, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (request_id, event, reason, model, prompt_tokens,
              completion_tokens, duration_ms, int(from_cache), error, query_masked,
              source, user_id))
        conn.commit()
        conn.close()

        # JSONL: одна строка на событие (для выгрузки и аналитики)
        record = {
            "request_id": request_id, "event": event, "reason": reason,
            "model": model, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "duration_ms": duration_ms,
            "from_cache": from_cache, "error": error,
            "query_masked": query_masked, "source": source, "user_id": user_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._append_jsonl(record)

    # ------------------------------------------------------------- проверки

    def validate(self, query: str) -> Optional[str]:
        """Проверка запроса до обработки. Возвращает причину отказа или None."""
        if not query or not query.strip():
            return REASON_EMPTY
        if len(query) > self.max_query_len:
            return REASON_TOO_LONG
        return None

    # ------------------------------------------------------------------ PII

    @staticmethod
    def mask_pii(text: str) -> str:
        """Маскирование персональных данных: телефоны, email, карты, паспорта."""
        patterns = [
            (r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', "[email]"),
            (r'(?:\+7|8)[\s\-\(]?\d{3}[\)\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}', "[телефон]"),
            (r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', "[карта]"),
            (r'\b\d{2}\s?\d{2}\s?\d{6}\b', "[документ]"),
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        return text

    # -------------------------------------------------------------- ретеншн

    def cleanup(self):
        """Удаление логов старше срока хранения (политика по уроку)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM request_log WHERE created_at < datetime('now', ?)",
            (f"-{self.retention_days} days",)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    def _append_jsonl(self, record: Dict[str, Any]):
        """Дописывает событие в JSONL; ротация файла при превышении 10 МБ."""
        try:
            if os.path.exists(self.jsonl_path) and \
                    os.path.getsize(self.jsonl_path) > 10 * 1024 * 1024:
                os.replace(self.jsonl_path, self.jsonl_path + ".old")
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # лог не должен ломать основной поток ответа

    # ------------------------------------------------------------ статистика

    def get_stats(self, period_days: int = 7) -> Dict[str, Any]:
        """
        Статистика из логов за период (для команды stats и веб-панели).

        Возвращает: количество запросов, принятых/отклонённых с причинами,
        долю кеша, среднюю длительность, токены по моделям, ошибки.
        """
        since = f"datetime('now', '-{period_days} days')"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT COUNT(*) FROM request_log WHERE event = ? AND created_at >= {since}",
            (EVENT_RECEIVED,))
        total = cursor.fetchone()[0]

        cursor.execute(
            f"SELECT COUNT(*) FROM request_log WHERE event = ? AND created_at >= {since}",
            (EVENT_ACCEPTED,))
        accepted = cursor.fetchone()[0]

        cursor.execute("""
            SELECT reason, COUNT(*) FROM request_log
            WHERE event = ? AND created_at >= {since}
            GROUP BY reason
        """.format(since=since), (EVENT_REJECTED,))
        rejected_by_reason = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute(f"""
            SELECT
                SUM(from_cache),
                COUNT(*),
                AVG(duration_ms)
            FROM request_log
            WHERE event = ? AND created_at >= {since}
        """, (EVENT_ANSWER_READY,))
        cached, answered, avg_duration = cursor.fetchone()

        cursor.execute(f"""
            SELECT model,
                   SUM(prompt_tokens), SUM(completion_tokens)
            FROM request_log
            WHERE event = ? AND created_at >= {since} AND model IS NOT NULL
            GROUP BY model
        """, (EVENT_ANSWER_READY,))
        tokens_by_model = {
            row[0]: {"prompt": row[1] or 0, "completion": row[2] or 0}
            for row in cursor.fetchall()
        }

        cursor.execute(f"""
            SELECT COUNT(*) FROM request_log
            WHERE error IS NOT NULL AND created_at >= {since}
        """)
        errors = cursor.fetchone()[0]

        # Точные границы периода — по реальным событиям в логе (для шапки /stats)
        cursor.execute(f"""
            SELECT MIN(created_at), MAX(created_at) FROM request_log
            WHERE event = ? AND created_at >= {since}
        """, (EVENT_RECEIVED,))
        first_at, last_at = cursor.fetchone()

        # Разбивка запросов по источникам (каналам) — поле source из схемы урока
        cursor.execute(f"""
            SELECT COALESCE(source, 'unknown'), COUNT(*) FROM request_log
            WHERE event = ? AND created_at >= {since}
            GROUP BY source
        """, (EVENT_RECEIVED,))
        by_source = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        cache_share = round(100.0 * cached / answered, 1) if answered else 0.0

        return {
            "period_days": period_days,
            "period_start": first_at[:10] if first_at else None,
            "period_end": last_at[:10] if last_at else None,
            "total_requests": total,
            "accepted": accepted,
            "rejected": sum(rejected_by_reason.values()),
            "rejected_by_reason": rejected_by_reason,
            "by_source": by_source,
            "answered": answered or 0,
            "cache_hits": cached or 0,
            "cache_share_pct": cache_share,
            "avg_duration_ms": round(avg_duration) if avg_duration else None,
            "tokens_by_model": tokens_by_model,
            "errors": errors,
        }

    def get_recent(self, limit: int = 25) -> Dict[str, Any]:
        """
        Последние события конвейера (для веб-панели оператора).

        Возвращает события в обратном хронологическом порядке: время,
        событие, источник, замаскированный фрагмент вопроса, из кеша,
        длительность, ошибка. Полные тексты пользователей сюда не попадают
        (PII-правило урока) — только замаскированный фрагмент до 120 символов.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT created_at, event, COALESCE(source, 'unknown'),
                   query_masked, from_cache, duration_ms, reason, error
            FROM request_log
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        events = []
        for created_at, event, source, query, cached, ms, reason, error in rows:
            events.append({
                "time": (created_at or "")[:16],
                "event": event,
                "source": source,
                "query": (query[:120] + "…") if query and len(query) > 120 else query,
                "from_cache": bool(cached) if cached is not None else None,
                "duration_ms": ms,
                "reason": reason,
                "error": error,
            })
        return {"events": events}


if __name__ == "__main__":
    # Демонстрация конвейера на искусственном запросе
    logger = RequestLogger(db_path="test_logs.db", jsonl_path="test_logs.jsonl")
    rid = logger.new_request()
    logger.log(rid, EVENT_RECEIVED, query="Позвоните мне по +7 912 345 67 89 насчёт аккредитива")
    logger.log(rid, EVENT_ACCEPTED)
    logger.log(rid, EVENT_STARTED)
    logger.log(rid, EVENT_ANSWER_READY, model="qwen3.7-max", prompt_tokens=812,
               completion_tokens=210, duration_ms=1830)
    logger.log(rid, EVENT_SENT, duration_ms=1831)

    rejected = logger.new_request()
    reason = logger.validate("")
    logger.log(rejected, EVENT_RECEIVED, query="")
    logger.log(rejected, EVENT_REJECTED, reason=reason)

    print("Статистика:", json.dumps(logger.get_stats(), ensure_ascii=False, indent=2))
    os.remove("test_logs.db")
    os.remove("test_logs.jsonl")