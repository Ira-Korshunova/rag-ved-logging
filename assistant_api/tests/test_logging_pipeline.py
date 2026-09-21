"""
Тесты модуля логирования по конвейеру урока PEcf09.
Запуск: pytest tests/test_logging_pipeline.py -v
"""

import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_pipeline import (
    RequestLogger,
    EVENT_RECEIVED, EVENT_ACCEPTED, EVENT_REJECTED,
    EVENT_STARTED, EVENT_ANSWER_READY, EVENT_SENT,
    REASON_EMPTY, REASON_TOO_LONG,
)


@pytest.fixture
def logger(tmp_path):
    lg = RequestLogger(
        db_path=str(tmp_path / "logs.db"),
        jsonl_path=str(tmp_path / "logs.jsonl"),
        retention_days=90,
    )
    yield lg


def _count_events(logger, event=None):
    conn = sqlite3.connect(logger.db_path)
    cursor = conn.cursor()
    if event:
        cursor.execute("SELECT COUNT(*) FROM request_log WHERE event = ?", (event,))
    else:
        cursor.execute("SELECT COUNT(*) FROM request_log")
    n = cursor.fetchone()[0]
    conn.close()
    return n


class TestPipeline:
    """Конвейер из 5 событий пишется по цепочке."""

    def test_five_stage_pipeline(self, logger):
        rid = logger.new_request()
        logger.log(rid, EVENT_RECEIVED, query="Что такое аккредитив?")
        logger.log(rid, EVENT_ACCEPTED)
        logger.log(rid, EVENT_STARTED)
        logger.log(rid, EVENT_ANSWER_READY, model="qwen3.7-max",
                   prompt_tokens=100, completion_tokens=50, duration_ms=1200)
        logger.log(rid, EVENT_SENT, duration_ms=1210)

        assert _count_events(logger, EVENT_RECEIVED) == 1
        assert _count_events(logger, EVENT_ACCEPTED) == 1
        assert _count_events(logger, EVENT_STARTED) == 1
        assert _count_events(logger, EVENT_ANSWER_READY) == 1
        assert _count_events(logger, EVENT_SENT) == 1

    def test_events_share_request_id(self, logger):
        rid = logger.new_request()
        for event in (EVENT_RECEIVED, EVENT_ACCEPTED, EVENT_STARTED,
                      EVENT_ANSWER_READY, EVENT_SENT):
            logger.log(rid, event)
        conn = sqlite3.connect(logger.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT request_id) FROM request_log")
        n = cursor.fetchone()[0]
        conn.close()
        assert n == 1


class TestRejections:
    """Отклонения логируются с причиной (шаг 2 конвейера)."""

    def test_empty_query_rejected(self, logger):
        assert logger.validate("") == REASON_EMPTY
        assert logger.validate("   ") == REASON_EMPTY

    def test_too_long_query_rejected(self, logger):
        assert logger.validate("а" * 3000) == REASON_TOO_LONG

    def test_normal_query_accepted(self, logger):
        assert logger.validate("Чем отличается CIF от FOB?") is None

    def test_rejection_recorded_with_reason(self, logger):
        rid = logger.new_request()
        reason = logger.validate("")
        logger.log(rid, EVENT_RECEIVED, query="")
        logger.log(rid, EVENT_REJECTED, reason=reason)

        conn = sqlite3.connect(logger.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT event, reason FROM request_log WHERE reason IS NOT NULL")
        row = cursor.fetchone()
        conn.close()
        assert row == (EVENT_REJECTED, REASON_EMPTY)

    def test_rejection_visible_in_stats(self, logger):
        rid = logger.new_request()
        logger.log(rid, EVENT_RECEIVED, query="")
        logger.log(rid, EVENT_REJECTED, reason=REASON_EMPTY)
        stats = logger.get_stats(period_days=1)
        assert stats["rejected"] == 1
        assert stats["rejected_by_reason"] == {REASON_EMPTY: 1}


class TestPII:
    """Анонимность: PII маскируется до записи (правило урока)."""

    def test_phone_masked(self, logger):
        masked = logger.mask_pii("Позвоните по +7 912 345 67 89")
        assert "+7 912 345 67 89" not in masked
        assert "[телефон]" in masked

    def test_email_masked(self, logger):
        masked = logger.mask_pii("пишите на ivan@example.com")
        assert "ivan@example.com" not in masked
        assert "[email]" in masked

    def test_card_masked(self, logger):
        masked = logger.mask_pii("карта 4111 1111 1111 1111")
        assert "4111 1111 1111 1111" not in masked
        assert "[карта]" in masked

    def test_plain_query_not_masked(self, logger):
        text = "Что такое аккредитив во внешней торговле?"
        assert logger.mask_pii(text) == text

    def test_logged_query_is_masked(self, logger):
        rid = logger.new_request()
        query = "Мой телефон +7 912 345 67 89, вопрос про инвойс"
        logger.log(rid, EVENT_RECEIVED, query=query)
        conn = sqlite3.connect(logger.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query_masked FROM request_log WHERE event = ?", (EVENT_RECEIVED,))
        stored = cursor.fetchone()[0]
        conn.close()
        assert "89" not in stored.split("[телефон]")[0][-3:]
        assert "[телефон]" in stored


class TestRetention:
    """Срок хранения: cleanup удаляет записи старше лимита."""

    def test_cleanup_removes_old_records(self, tmp_path):
        db_path = str(tmp_path / "old.db")
        lg = RequestLogger(db_path=db_path, jsonl_path=str(tmp_path / "old.jsonl"),
                           retention_days=0)
        rid = lg.new_request()
        lg.log(rid, EVENT_RECEIVED, query="старый запрос")
        # искусственно старим запись на 100 дней
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE request_log SET created_at = datetime('now', '-100 days')")
        conn.commit()
        conn.close()

        deleted = lg.cleanup()
        assert deleted == 1
        assert _count_events(lg) == 0

    def test_cleanup_keeps_fresh_records(self, logger):
        rid = logger.new_request()
        logger.log(rid, EVENT_RECEIVED, query="свежий запрос")
        assert logger.cleanup() == 0
        assert _count_events(logger) == 1


class TestStats:
    """Статистика из логов: запросы, кеш, длительность, токены, ошибки."""

    def _simulate_answered(self, logger, from_cache, duration_ms, tokens):
        rid = logger.new_request()
        logger.log(rid, EVENT_RECEIVED, query="вопрос про ВЭД")
        logger.log(rid, EVENT_ACCEPTED)
        logger.log(rid, EVENT_STARTED)
        logger.log(rid, EVENT_ANSWER_READY, model="qwen3.7-max",
                   prompt_tokens=100, completion_tokens=50,
                   duration_ms=duration_ms, from_cache=from_cache)
        logger.log(rid, EVENT_SENT, duration_ms=duration_ms)

    def test_stats_totals(self, logger):
        self._simulate_answered(logger, from_cache=False, duration_ms=2000, tokens=None)
        self._simulate_answered(logger, from_cache=True, duration_ms=1000, tokens=None)
        stats = logger.get_stats(period_days=1)
        assert stats["total_requests"] == 2
        assert stats["answered"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_share_pct"] == 50.0
        assert stats["avg_duration_ms"] == 1500
        assert stats["errors"] == 0

    def test_stats_tokens_by_model(self, logger):
        self._simulate_answered(logger, from_cache=False, duration_ms=100, tokens=None)
        stats = logger.get_stats(period_days=1)
        assert "qwen3.7-max" in stats["tokens_by_model"]
        assert stats["tokens_by_model"]["qwen3.7-max"]["prompt"] == 100
        assert stats["tokens_by_model"]["qwen3.7-max"]["completion"] == 50


class TestJsonl:
    """JSONL-выгрузка: по строке на событие, валидный JSON."""

    def test_jsonl_lines_valid(self, logger):
        rid = logger.new_request()
        logger.log(rid, EVENT_RECEIVED, query="вопрос")
        logger.log(rid, EVENT_SENT)
        with open(logger.jsonl_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["event"] == EVENT_RECEIVED
        assert lines[1]["event"] == EVENT_SENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])