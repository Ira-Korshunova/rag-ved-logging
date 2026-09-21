"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.
"""

from typing import Dict, Any, List
import os
import time
from openai import OpenAI

from vector_store import VectorStore
from cache import RAGCache
from logging_pipeline import (
    RequestLogger,
    EVENT_RECEIVED, EVENT_ACCEPTED, EVENT_REJECTED,
    EVENT_STARTED, EVENT_ANSWER_READY, EVENT_SENT,
)


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""
    
    def __init__(self,
                 collection_name: str = "rag_collection",
                 cache_db_path: str = "rag_cache.db",
                 data_file: str = "data",
                 model: str = None):
        """
        Инициализация RAG pipeline.

        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_file: путь к файлу ИЛИ папке с документами (по умолчанию data/)
            model: модель LLM для генерации ответов. Если None — берётся MODEL_NAME из .env,
                   иначе gpt-4o-mini
        """
        # Проверка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")

        # Модель генерации: приоритет — аргумент, затем MODEL_NAME из .env, затем дефолт
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        # TOP_K из .env (сколько документов отдавать в контекст)
        self.top_k = int(os.getenv("TOP_K", "3"))

        # OpenAI-совместимый клиент (работает с OpenAI и DashScope/Qwen через OPENAI_BASE_URL)
        client_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        self.openai_client = OpenAI(**client_kwargs)

        # Инициализация компонентов
        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=os.getenv("CHROMA_PATH", "./chroma_db"),
        )

        # Загрузка документов, если коллекция пустая.
        # data_file может быть как папкой, так и одиночным файлом.
        if self.vector_store.collection.count() == 0:
            if os.path.isdir(data_file):
                print(f"Загрузка документов из папки {data_file}...")
                self.vector_store.load_documents_from_folder(data_file)
            elif os.path.isfile(data_file):
                print(f"Загрузка документов из файла {data_file}...")
                self.vector_store.load_documents(data_file)
            else:
                raise FileNotFoundError(
                    f"Не найден источник данных {data_file} (ни папка, ни файл)"
                )
        
        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path)

        # Логирование по конвейеру урока PEcf09 (5 событий на запрос)
        log_db_path = os.getenv("LOG_DB_PATH", "request_logs.db")
        self.logger = RequestLogger(db_path=log_db_path)
        print(f"Инициализация логирования... ({log_db_path})")

        print("RAG Pipeline инициализирован (API mode)")
    
    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Создание промпта для LLM с контекстом.
        
        Args:
            query: вопрос пользователя
            context_docs: релевантные документы из векторного хранилища
            
        Returns:
            сформированный промпт
        """
        # Формирование контекста из документов с указанием источника
        context_parts = []
        sources_used = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get('source', 'источник неизвестен')
            sources_used.append(source)
            context_parts.append(f"Документ {i} [источник: {source}]:\n{doc['text']}\n")

        context = "\n".join(context_parts)

        # Список уникальных источников для подсказки модели
        unique_sources = []
        for s in sources_used:
            if s not in unique_sources:
                unique_sources.append(s)
        sources_line = ", ".join(unique_sources) if unique_sources else "нет"

        # Создание промпта
        prompt = f"""Ты - полезный AI ассистент по внешнеэкономической деятельности (ВЭД). Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- В конце ответа укажи, из каких документов взята информация, в квадратных скобках, например: [источник: incoterms_2020.txt, ved_payments.txt]
- Доступные источники в контексте: {sources_line}
- Если в контексте нет информации для ответа, так и скажи и не придумывай
- Будь точным и кратким
- Отвечай на русском языке

Ответ:"""

        return prompt
    
    def _generate_answer(self, prompt: str) -> tuple:
        """
        Генерация ответа через OpenAI API.

        Args:
            prompt: промпт для модели

        Returns:
            кортеж (ответ, использование токенов) — токены нужны для логов
        """
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Низкая температура для более точных ответов
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        tokens = {
            "prompt": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion": getattr(usage, "completion_tokens", None) if usage else None,
        }
        return answer, tokens
    
    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Основной метод для обработки запроса пользователя через API.

        Поток (с логированием по конвейеру урока PEcf09):
        1. Проверка кеша
        2. Если в кеше нет - поиск в векторном хранилище
        3. Формирование промпта с контекстом
        4. Генерация ответа через LLM API
        5. Сохранение в кеш

        Каждое событие конвейера логируется: запрос получен → принят/отклонён
        (с причиной) → обработка начата → ответ подготовлен → ответ отправлен.

        Args:
            user_query: запрос пользователя
            use_cache: использовать ли кеш

        Returns:
            словарь с ответом и метаданными
        """
        print(f"\n{'='*60}")
        print(f"Запрос: {user_query}")
        print(f"{'='*60}")

        t_start = time.monotonic()
        request_id = self.logger.new_request()
        # Событие 1: пользователь отправил запрос
        self.logger.log(request_id, EVENT_RECEIVED, query=user_query)

        # Проверка запроса до обработки (отклонение с причиной)
        reason = self.logger.validate(user_query)
        if reason:
            print(f"[!] Запрос отклонён: {reason}")
            # Событие 2б: запрос отклонён, причина фиксируется в логе
            self.logger.log(request_id, EVENT_REJECTED, reason=reason)
            raise ValueError(f"Запрос отклонён: {reason}")

        # Событие 2а: запрос принят
        self.logger.log(request_id, EVENT_ACCEPTED)

        # Шаг 1: Проверка кеша
        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(user_query)

            if cached_result:
                print("[+] Ответ найден в кеше")
                duration_ms = int((time.monotonic() - t_start) * 1000)
                # События 3-5: из кеша ответ готов мгновенно
                self.logger.log(request_id, EVENT_STARTED)
                self.logger.log(request_id, EVENT_ANSWER_READY,
                                model="cache", from_cache=True,
                                duration_ms=duration_ms)
                self.logger.log(request_id, EVENT_SENT, duration_ms=duration_ms)
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": cached_result.get("context"),
                    "cached_at": cached_result.get("created_at"),
                    "request_id": request_id,
                }
            else:
                print("[-] Ответ не найден в кеше")

        # Событие 3: ассистент приступил к выполнению запроса
        self.logger.log(request_id, EVENT_STARTED)

        try:
            # Шаг 2: Поиск релевантных документов
            print("[*] Поиск релевантных документов через API...")
            context_docs = self.vector_store.search(user_query, top_k=self.top_k)
            print(f"[+] Найдено {len(context_docs)} релевантных документов")

            # Шаг 3: Формирование промпта
            print("[*] Формирование промпта...")
            prompt = self._create_prompt(user_query, context_docs)

            # Шаг 4: Генерация ответа через API
            print(f"[*] Генерация ответа через LLM ({self.model})...")
            answer, tokens = self._generate_answer(prompt)
            print("[+] Ответ получен от API")

        except Exception as e:
            # Ошибка обработки: ответ не отправлен — фиксируем на событии 5 с текстом ошибки
            duration_ms = int((time.monotonic() - t_start) * 1000)
            self.logger.log(request_id, EVENT_SENT, error=str(e),
                            duration_ms=duration_ms)
            raise

        duration_ms = int((time.monotonic() - t_start) * 1000)
        # Событие 4: ассистент подготовил ответ (модель, токены, длительность)
        self.logger.log(request_id, EVENT_ANSWER_READY, model=self.model,
                        prompt_tokens=tokens.get("prompt"),
                        completion_tokens=tokens.get("completion"),
                        duration_ms=duration_ms,
                        from_cache=False)

        # Шаг 5: Сохранение в кеш
        if use_cache:
            print("[*] Сохранение в кеш...")
            context_for_cache = [doc['text'] for doc in context_docs]
            self.cache.set(user_query, answer, context_for_cache)
            print("[+] Сохранено в кеш")

        # Событие 5: система отправила ответ
        self.logger.log(request_id, EVENT_SENT, duration_ms=duration_ms)

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "API",
            "request_id": request_id
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики системы.

        Returns:
            словарь со статистикой (векторное хранилище, кеш, модель, логи)
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "API",
            "requests_log": self.logger.get_stats(period_days=7),
        }


if __name__ == "__main__":
    # Тестирование RAG pipeline в API режиме
    import sys
    
    try:
        pipeline = RAGPipeline()
        
        # Тестовые запросы по ВЭД
        test_queries = [
            "Что такое аккредитив во внешней торговле?",
            "Чем отличается CIF от FOB по Incoterms 2020?",
            "Какие документы нужны для таможенного оформления импорта?"
        ]
        
        for query in test_queries:
            result = pipeline.query(query)
            print(f"\n{'='*60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print(f"{'='*60}\n")
        
        # Повторный запрос (должен быть из кеша)
        print("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        print(f"Из кеша: {result['from_cache']}")
        
        # Статистика
        stats = pipeline.get_stats()
        print(f"\nСтатистика системы:")
        print(f"Векторное хранилище: {stats['vector_store']}")
        print(f"Кеш: {stats['cache']}")
        print(f"Режим: {stats['mode']}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

