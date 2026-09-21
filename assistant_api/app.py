"""
Консольное приложение для взаимодействия с RAG ассистентом (API mode).
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Загрузка переменных окружения из .env файла
# Ищем .env в корне проекта (на уровень выше)
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


def print_banner():
    """Вывод приветственного баннера."""
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    banner = f"""
╔══════════════════════════════════════════════════════════╗
║   RAG-ассистент по ВЭД (внешнеэкономической деятельности) ║
║   Retrieval-Augmented Generation · OpenAI-compatible API  ║
╠══════════════════════════════════════════════════════════╣
║   Модель LLM:      {model:<41}║
║   Эндпоинт:        {base_url:<41}║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("База знаний: документы по ВЭД (таможня, Incoterms, ТН ВЭД, расчёты)")
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' для просмотра статистики")
    print("Введите 'clear' для очистки кеша\n")


def print_response(result: dict):
    """
    Форматированный вывод ответа.
    
    Args:
        result: словарь с результатом запроса
    """
    print(f"\n{'─'*60}")
    print(f"📝 Вопрос: {result['query']}")
    print(f"{'─'*60}")
    
    # Индикатор источника ответа
    if result['from_cache']:
        print("💾 Источник: КЕШ")
        if 'cached_at' in result:
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"🌐 Источник: LLM ({result.get('model', 'model')})")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")
    
    print(f"\n💬 Ответ:\n{result['answer']}")
    
    # Показать контекст (опционально) — с указанием источника каждого документа
    if not result['from_cache'] and result.get('context_docs'):
        print(f"\n📚 Использованный контекст:")
        for i, doc in enumerate(result['context_docs'][:3], 1):  # Показываем первые 3
            preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
            print(f"   {i}. [{doc.get('source', '?')}] {preview}")
    
    print(f"{'─'*60}\n")


def print_stats(pipeline: RAGPipeline):
    """
    Вывод статистики системы.
    
    Args:
        pipeline: экземпляр RAG pipeline
    """
    stats = pipeline.get_stats()
    
    print(f"\n{'═'*60}")
    print("📊 СТАТИСТИКА СИСТЕМЫ")
    print(f"{'═'*60}")
    
    print("\n🗄️  Векторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")
    
    print("\n💾 Кеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats['cache']['oldest_entry']:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats['cache']['newest_entry']:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")

    # Статистика логов за неделю (конвейер урока PEcf09)
    log = stats.get('requests_log', {})
    print(f"\n📋 Логи запросов (за {log.get('period_days', 7)} дн.):")
    print(f"   Запросов: {log.get('total_requests', 0)}"
          f" | Принято: {log.get('accepted', 0)}"
          f" | Отклонено: {log.get('rejected', 0)}")
    if log.get('rejected_by_reason'):
        for reason, count in log['rejected_by_reason'].items():
            print(f"     - {reason}: {count}")
    print(f"   Ответов: {log.get('answered', 0)}"
          f" | Из кеша: {log.get('cache_hits', 0)}"
          f" ({log.get('cache_share_pct', 0)}%)")
    if log.get('avg_duration_ms') is not None:
        print(f"   Средняя длительность: {log['avg_duration_ms']} мс")
    if log.get('tokens_by_model'):
        for model, tok in log['tokens_by_model'].items():
            print(f"   Токены [{model}]: prompt {tok['prompt']}, completion {tok['completion']}")
    print(f"   Ошибок: {log.get('errors', 0)}")

    print(f"\n🤖 Модель: {stats['model']}")
    print(f"🌐 Режим: {stats['mode']}")
    print(f"{'═'*60}\n")


def main():
    """Главная функция приложения."""
    print_banner()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: переменная окружения OPENAI_API_KEY не установлена")
        print("\nУстановите её в файле .env в корне проекта:")
        print("  OPENAI_API_KEY=ваш-ключ")
        print("  OPENAI_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        print("  MODEL_NAME=qwen3.7-max")
        print("  EMBEDDING_MODEL=text-embedding-v3")
        sys.exit(1)

    try:
        # Инициализация RAG pipeline.
        # data_file="data" — папка со всеми ВЭД-документами (индексируются все .txt/.md).
        # model берётся из MODEL_NAME в .env (qwen3.7-max).
        print("🚀 Инициализация системы...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_file="data"
        )
        print("\n✅ Система готова к работе!\n")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        sys.exit(1)
    
    # Основной цикл взаимодействия
    while True:
        try:
            # Получение запроса от пользователя
            user_input = input("💭 Ваш вопрос: ").strip()
            
            # Обработка специальных команд
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if user_input.lower() == 'stats':
                print_stats(pipeline)
                continue
            
            if user_input.lower() == 'clear':
                confirm = input("⚠️  Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ['yes', 'y', 'да']:
                    pipeline.cache.clear()
                    print("✅ Кеш очищен")
                continue
            
            if not user_input:
                print("⚠️  Пожалуйста, введите вопрос\n")
                continue
            
            # Обработка запроса через RAG pipeline
            try:
                result = pipeline.query(user_input)
            except ValueError as ve:
                # Отклонение с причиной (пустой/слишком длинный запрос)
                print(f"\n🚫 {ve}\n")
                continue

            # Вывод результата
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()

