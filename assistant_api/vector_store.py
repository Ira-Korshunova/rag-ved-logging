"""
Модуль работы с векторным хранилищем ChromaDB.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая коллекция '{collection_name}'")
        
        # Режим эмбеддингов: "api" (OpenAI-совместимый: DashScope/Qwen и т.п.)
        # или "local" (sentence-transformers, модель качается с Hugging Face).
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "api")

        if self.embedding_provider == "local":
            # Локальная модель — ленивая загрузка при первом эмбеддинге.
            self.openai_client = None
            self._local_st_model = None
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        else:
            # OpenAI-совместимый клиент для создания embeddings.
            # Работает как с OpenAI, так и с любым совместимым API (DashScope/Qwen и т.п.)
            # через OPENAI_BASE_URL из .env.
            client_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url
            self.openai_client = OpenAI(**client_kwargs)

            # Модель эмбеддингов (по умолчанию text-embedding-v3 для DashScope/Qwen)
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def load_documents(self, file_path: str):
        """
        Загрузка документов из файла в векторное хранилище.
        
        Args:
            file_path: путь к файлу с документами
        """
        # Проверка существования файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")
        
        # Чтение файла
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Разбиение на чанки
        chunks = self._chunk_text(text)
        print(f"Текст разбит на {len(chunks)} чанков")
        
        # Проверка, не загружены ли уже документы
        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return
        
        # Создание embeddings и добавление в ChromaDB
        documents = []
        ids = []
        embeddings = []
        metadatas = []
        source_name = os.path.basename(file_path)

        for i, chunk in enumerate(chunks):
            # Создание embedding через OpenAI
            embedding = self._create_embedding(chunk)

            documents.append(chunk)
            ids.append(f"doc_{i}")
            embeddings.append(embedding)
            metadatas.append({"source": source_name})

            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(chunks)} чанков")

        # Добавление в ChromaDB батчами
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        print(f"Загружено {len(chunks)} документов в коллекцию '{self.collection_name}'")

    def load_documents_from_folder(self, folder_path: str, extensions=(".txt", ".md")):
        """
        Загрузка ВСЕХ документов из папки в векторное хранилище.

        Каждый файл читается целиком, разбивается на чанки и индексируется.
        В id чанка кодируется имя файла-источника, чтобы можно было
        отличать документы друг от друга.

        Args:
            folder_path: путь к папке с документами
            extensions: кортеж расширений файлов, которые индексируем
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Папка {folder_path} не найдена")

        # Проверка, не загружены ли уже документы
        if self.collection.count() > 0:
            print(f"Документы уже загружены в коллекцию ({self.collection.count()} чанков)")
            return

        files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(extensions)
        ]

        if not files:
            raise FileNotFoundError(f"В папке {folder_path} нет файлов {extensions}")

        print(f"Найдено {len(files)} файлов для индексации:")
        for f in files:
            print(f"  - {os.path.basename(f)}")

        documents = []
        ids = []
        embeddings = []
        metadatas = []
        chunk_counter = 0

        for file_path in files:
            file_name = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                print(f"  [пропуск] {file_name} — пустой файл")
                continue

            chunks = self._chunk_text(text)
            print(f"  {file_name}: {len(chunks)} чанков")

            for chunk in chunks:
                embedding = self._create_embedding(chunk)
                documents.append(chunk)
                ids.append(f"{file_name}_{chunk_counter}")
                embeddings.append(embedding)
                metadatas.append({"source": file_name})
                chunk_counter += 1

        # Добавление в ChromaDB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        print(f"Загружено {chunk_counter} чанков из {len(files)} файлов в коллекцию '{self.collection_name}'")
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста.

        Режим задаётся EMBEDDING_PROVIDER:
            "local" — sentence-transformers (без внешних API и квот);
            "api"   — OpenAI-совместимый клиент (DashScope/Qwen и т.п.).

        Args:
            text: текст для векторизации

        Returns:
            вектор embeddings
        """
        if self.embedding_provider == "local":
            if self._local_st_model is None:
                from sentence_transformers import SentenceTransformer
                print(f"Загрузка локальной модели эмбеддингов {self.embedding_model}...")
                self._local_st_model = SentenceTransformer(self.embedding_model)
            return self._local_st_model.encode(text, normalize_embeddings=True).tolist()
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            
        Returns:
            список документов с метаданными
        """
        # Создание embedding для запроса
        query_embedding = self._create_embedding(query)
        
        # Поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Форматирование результатов
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            metas = results.get('metadatas') or [[]]
            for i in range(len(results['documents'][0])):
                meta = metas[0][i] if metas and i < len(metas[0]) and metas[0][i] else {}
                source = meta.get('source') if isinstance(meta, dict) else None
                # Фолбэк: источник не сохранён — пробуем вытащить из id
                if not source:
                    source = results['ids'][0][i].rsplit('_', 1)[0]
                documents.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'source': source,
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")

    # Загрузка всех документов из папки data/
    if os.path.isdir("data"):
        vector_store.load_documents_from_folder("data")

    # Поиск по ВЭД-вопросу
    results = vector_store.search("Что такое аккредитив во внешней торговле?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

