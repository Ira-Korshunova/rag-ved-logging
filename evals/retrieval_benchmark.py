# -*- coding: utf-8 -*-
"""
Бенчмарк качества поиска (retrieval) для сравнения локальных моделей эмбеддингов.

Задача: подтвердить числами, что замена BGE-M3 (~2,3 ГБ RAM) на компактную
paraphrase-multilingual-MiniLM-L12-v2 (~0,5 ГБ RAM) не ухудшает релевантность
поиска до уровня, непригодного для демо-витрины.

Методика:
  1. Эталонная таблица ETALON: вопрос пользователя -> файл, в котором
     содержится ответ (ground truth по документам базы assistant_api/data).
  2. Для каждой модели строится собственный индекс ChromaDB во временной
     папке (на рабочий индекс проекта бенчмарк не влияет).
  3. По каждому вопросу выполняется поиск top_k=5 через штатную
     VectorStore.search() — то есть измеряется реальный продукт-конвейер.
  4. Метрики:
     - hit@5 (факт)   — доля вопросов, где ключевой термин ответа попадает
                        хотя бы в один из топ-5 чанков (основная метрика:
                        LLM получает ответ в контексте);
     - hit@5 (по файлу) — доля вопросов, где эталонный файл в топ-5;
     - mean rank      — средний ранг первого чанка эталонного файла (1 = лучший);
     - query_ms       — среднее время одного запроса (embedding + поиск);
     - build_s        — время индексации всех документов;
     - dim / chunks   — размерность вектора и число чанков в индексе.

Замечание о границах методики: бенчмарк измеряет только этап поиска.
Качество генерации ответов определяется LLM (DeepSeek) и от выбора
модели эмбеддингов не зависит. Слабый поиск проявляется как ответ
"в контексте нет информации" либо как неполный ответ при пропуске нужного фрагмента.

Запуск из корня репозитория:
    .venv/bin/python evals/retrieval_benchmark.py
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "assistant_api"))
from vector_store import VectorStore  # noqa: E402

# Путь к исходным документам базы знаний (13 файлов)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "assistant_api", "data")
TOP_K = 5

# Эталонная таблица: (вопрос пользователя, ключевой термин ответа, файл-источник).
# Ключевой термин — факт-маркер, по которому проверяется, что в найденных чанках
# реально содержится ответ (независимо от того, в каком именно файле он лежит).
# Проверка факта важнее "угадывания файла": ответ может дублироваться в нескольких
# документах базы. Файл-источник служит вспомогательной метрикой ранга.
ETALON = [
    ("Какие документы нужны для импорта?", "документ", "Список_документов_для_ВЭД.md"),
    ("Что такое ИМ40?", "ИМ40", "customs_procedures.txt"),
    ("Чем CIF отличается от FOB?", "CIF", "incoterms_2020.txt"),
    ("Какие формы расчётов по внешнеторговому контракту бывают?", "расчёт", "ved_payments.txt"),
    ("Что такое аккредитив?", "аккредитив", "ved_payments.txt"),
    ("Как устроен код ТН ВЭД?", "групп", "tn_ved_structure.txt"),
    ("Какие есть таможенные процедуры?", "процедур", "customs_procedures.txt"),
    ("Что такое таможенный склад как процедура?", "склад", "customs_procedures.txt"),
    ("Что регулирует 173-ФЗ?", "173", "fz_173_currency.txt"),
    ("Что такое репатриация валютной выручки?", "репатриаци", "fz_173_currency.txt"),
    ("Что регулирует 289-ФЗ?", "289", "fz_289_customs.txt"),
    ("Как получить предварительное решение о классификации товара?", "классификаци", "tn_ved_structure.txt"),
    ("Что такое коносамент?", "коносамент", "logistics_dictionary.txt"),
    ("Что такое демередж?", "демередж", "словарь_МТ.txt"),
    ("Какова структура ТК ЕАЭС?", "глав", "tk_eaes.txt"),
    ("Что говорит ТК ЕАЭС о декларировании товаров?", "декларирован", "tk_eaes.txt"),
    ("Какие решения ЕЭК важны для ВЭД?", "ЕЭК", "eec_decisions_key.txt"),
    ("Расскажи пример кейса импорта из Китая", "кейс", "ved_case_example.txt"),
    ("Как проверить иностранного контрагента перед сделкой?", "контрагент", "ved_faq.txt"),
    ("Какие риски есть во внешнеэкономической деятельности?", "риск", "ved_faq.txt"),
    ("Что такое AGT — агент в перевозке?", "AGT", "словарь_МТ.txt"),
    ("Что такое извещение о прибытии судна?", "ARRIVAL", "словарь_МТ.txt"),
    ("Какие условия поставки по Инкотермс 2020?", "Инкотермс", "incoterms_2020.txt"),
    ("Что означает базис поставки DAP?", "DAP", "incoterms_2020.txt"),
    ("Какие виды банковских гарантий применяются в ВЭД?", "гарант", "ved_payments.txt"),
    ("Что такое СВХ — склад временного хранения?", "СВХ", "ved_case_example.txt"),
]


def run_model(label: str, model_id: str, env_extra: dict | None = None) -> dict:
    """Индексирует базу заданной моделью во временную папку и прогоняет эталон."""
    os.environ["EMBEDDING_PROVIDER"] = "local"
    os.environ["EMBEDDING_MODEL"] = model_id
    for k, v in (env_extra or {}).items():
        os.environ[k] = v

    tmp_dir = tempfile.mkdtemp(prefix="chroma_bench_")
    store = VectorStore(collection_name=f"bench_{label}", persist_directory=tmp_dir)

    t0 = time.time()
    stats = store.add_documents_from_folder(DATA_DIR)
    build_s = time.time() - t0

    hits = 0          # факт: в найденных чанках есть ключевой термин ответа
    file_hits = 0     # файл: эталонный файл попал в топ-k
    ranks = []
    latencies = []
    misses = []

    # Разогрев модели до замеров латентности
    store.search(ETALON[0][0], top_k=TOP_K)

    for question, term, expected in ETALON:
        t1 = time.time()
        results = store.search(question, top_k=TOP_K)
        latencies.append((time.time() - t1) * 1000)

        sources = [r["source"] for r in results]
        text_blob = " ".join(r["text"] for r in results).lower()
        term_found = term.lower() in text_blob
        rank = sources.index(expected) + 1 if expected in sources else None
        if term_found:
            hits += 1
        if rank is not None:
            file_hits += 1
            ranks.append(rank)
        if not term_found:
            misses.append({
                "question": question, "term": term,
                "sources": list(dict.fromkeys(sources)),
            })
    n = len(ETALON)
    return {
        "label": label,
        "model": model_id,
        "chunks": stats["total"],
        "build_s": round(build_s, 1),
        "hit@5": round(hits / n, 3),
        "hits": hits,
        "n": n,
        "file_hit@5": round(file_hits / n, 3),
        "mean_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "query_ms": round(sum(latencies) / len(latencies)),
        "dim": len(store._create_embedding("тест")),
        "misses": misses,
    }


def main() -> None:
    print("Бенчмарк поиска: эталонных вопросов {} (top_k={})\n".format(len(ETALON), TOP_K))
    results = [
        run_model("bge-m3", "BAAI/bge-m3"),
        run_model("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        run_model("e5-small", "intfloat/multilingual-e5-small", env_extra={
            "EMBEDDING_QUERY_PREFIX": "query: ",
            "EMBEDDING_DOCUMENT_PREFIX": "passage: ",
        }),
    ]

    for r in results:
        print("=" * 62)
        print(f"Модель {r['label']}: {r['model']}")
        print(f"  чанков в индексе : {r['chunks']}")
        print(f"  время индексации : {r['build_s']} с")
        print(f"  размерность вект.: {r['dim']}")
        print(f"  hit@5 (факт)     : {r['hits']}/{r['n']} ({r['hit@5']:.0%})")
        print(f"  hit@5 (по файлу) : {r['file_hit@5']:.0%}")
        print(f"  средний ранг     : {r['mean_rank']}")
        print(f"  время запроса    : {r['query_ms']} мс")
        for m in r["misses"]:
            print(f"  ПРОМАХ: «{m['question']}» (термин «{m['term']}» не в топ-{TOP_K}); топ-источники: {', '.join(m['sources'])}")
        print()

    return results


if __name__ == "__main__":
    main()