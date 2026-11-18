# Asistente bibliográfico multiagente sobre EEG

Proyecto de Procesamiento del Lenguaje Natural III (Maestría en Inteligencia Artificial, FIUBA).

Responde preguntas sobre la literatura de confiabilidad test-retest de la conectividad funcional
medida con EEG y sobre EEG en personas bilingües, con citas verificadas a los artículos. El corpus
sale de dos búsquedas en PubMed (las del trabajo de vigilancia de GTI) y se puede regenerar.

## Arquitectura

```
                    ┌──────────────────────── API (FastAPI) ────────────────────────┐
 pregunta ──► clave + límite de tasa ──► limpieza ──► detector de inyección ──► caché
                                                                                 │
                    ┌──────────────── grafo de agentes (LangGraph) ──────────────┤
                    │  planificador (Haiku 5.5): alcance + consultas en inglés   │
                    │        │                                                   │
                    │  recuperador: BM25 + embeddings ─► RRF ─► cross-encoder    │
                    │        │            ▲                                      │
                    │  analista (Opus 5): respuesta con citas [n]                │
                    │        │            ▲ corregir      ▲ buscar_mas           │
                    │  verificador (Haiku 5.5): ¿cada afirmación está respaldada?│
                    └────────┬───────────────────────────────────────────────────┘
                             ▼
     validación de salida (citas válidas, referencias desde metadatos) ──► auditoría (JSONL)
```

| Requisito | Dónde |
|---|---|
| API funcional | `app/main.py` (FastAPI): `POST /consultas`, `GET /corpus`, `GET /salud` |
| Dos o más agentes con comunicación dinámica | `src/asistente/agentes/grafo.py`: planificador, analista y verificador; el verificador devuelve indicaciones al analista o pide otra búsqueda |
| RAG con retriever + ranker y trazabilidad | `src/asistente/rag/`: BM25 + MiniLM, fusión RRF, cross-encoder; cada cita apunta a PMID, sección y fragmento |
| Seguridad | `src/asistente/seguridad/`: claves de acceso, límite de tasa, limpieza de caracteres invisibles, detector de inyección (pregunta y documentos), documentos escapados en el prompt, validación de citas, auditoría |
| Flujo modular | módulos desacoplados (corpus, índice, recuperador, agentes, servicio, API); el índice local se puede cambiar por una base vectorial administrada |
| Optimización de costo y latencia | modelo chico para planificar y verificar, caché de respuestas, contexto acotado (6 fragmentos, máx. 3 por artículo) |
| Métricas | `scripts/evaluar_*.py`: P@k, R@k, nDCG, MRR, corrección, fidelidad, costo, tokens y latencia |

## Instalación

Requiere Python 3.11 o 3.12. En Mac con procesador Intel, torch 2.2.2 es la última versión
disponible y obliga a usar numpy<2 (ya fijado en `requirements.txt`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env   # completar ANTHROPIC_API_KEY y APP_API_KEYS
```

## Ejecución

```bash
python scripts/construir_corpus.py     # PubMed + Europe PMC -> data/corpus.jsonl
python scripts/construir_indice.py     # fragmentos + detector de inyección + embeddings
uvicorn app.main:app --port 8000
```

```bash
curl -s localhost:8000/consultas -H "X-API-Key: $CLAVE" -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué tan confiable es el PLI entre sesiones con ojos cerrados?"}'
```

## Evaluación

```bash
python scripts/generar_preguntas.py --n 40   # conjunto de evaluación (revisar a mano: campo "revisada")
python scripts/evaluar_recuperacion.py       # BM25 / denso / híbrido / híbrido + reranker
python scripts/evaluar_sistema.py            # sistema completo, con juez independiente
python scripts/evaluar_sistema.py --sin_verificador
python scripts/evaluar_seguridad.py          # pruebas adversariales
pytest
```

Los resultados quedan en `eval/resultados/`.

## Resultados

Sobre 40 preguntas de evaluación (30 en español, 10 en inglés), con la configuración por defecto
(detalle en `docs/informe/informe.pdf`):

- El artículo correcto se recupera y se cita en el 92,5 % de las respuestas.
- El 98,5 % de las afirmaciones está respaldado por el documento citado. La corrección media frente
  a la respuesta de referencia es 3,95 sobre 5.
- Rechaza las 5 preguntas fuera de alcance y reconoce las 3 que el corpus no puede responder.
- Cuesta USD 0,041 por consulta, con una mediana de 26 s; sin el verificador, USD 0,024 y 12 s.
- La guardia bloqueó 11 de los 12 ataques directos (el restante lo rechazó el planificador) y no
  bloqueó ninguna pregunta legítima.
- El reranker no mejoró la recuperación frente a la fusión RRF. Se desactiva con
  `"modo": "hibrido"` en `config/config.json`.

## Estructura

```
app/main.py                 API
config/config.json          parámetros (modelos, fragmentación, recuperación, seguridad, precios)
src/asistente/
  corpus/                   clientes de PubMed y Europe PMC, parser JATS, fragmentación
  rag/                      índice (embeddings + BM25) y recuperador híbrido
  agentes/                  esquemas, prompts y grafo LangGraph
  seguridad/                limpieza, detector de inyección, acceso, auditoría
  metricas/                 métricas de recuperación
  llm.py                    cliente de Claude con contabilidad de tokens y costo
  servicio.py               guardia + caché + grafo + validación de salida + auditoría
scripts/                    construcción del corpus y del índice, evaluaciones
tests/                      tests de componentes
```
