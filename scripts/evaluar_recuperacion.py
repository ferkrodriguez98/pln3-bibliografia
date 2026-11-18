"""Evalúa la recuperación por componentes: BM25, denso, híbrido (RRF) e híbrido + reranker, con la
pregunta original, con las consultas reescritas por el planificador (multi-query en inglés) o con
ambas. Si existe eval/relevancia_pool.json (scripts/juzgar_relevancia.py), también calcula las
métricas por fragmento con los juicios de relevancia por pooling.

Métricas a nivel fragmento y a nivel artículo, en total y por idioma de la pregunta.

Uso: python scripts/evaluar_recuperacion.py
"""

import argparse
import json
import logging
import time

from asistente.agentes import prompts
from asistente.agentes.esquemas import Plan
from asistente.config import RAIZ, cargar_config, configurar_logging, ruta
from asistente.llm import ClienteLLM
from asistente.metricas.recuperacion import percentil, resumen
from asistente.rag.indice import Codificador, Indice
from asistente.rag.recuperador import MODOS, RecuperadorHibrido, Reordenador

log = logging.getLogger("evaluar_recuperacion")
K = 10


def consultas_planificador(preguntas: list[dict], cfg: dict, archivo) -> dict[str, list[str]]:
    """Reescrituras del planificador, guardadas en disco para no pagarlas en cada corrida."""
    previas = json.loads(archivo.read_text()) if archivo.exists() else {}
    faltan = [p for p in preguntas if p["pregunta"] not in previas]
    if faltan:
        llm = ClienteLLM(cfg["agentes"], cfg["precios_usd_por_millon"])
        for p in faltan:
            plan, _ = llm.estructurado("planificador", prompts.PLANIFICADOR,
                                       f"<pregunta>{p['pregunta']}</pregunta>", Plan)
            previas[p["pregunta"]] = plan.consultas or [p["pregunta"]]
        archivo.write_text(json.dumps(previas, ensure_ascii=False, indent=2))
    return previas


def a_articulos(ranking: list[str]) -> list[str]:
    vistos = []
    for fid in ranking:
        pmid = fid.split(":")[0]
        if pmid not in vistos:
            vistos.append(pmid)
    return vistos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preguntas", default="eval/preguntas.json")
    ap.add_argument("--salida", default="eval/resultados/recuperacion.json")
    args = ap.parse_args()
    configurar_logging()
    cfg = cargar_config()
    preguntas = [p for p in json.loads((RAIZ / args.preguntas).read_text()) if p["tipo"] == "respondible"]
    indice = Indice.cargar(ruta(cfg["indice"]["dir"]))
    rec = RecuperadorHibrido(indice, Codificador(indice.modelo_embeddings),
                             Reordenador(cfg["recuperacion"]["modelo_reranker"]), cfg["recuperacion"])
    reescritas = consultas_planificador(preguntas, cfg, RAIZ / "eval" / "consultas_planificador.json")
    archivo_pool = RAIZ / "eval" / "relevancia_pool.json"
    pool = json.loads(archivo_pool.read_text()) if archivo_pool.exists() else {}

    resultados = []
    for modo in MODOS:
        variantes = ["original", "planificador", "original+planificador"]
        if modo == "hibrido_rerank":
            variantes.append("original+planificador, rerank en inglés")
        for variante in variantes:
            rank_f, rank_a, rel_f, rel_a, rel_pool, idiomas, lat = [], [], [], [], [], [], []
            for p in preguntas:
                combinadas = [p["pregunta"]] + reescritas[p["pregunta"]]
                consultas = {"original": [p["pregunta"]], "planificador": reescritas[p["pregunta"]]}.get(variante, combinadas)
                # Variante: el cross-encoder (en inglés) solo ve las consultas en inglés.
                rerank = None
                if variante.endswith("rerank en inglés"):
                    rerank = reescritas[p["pregunta"]] + ([p["pregunta"]] if p["idioma"] == "en" else [])
                t0 = time.perf_counter()
                ranking = rec.ranking_ids(consultas, modo, K, consultas_rerank=rerank)
                lat.append((time.perf_counter() - t0) * 1000)
                rank_f.append(ranking)
                rank_a.append(a_articulos(ranking))
                rel_f.append(set(p["fragmentos_relevantes"]))
                rel_pool.append(set(pool.get(p["id"], p["fragmentos_relevantes"])))
                rel_a.append(set(p["articulos_relevantes"]))
                idiomas.append(p["idioma"])
            fila = {"modo": modo, "consultas": variante, "n": len(preguntas),
                    "latencia_p50_ms": percentil(lat, 50), "latencia_p95_ms": percentil(lat, 95),
                    "fragmento": resumen(rank_f, rel_f), "articulo": resumen(rank_a, rel_a, ks=(1, 3, 5)),
                    "fragmento_pool": resumen(rank_f, rel_pool) if pool else None}
            for idioma in sorted(set(idiomas)):
                sel = [i for i, x in enumerate(idiomas) if x == idioma]
                fila[f"fragmento_{idioma}"] = resumen([rank_f[i] for i in sel], [rel_f[i] for i in sel])
                fila[f"articulo_{idioma}"] = resumen([rank_a[i] for i in sel], [rel_a[i] for i in sel], ks=(1, 3, 5))
            resultados.append(fila)
            log.info("%-15s %-22s frag Hit@5=%.2f | pool P@5=%.2f nDCG@5=%.2f | art Hit@3=%.2f | p50=%.0f ms",
                     modo, variante, fila["fragmento"]["Hit@5"],
                     (fila["fragmento_pool"] or {}).get("P@5", float("nan")),
                     (fila["fragmento_pool"] or {}).get("nDCG@5", float("nan")),
                     fila["articulo"]["Hit@3"], fila["latencia_p50_ms"])

    destino = RAIZ / args.salida
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=2))
    log.info("Resultados -> %s", destino)


if __name__ == "__main__":
    main()
