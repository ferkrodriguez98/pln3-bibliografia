"""Evalúa el sistema completo (agentes + RAG) sobre el conjunto de preguntas.

- Respondibles: recuperación del fragmento de origen, cita del artículo correcto, corrección frente
  a la respuesta de referencia y fidelidad a los documentos citados (juez independiente).
- Fuera de alcance: si el planificador las rechaza.
- Sin respuesta en el corpus: si el sistema reconoce que no hay evidencia en lugar de inventar.
- Agentes: veredictos del verificador, iteraciones, correcciones; costo, tokens y latencia.

El juez es un modelo distinto del verificador, para que no se evalúe a sí mismo.

Uso: python scripts/evaluar_sistema.py [--sin_verificador] [--limite 10]
"""

import argparse
import json
import logging
from collections import Counter
from statistics import mean

from pydantic import BaseModel, Field

from asistente.agentes.prompts import bloque_documentos
from asistente.config import RAIZ, cargar_config, configurar_logging, ruta
from asistente.llm import ClienteLLM
from asistente.metricas.recuperacion import percentil
from asistente.rag.indice import Indice
from asistente.servicio import Servicio

log = logging.getLogger("evaluar_sistema")

JUEZ = """Sos un evaluador experto en neurociencia y EEG. Evaluás la respuesta de un asistente de \
revisión bibliográfica. Recibís la pregunta, los documentos que el asistente citó, la respuesta y, \
si existe, una respuesta de referencia escrita a partir del artículo correcto.

- correccion (1 a 5): 5 si la respuesta contesta la pregunta y es consistente con la referencia; \
1 si es incorrecta o no contesta. Si no hay referencia, evaluá solo si contesta con lo que dicen los \
documentos.
- Contá las afirmaciones sustantivas de la respuesta y cuántas están respaldadas por los documentos \
citados junto a ellas.
- reconoce_falta_de_evidencia: true si la respuesta dice explícitamente que los documentos no \
alcanzan para responder (total o parcialmente)."""


class Juicio(BaseModel):
    correccion: int = Field(ge=1, le=5)
    afirmaciones: int
    afirmaciones_respaldadas: int
    reconoce_falta_de_evidencia: bool
    comentario: str


def juzgar(juez: ClienteLLM, p: dict, fila: dict, frag_por_id: dict) -> tuple[dict, float]:
    """El juez ve los fragmentos recuperados con la misma numeración que usó el analista."""
    recuperados = [frag_por_id[f] for f in fila["recuperados"] if f in frag_por_id]
    mensaje = (f"<pregunta>{p['pregunta']}</pregunta>\n\n{bloque_documentos(recuperados)}\n\n"
               f"<respuesta>{fila['respuesta']}</respuesta>\n\n"
               f"<referencia>{p.get('respuesta_referencia', 'sin referencia')}</referencia>")
    j, uso = juez.estructurado("analista", JUEZ, mensaje, Juicio)
    return j.model_dump(), uso.costo_usd or 0.0


def ejecutar(servicio: Servicio, p: dict, pool: dict) -> dict:
    s = servicio.consultar(p["pregunta"], usuario="evaluacion", usar_cache=False)
    fila = {"id": p["id"], "tipo": p["tipo"], "idioma": p["idioma"], "estado": s["estado"],
            "veredicto": s.get("veredicto"), "iteraciones": s["metricas"].get("iteraciones", 0),
            "latencia_ms": s["metricas"]["latencia_ms"], "costo_usd": s["metricas"].get("costo_usd") or 0,
            "tokens": s["metricas"].get("tokens_entrada", 0) + s["metricas"].get("tokens_salida", 0),
            "respuesta": s["respuesta"], "referencias": [r["fragmento"] for r in s["referencias"]],
            "recuperados": s.get("fragmentos_recuperados", [])}
    if p["tipo"] == "respondible":
        fila["recupero_fragmento"] = bool(set(p["fragmentos_relevantes"]) & set(fila["recuperados"]))
        relevantes = set(pool.get(p["id"], p["fragmentos_relevantes"]))
        fila["recupero_fragmento_relevante"] = bool(relevantes & set(fila["recuperados"]))
        fila["recupero_articulo"] = any(f.split(":")[0] in p["articulos_relevantes"] for f in fila["recuperados"])
        fila["cito_articulo"] = any(f.split(":")[0] in p["articulos_relevantes"] for f in fila["referencias"])
    return fila


def resumir(detalle: list[dict], variante: str, costo_juez: float) -> dict:
    resp = [d for d in detalle if d["tipo"] == "respondible"]
    juzgadas = [d for d in resp if "juicio" in d]
    afirm = sum(d["juicio"]["afirmaciones"] for d in juzgadas)
    con_llm = [d for d in detalle if d["estado"] not in ("bloqueada",)]
    media = lambda clave: mean(d[clave] for d in resp) if resp else None
    return {
        "variante": variante,
        "n_preguntas": len(detalle),
        "respondibles": {
            "n": len(resp),
            "recupero_fragmento_origen": media("recupero_fragmento"),
            "recupero_fragmento_relevante": media("recupero_fragmento_relevante"),
            "recupero_articulo": media("recupero_articulo"),
            "cito_articulo_correcto": media("cito_articulo"),
            "correccion_media_1a5": mean(d["juicio"]["correccion"] for d in juzgadas) if juzgadas else None,
            "correccion_distribucion": dict(sorted(Counter(d["juicio"]["correccion"] for d in juzgadas).items())),
            "fidelidad_afirmaciones": (sum(d["juicio"]["afirmaciones_respaldadas"] for d in juzgadas) / afirm
                                       if afirm else None),
            "reconoce_falta_de_evidencia": mean(d["juicio"]["reconoce_falta_de_evidencia"] for d in juzgadas)
            if juzgadas else None,
            "estados": dict(Counter(d["estado"] for d in resp)),
        },
        "fuera_de_alcance_rechazadas": _tasa(detalle, "fuera_de_alcance", lambda d: d["estado"] == "fuera_de_alcance"),
        "sin_respuesta_reconocidas": _tasa(detalle, "sin_respuesta", lambda d: d["estado"] == "fuera_de_alcance"
                                           or d.get("juicio", {}).get("reconoce_falta_de_evidencia", False)),
        "agentes": {
            "veredictos_finales": dict(Counter(d["veredicto"] for d in detalle if d["veredicto"])),
            "iteraciones": dict(Counter(d["iteraciones"] for d in detalle if d["veredicto"])),
        },
        "costo_latencia": {
            "costo_total_usd": sum(d["costo_usd"] for d in detalle),
            "costo_medio_usd": mean(d["costo_usd"] for d in con_llm) if con_llm else None,
            "tokens_medios": mean(d["tokens"] for d in con_llm) if con_llm else None,
            "latencia_p50_ms": percentil([d["latencia_ms"] for d in con_llm], 50),
            "latencia_p95_ms": percentil([d["latencia_ms"] for d in con_llm], 95),
            "costo_juez_usd": costo_juez,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preguntas", default="eval/preguntas.json")
    ap.add_argument("--sin_verificador", action="store_true")
    ap.add_argument("--limite", type=int, default=0, help="evaluar solo las primeras N preguntas")
    ap.add_argument("--rejuzgar", action="store_true",
                    help="no vuelve a consultar: juzga de nuevo las respuestas guardadas")
    args = ap.parse_args()
    configurar_logging()
    cfg = cargar_config()
    variante = "sin_verificador" if args.sin_verificador else "completo"
    destino = RAIZ / "eval" / "resultados" / f"sistema{'_sin_verificador' if args.sin_verificador else ''}.json"
    juez = ClienteLLM(cfg["agentes"], cfg["precios_usd_por_millon"])
    preguntas = json.loads((RAIZ / args.preguntas).read_text())
    if args.limite:
        preguntas = preguntas[:args.limite]
    por_id = {p["id"]: p for p in preguntas}
    archivo_pool = RAIZ / "eval" / "relevancia_pool.json"
    pool = json.loads(archivo_pool.read_text()) if archivo_pool.exists() else {}

    if args.rejuzgar:
        indice = Indice.cargar(ruta(cfg["indice"]["dir"]))
        frag_por_id = {f.id: f.a_dict() for f in indice.fragmentos}
        detalle = json.loads(destino.read_text())["detalle"]
        servicio = None
    else:
        servicio = Servicio(cfg, usar_verificador=not args.sin_verificador)
        frag_por_id = {f.id: f.a_dict() for f in servicio.indice.fragmentos}
        detalle = []

    costo_juez = 0.0
    for i, p in enumerate(preguntas):
        fila = detalle[i] if args.rejuzgar else ejecutar(servicio, p, pool)
        if args.rejuzgar and p["tipo"] == "respondible":
            relevantes = set(pool.get(p["id"], p["fragmentos_relevantes"]))
            fila["recupero_fragmento_relevante"] = bool(relevantes & set(fila["recuperados"]))
        fila.pop("juicio", None)
        if p["tipo"] != "fuera_de_alcance" and fila["estado"] in ("respondida", "no_verificada"):
            fila["juicio"], c = juzgar(juez, por_id[fila["id"]], fila, frag_por_id)
            costo_juez += c
        if not args.rejuzgar:
            detalle.append(fila)
        log.info("%s %-16s %-13s %s", p["id"], p["tipo"], fila["estado"], fila.get("juicio", {}).get("correccion", "-"))

    resumen = resumir(detalle, variante, costo_juez)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"resumen": resumen, "detalle": detalle}, ensure_ascii=False, indent=2))
    log.info("Resumen:\n%s", json.dumps(resumen, ensure_ascii=False, indent=2))


def _tasa(detalle: list[dict], tipo: str, condicion) -> float | None:
    sel = [d for d in detalle if d["tipo"] == tipo]
    return mean(bool(condicion(d)) for d in sel) if sel else None


if __name__ == "__main__":
    main()
