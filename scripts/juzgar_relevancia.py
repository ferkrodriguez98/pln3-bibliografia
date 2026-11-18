"""Juicios de relevancia por pooling: para cada pregunta se juntan los 10 primeros fragmentos de todas
las configuraciones evaluadas y un juez decide cuáles responden la pregunta.

Con un solo fragmento relevante por pregunta (el de origen) las métricas por fragmento subestiman la
recuperación, porque los fragmentos vecinos y el resumen del mismo artículo también responden.

Uso: python scripts/juzgar_relevancia.py   (después de evaluar_recuperacion.py)
"""

import json
import logging
from html import escape

from pydantic import BaseModel, Field

from asistente.config import RAIZ, cargar_config, configurar_logging, ruta
from asistente.llm import ClienteLLM
from asistente.rag.indice import Codificador, Indice
from asistente.rag.recuperador import MODOS, RecuperadorHibrido, Reordenador

log = logging.getLogger("juzgar_relevancia")

SISTEMA = """Sos un evaluador experto en EEG y neurociencia. Recibís una pregunta y fragmentos de \
artículos. Marcá como relevantes los fragmentos que contienen información que responde la pregunta \
(total o en una parte sustancial). No alcanza con que traten el mismo tema."""


class Juicio(BaseModel):
    relevantes: list[str] = Field(description="ids de los fragmentos relevantes")


def main() -> None:
    configurar_logging()
    cfg = cargar_config()
    preguntas = [p for p in json.loads((RAIZ / "eval/preguntas.json").read_text()) if p["tipo"] == "respondible"]
    reescritas = json.loads((RAIZ / "eval/consultas_planificador.json").read_text())
    indice = Indice.cargar(ruta(cfg["indice"]["dir"]))
    rec = RecuperadorHibrido(indice, Codificador(indice.modelo_embeddings),
                             Reordenador(cfg["recuperacion"]["modelo_reranker"]), cfg["recuperacion"])
    juez = ClienteLLM(cfg["agentes"], cfg["precios_usd_por_millon"])
    archivo = RAIZ / "eval/relevancia_pool.json"
    juicios = json.loads(archivo.read_text()) if archivo.exists() else {}
    costo = 0.0
    for p in preguntas:
        if p["id"] in juicios:
            continue
        pool = []
        for modo in MODOS:
            for consultas in ([p["pregunta"]], reescritas[p["pregunta"]], [p["pregunta"]] + reescritas[p["pregunta"]]):
                for fid in rec.ranking_ids(consultas, modo, 10):
                    if fid not in pool:
                        pool.append(fid)
        frags = [indice.fragmentos[indice.por_id[f]] for f in pool]
        bloque = "\n".join(f'<fragmento id="{f.id}">{escape(f.titulo, quote=False)}. '
                           f'{escape(f.texto, quote=False)}</fragmento>' for f in frags)
        j, uso = juez.estructurado("analista", SISTEMA, f"<pregunta>{p['pregunta']}</pregunta>\n\n{bloque}", Juicio)
        costo += uso.costo_usd or 0
        validos = [f for f in j.relevantes if f in pool]
        juicios[p["id"]] = sorted(set(validos) | set(p["fragmentos_relevantes"]))
        log.info("%s: pool=%d relevantes=%d", p["id"], len(pool), len(juicios[p["id"]]))
        archivo.write_text(json.dumps(juicios, indent=2))
    log.info("Juicios -> %s (costo USD %.3f)", archivo, costo)


if __name__ == "__main__":
    main()
