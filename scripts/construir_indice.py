"""Fragmenta el corpus, revisa cada fragmento con el detector de inyección y construye el índice.

Uso: python scripts/construir_indice.py
"""

import json
import logging
import time

from asistente.config import cargar_config, configurar_logging, ruta
from asistente.corpus.fragmentos import fragmentar
from asistente.corpus.jats import secciones
from asistente.llm import ClienteLLM
from asistente.rag.indice import Codificador, Indice
from asistente.seguridad.inyeccion import DetectorInyeccion, coincide_patron, confirmador_llm

log = logging.getLogger("construir_indice")


def main() -> None:
    configurar_logging()
    cfg = cargar_config()
    c_corpus, c_frag, c_ind, c_seg = cfg["corpus"], cfg["fragmentacion"], cfg["indice"], cfg["seguridad"]

    fragmentos = []
    with open(ruta(c_corpus["archivo_corpus"]), encoding="utf-8") as f:
        articulos = [json.loads(l) for l in f]
    for a in articulos:
        if a["texto_completo"]:
            secs = secciones((ruta(c_corpus["dir_crudos"]) / f"{a['pmid']}.xml").read_bytes())
            if not secs or secs[0][0] != "Abstract":
                secs.insert(0, ("Abstract", a["resumen"]))
            fuente = "texto_completo"
        else:
            secs, fuente = [("Abstract", a["resumen"])], "resumen"
        fragmentos += fragmentar(a, secs, fuente, c_frag["max_palabras"], c_frag["solape_palabras"],
                                 c_frag["min_palabras"])
    log.info("%d artículos -> %d fragmentos", len(articulos), len(fragmentos))

    t0 = time.time()
    detector = DetectorInyeccion(c_seg["modelo_inyeccion"], c_seg["umbral_inyeccion"])
    puntajes = detector.puntajes([f.texto for f in fragmentos])
    confirmar = confirmador_llm(ClienteLLM(cfg["agentes"], cfg["precios_usd_por_millon"]))
    for frag, p in zip(fragmentos, puntajes):
        if coincide_patron(frag.texto):
            frag.sospechoso = True
        elif p >= detector.umbral:
            # El clasificador solo no alcanza para descartar un fragmento: confirma el modelo auxiliar.
            frag.sospechoso, _ = confirmar(frag.texto)
            log.info("  %s marcado por el clasificador (%.3f); confirmado: %s", frag.id, p, frag.sospechoso)
    sospechosos = [(f, p) for f, p in zip(fragmentos, puntajes) if f.sospechoso]
    log.info("Detector de inyección: %d de %d fragmentos marcados como sospechosos (%.0f s)",
             len(sospechosos), len(fragmentos), time.time() - t0)
    for f, p in sospechosos[:10]:
        log.info("  sospechoso %s (%.3f): %s...", f.id, p, f.texto[:120])

    t0 = time.time()
    indice = Indice.construir(fragmentos, Codificador(c_ind["modelo_embeddings"]), c_ind["lote_embeddings"])
    log.info("Embeddings: %.0f s", time.time() - t0)
    indice.guardar(ruta(c_ind["dir"]))


if __name__ == "__main__":
    main()
