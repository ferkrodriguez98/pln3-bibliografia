"""Cobertura del contexto que recibe el analista según cuántos fragmentos se entregan y cuántos se
permiten por artículo, con la configuración de consultas del sistema y los juicios por pooling.

Uso: python scripts/analizar_contexto.py
"""

import json
from statistics import mean

from asistente.config import RAIZ, cargar_config, configurar_logging, ruta
from asistente.rag.indice import Codificador, Indice
from asistente.rag.recuperador import RecuperadorHibrido, Reordenador


def main() -> None:
    configurar_logging("WARNING")
    cfg = cargar_config()
    preguntas = [p for p in json.loads((RAIZ / "eval/preguntas.json").read_text()) if p["tipo"] == "respondible"]
    reescritas = json.loads((RAIZ / "eval/consultas_planificador.json").read_text())
    pool = json.loads((RAIZ / "eval/relevancia_pool.json").read_text())
    indice = Indice.cargar(ruta(cfg["indice"]["dir"]))
    rec = RecuperadorHibrido(indice, Codificador(indice.modelo_embeddings),
                             Reordenador(cfg["recuperacion"]["modelo_reranker"]), cfg["recuperacion"])
    filas = []
    for top, por_art in [(6, 2), (6, 3), (8, 2), (8, 3), (10, 3)]:
        for modo in ("hibrido", "hibrido_rerank"):
            hits, n_rel, origen = [], [], []
            for p in preguntas:
                consultas = [p["pregunta"]] + reescritas[p["pregunta"]]
                rerank = reescritas[p["pregunta"]] + ([p["pregunta"]] if p["idioma"] == "en" else [])
                ids = [f.id for f in rec.buscar(consultas, modo, top_final=top, max_por_articulo=por_art,
                                                consultas_rerank=rerank).fragmentos]
                relevantes = set(pool[p["id"]])
                hits.append(bool(relevantes & set(ids)))
                n_rel.append(len(relevantes & set(ids)))
                origen.append(p["fragmentos_relevantes"][0] in ids)
            filas.append({"top": top, "max_por_articulo": por_art, "modo": modo, "con_relevante": mean(hits),
                          "relevantes_medios": mean(n_rel), "con_origen": mean(origen)})
            print(f"top={top:2d} por_art={por_art} {modo:15s} con relevante={mean(hits):.2f} "
                  f"relevantes medios={mean(n_rel):.2f} con origen={mean(origen):.2f}")
    (RAIZ / "eval/resultados/contexto.json").write_text(json.dumps(filas, indent=2))


if __name__ == "__main__":
    main()
