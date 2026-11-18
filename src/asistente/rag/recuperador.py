"""Recuperación híbrida: BM25 + embeddings densos, fusión RRF y reordenamiento con cross-encoder.

Los modos parciales (solo BM25, solo denso, híbrido sin reranker) existen para poder comparar
cada componente en la evaluación.
"""

import logging
import time
from dataclasses import dataclass, field

from asistente.corpus.fragmentos import Fragmento
from asistente.rag.indice import Codificador, Indice

log = logging.getLogger(__name__)

MODOS = ("bm25", "denso", "hibrido", "hibrido_rerank")


def rrf(listas: list[list[int]], k: float = 60.0) -> list[int]:
    """Reciprocal Rank Fusion: puntaje = suma de 1 / (k + rango) sobre todas las listas."""
    puntajes: dict[int, float] = {}
    for lista in listas:
        for rango, item in enumerate(lista, start=1):
            puntajes[item] = puntajes.get(item, 0.0) + 1.0 / (k + rango)
    return sorted(puntajes, key=puntajes.get, reverse=True)


@dataclass
class Resultado:
    fragmentos: list[Fragmento]
    puntajes: list[float]
    tiempos_ms: dict = field(default_factory=dict)
    descartados_sospechosos: int = 0


class Reordenador:
    def __init__(self, nombre_modelo: str):
        from sentence_transformers import CrossEncoder
        self.modelo = CrossEncoder(nombre_modelo, device="cpu")

    def puntuar(self, consulta: str, textos: list[str]) -> list[float]:
        return self.modelo.predict([(consulta, t) for t in textos], show_progress_bar=False).tolist()


class RecuperadorHibrido:
    def __init__(self, indice: Indice, codificador: Codificador, reordenador: Reordenador | None, cfg: dict):
        self.indice = indice
        self.codificador = codificador
        self.reordenador = reordenador
        self.cfg = cfg

    def buscar(self, consultas: list[str], modo: str | None = None, top_final: int | None = None,
               excluir_sospechosos: bool = True, max_por_articulo: int | None = None,
               consultas_rerank: list[str] | None = None) -> Resultado:
        """Busca con una o varias consultas (multi-query) y devuelve los mejores fragmentos.

        El reranker puntúa cada candidato contra consultas_rerank (por defecto, todas las consultas)
        y se queda con el máximo, así una reformulación buena no queda tapada por otra mala. El
        cross-encoder está entrenado en inglés: conviene pasarle solo las consultas en inglés.
        """
        modo = modo or self.cfg.get("modo", "hibrido_rerank")
        if modo not in MODOS:
            raise ValueError(f"Modo desconocido: {modo}")
        k = self.cfg["top_por_lista"]
        top_final = top_final or self.cfg["top_final"]
        max_por_articulo = max_por_articulo or self.cfg["max_por_articulo"]
        tiempos = {}

        t0 = time.perf_counter()
        listas = []
        if modo in ("denso", "hibrido", "hibrido_rerank"):
            vectores = self.codificador.codificar(consultas)
            listas += [self.indice.buscar_denso(v, k) for v in vectores]
        if modo in ("bm25", "hibrido", "hibrido_rerank"):
            listas += [self.indice.buscar_bm25(c, k) for c in consultas]
        candidatos = rrf(listas, self.cfg["rrf_k"]) if len(listas) > 1 else listas[0]
        tiempos["recuperacion"] = (time.perf_counter() - t0) * 1000

        descartados = 0
        if excluir_sospechosos:
            filtrados = [i for i in candidatos if not self.indice.fragmentos[i].sospechoso]
            descartados = len(candidatos) - len(filtrados)
            candidatos = filtrados

        if modo == "hibrido_rerank" and self.reordenador is not None:
            t0 = time.perf_counter()
            candidatos = candidatos[: self.cfg["top_rerank"]]
            textos = [self.indice.fragmentos[i].texto for i in candidatos]
            por_consulta = [self.reordenador.puntuar(c, textos) for c in (consultas_rerank or consultas)]
            puntajes = [max(col) for col in zip(*por_consulta)] if candidatos else []
            orden = sorted(range(len(candidatos)), key=lambda j: puntajes[j], reverse=True)
            candidatos = [candidatos[j] for j in orden]
            puntajes = [puntajes[j] for j in orden]
            tiempos["reranking"] = (time.perf_counter() - t0) * 1000
        else:
            puntajes = [1.0 / (r + 1) for r in range(len(candidatos))]

        elegidos, elegidos_p, por_articulo = [], [], {}
        for i, p in zip(candidatos, puntajes):
            frag = self.indice.fragmentos[i]
            if por_articulo.get(frag.pmid, 0) >= max_por_articulo:
                continue
            por_articulo[frag.pmid] = por_articulo.get(frag.pmid, 0) + 1
            elegidos.append(frag)
            elegidos_p.append(float(p))
            if len(elegidos) >= top_final:
                break
        return Resultado(elegidos, elegidos_p, tiempos, descartados)

    def ranking_ids(self, consultas: list[str], modo: str, k: int, consultas_rerank: list[str] | None = None) -> list[str]:
        """Ranking de ids sin límite por artículo, para las métricas de recuperación."""
        return [f.id for f in self.buscar(consultas, modo, top_final=k, max_por_articulo=k,
                                          consultas_rerank=consultas_rerank).fragmentos]
