"""Índice híbrido persistente: embeddings densos (numpy) + BM25 (tokens en JSON, sin pickle).

La interfaz (buscar_denso / buscar_bm25) permite reemplazar el almacenamiento local por una
base vectorial administrada (Pinecone, Chroma) sin tocar el resto del pipeline.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from asistente.corpus.fragmentos import Fragmento

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the
this to was were which with we our these those between than not no can may also been into
their there such using used""".split())


def tokenizar(texto: str) -> list[str]:
    """Tokenización léxica para BM25: minúsculas, conserva siglas y términos con guion (test-retest)."""
    return [t for t in _TOKEN.findall(texto.lower()) if t not in _STOPWORDS]


class Codificador:
    """Embeddings densos normalizados con un modelo de sentence-transformers (corre en CPU)."""

    def __init__(self, nombre_modelo: str):
        from sentence_transformers import SentenceTransformer
        self.nombre = nombre_modelo
        self.modelo = SentenceTransformer(nombre_modelo, device="cpu")

    def codificar(self, textos: list[str], lote: int = 64) -> np.ndarray:
        return self.modelo.encode(textos, batch_size=lote, normalize_embeddings=True,
                                  show_progress_bar=False).astype(np.float32)


class Indice:
    def __init__(self, fragmentos: list[Fragmento], embeddings: np.ndarray, tokens: list[list[str]],
                 modelo_embeddings: str):
        if len(fragmentos) != len(embeddings) or len(fragmentos) != len(tokens):
            raise ValueError("Fragmentos, embeddings y tokens no tienen el mismo largo")
        self.fragmentos = fragmentos
        self.embeddings = embeddings
        self.tokens = tokens
        self.modelo_embeddings = modelo_embeddings
        self.bm25 = BM25Okapi(tokens)
        self.por_id = {f.id: i for i, f in enumerate(fragmentos)}

    @classmethod
    def construir(cls, fragmentos: list[Fragmento], codificador: Codificador, lote: int = 64) -> "Indice":
        textos = [f"{f.titulo}. {f.texto}" for f in fragmentos]
        embeddings = codificador.codificar(textos, lote)
        tokens = [tokenizar(t) for t in textos]
        return cls(fragmentos, embeddings, tokens, codificador.nombre)

    def guardar(self, directorio: Path) -> None:
        directorio.mkdir(parents=True, exist_ok=True)
        np.save(directorio / "embeddings.npy", self.embeddings)
        with open(directorio / "fragmentos.jsonl", "w", encoding="utf-8") as f:
            for frag, toks in zip(self.fragmentos, self.tokens):
                f.write(json.dumps({**frag.a_dict(), "tokens_bm25": toks}, ensure_ascii=False) + "\n")
        (directorio / "meta.json").write_text(json.dumps({
            "modelo_embeddings": self.modelo_embeddings,
            "n_fragmentos": len(self.fragmentos),
            "n_articulos": len({f.pmid for f in self.fragmentos}),
        }, indent=2))
        log.info("Índice guardado en %s (%d fragmentos)", directorio, len(self.fragmentos))

    @classmethod
    def cargar(cls, directorio: Path) -> "Indice":
        if not (directorio / "meta.json").exists():
            raise FileNotFoundError(f"No hay índice en {directorio}: correr scripts/construir_indice.py")
        meta = json.loads((directorio / "meta.json").read_text())
        fragmentos, tokens = [], []
        with open(directorio / "fragmentos.jsonl", encoding="utf-8") as f:
            for linea in f:
                d = json.loads(linea)
                tokens.append(d.pop("tokens_bm25"))
                fragmentos.append(Fragmento(**d))
        return cls(fragmentos, np.load(directorio / "embeddings.npy"), tokens, meta["modelo_embeddings"])

    def buscar_denso(self, vector_consulta: np.ndarray, k: int) -> list[int]:
        return np.argsort(-(self.embeddings @ vector_consulta))[:k].tolist()

    def buscar_bm25(self, consulta: str, k: int) -> list[int]:
        puntajes = self.bm25.get_scores(tokenizar(consulta))
        orden = np.argsort(-puntajes)[:k]
        return [int(i) for i in orden if puntajes[i] > 0]
