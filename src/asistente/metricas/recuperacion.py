"""Métricas de recuperación (las mismas definiciones que el código de la Teoría 1)."""

import math
from statistics import mean


def precision_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    return sum(d in relevantes for d in ranking[:k]) / k if k else 0.0


def recall_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    return sum(d in relevantes for d in ranking[:k]) / len(relevantes) if relevantes else 0.0


def hit_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    return float(any(d in relevantes for d in ranking[:k]))


def ndcg_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in relevantes)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevantes))))
    return dcg / idcg if idcg else 0.0


def mrr(ranking: list[str], relevantes: set[str]) -> float:
    for i, d in enumerate(ranking, start=1):
        if d in relevantes:
            return 1.0 / i
    return 0.0


def resumen(rankings: list[list[str]], relevantes: list[set[str]], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Promedios sobre todas las preguntas."""
    salida = {}
    for k in ks:
        salida[f"P@{k}"] = mean(precision_at_k(r, rel, k) for r, rel in zip(rankings, relevantes))
        salida[f"R@{k}"] = mean(recall_at_k(r, rel, k) for r, rel in zip(rankings, relevantes))
        salida[f"Hit@{k}"] = mean(hit_at_k(r, rel, k) for r, rel in zip(rankings, relevantes))
        salida[f"nDCG@{k}"] = mean(ndcg_at_k(r, rel, k) for r, rel in zip(rankings, relevantes))
    salida["MRR"] = mean(mrr(r, rel) for r, rel in zip(rankings, relevantes))
    return salida


def percentil(valores: list[float], p: float) -> float:
    if not valores:
        return float("nan")
    orden = sorted(valores)
    i = (len(orden) - 1) * p / 100
    bajo, alto = math.floor(i), math.ceil(i)
    return orden[bajo] + (orden[alto] - orden[bajo]) * (i - bajo)
