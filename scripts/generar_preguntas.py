"""Genera el conjunto de evaluación con el método híbrido de la clase 2: un LLM escribe una pregunta
a partir de un fragmento muestreado y después una persona la revisa (campo "revisada").

El fragmento de origen es el relevante a nivel fragmento; su artículo, el relevante a nivel artículo.
Se agregan preguntas fuera de alcance y preguntas sin respuesta en el corpus, escritas a mano,
para evaluar al planificador y al verificador.

Uso: python scripts/generar_preguntas.py --n 40
"""

import argparse
import json
import logging
import random
import re

from pydantic import BaseModel, Field

from asistente.config import RAIZ, cargar_config, configurar_logging, ruta
from asistente.llm import ClienteLLM
from asistente.rag.indice import Indice

log = logging.getLogger("generar_preguntas")

SISTEMA = """Escribís preguntas de evaluación para un buscador de literatura científica sobre EEG. \
Recibís un fragmento de un artículo y escribís una pregunta que una tesista o un tesista haría y que \
ese fragmento responde de forma específica.

- La pregunta tiene que entenderse sola: no menciones "el fragmento", "el artículo" ni "este estudio". \
Si hace falta, nombrá la población, la medida o el diseño (por ejemplo, "en bebés de 10 meses").
- Parafraseá: no copies frases textuales del fragmento.
- Evitá preguntas genéricas que respondería cualquier artículo del tema.
- La respuesta de referencia resume en una o dos oraciones lo que dice el fragmento.
- Si el fragmento no permite una buena pregunta (es una lista de referencias, un agradecimiento, \
texto cortado), poné util=false."""

FUERA_DE_ALCANCE = [
    "¿Cuál es la capital de Australia?",
    "Escribime un poema sobre el otoño en Buenos Aires.",
    "¿Qué acciones me conviene comprar este año?",
    "Recomendame una receta de pan sin gluten.",
    "¿Quién ganó el último mundial de fútbol?",
]
SIN_RESPUESTA_EN_CORPUS = [
    "¿Qué tan confiable es la conectividad medida con fNIRS en adultos mayores bilingües?",
    "¿Qué efecto tiene la estimulación magnética transcraneal sobre la adquisición de una segunda lengua en niños?",
    "¿Cuál es la confiabilidad test-retest de la conectividad EEG en personas trilingües que hablan quechua?",
]


class PreguntaGenerada(BaseModel):
    util: bool
    pregunta: str = Field(description="La pregunta, en el idioma pedido")
    respuesta_referencia: str
    tipo: str = Field(description="resultado, metodo o concepto")


def muestrear(indice: Indice, n: int, semilla: int) -> list:
    """Hasta 2 fragmentos por artículo, priorizando resultados y discusión, sin fragmentos sospechosos."""
    rng = random.Random(semilla)
    candidatos = [f for f in indice.fragmentos if not f.sospechoso and len(f.texto.split()) >= 60]
    preferidos = re.compile(r"abstract|result|discussion|conclusion|method", re.I)
    rng.shuffle(candidatos)
    candidatos.sort(key=lambda f: 0 if preferidos.search(f.seccion) else 1)
    elegidos, por_articulo = [], {}
    for f in candidatos:
        if por_articulo.get(f.pmid, 0) < 2:
            elegidos.append(f)
            por_articulo[f.pmid] = por_articulo.get(f.pmid, 0) + 1
        if len(elegidos) == n:
            break
    return elegidos


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--proporcion_ingles", type=float, default=0.25)
    p.add_argument("--salida", default="eval/preguntas.json")
    args = p.parse_args()
    configurar_logging()
    cfg = cargar_config()
    indice = Indice.cargar(ruta(cfg["indice"]["dir"]))
    llm = ClienteLLM(cfg["agentes"], cfg["precios_usd_por_millon"])
    rng = random.Random(args.semilla)

    preguntas, costo = [], 0.0
    for frag in muestrear(indice, int(args.n * 1.2), args.semilla):
        idioma = "en" if rng.random() < args.proporcion_ingles else "es"
        mensaje = (f"Idioma de la pregunta: {'inglés' if idioma == 'en' else 'español rioplatense'}\n\n"
                   f"Título: {frag.titulo}\nSección: {frag.seccion}\n\n<fragmento>{frag.texto}</fragmento>")
        g, uso = llm.estructurado("analista", SISTEMA, mensaje, PreguntaGenerada)
        costo += uso.costo_usd or 0
        if not g.util:
            continue
        preguntas.append({"id": f"q{len(preguntas) + 1:02d}", "tipo": "respondible", "subtipo": g.tipo,
                          "idioma": idioma, "pregunta": g.pregunta, "respuesta_referencia": g.respuesta_referencia,
                          "fragmentos_relevantes": [frag.id], "articulos_relevantes": [frag.pmid],
                          "fragmento_origen": frag.texto, "revisada": False})
        if len(preguntas) == args.n:
            break
    for tipo, lista in (("fuera_de_alcance", FUERA_DE_ALCANCE), ("sin_respuesta", SIN_RESPUESTA_EN_CORPUS)):
        for texto in lista:
            preguntas.append({"id": f"q{len(preguntas) + 1:02d}", "tipo": tipo, "idioma": "es",
                              "pregunta": texto, "fragmentos_relevantes": [], "articulos_relevantes": [],
                              "revisada": True})
    destino = RAIZ / args.salida
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(preguntas, ensure_ascii=False, indent=2))
    log.info("%d preguntas -> %s (costo de generación: USD %.3f)", len(preguntas), destino, costo)


if __name__ == "__main__":
    main()
