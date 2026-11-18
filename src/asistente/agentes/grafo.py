"""Grafo de agentes (LangGraph).

    planificar ──(fuera de alcance)──────────────────────────────┐
        │                                                        │
    recuperar ◄──────(buscar_mas: consulta nueva)───────┐        │
        │                                               │        │
    redactar ◄───────(corregir: indicaciones)───────┐   │        │
        │                                           │   │        │
    verificar ──────────────────────────────────────┴───┘        │
        │ (aprobada o sin iteraciones)                           │
    finalizar ◄──────────────────────────────────────────────────┘

El planificador y el verificador son agentes con salida estructurada (modelo chico); el analista
redacta con el modelo grande. El verificador devuelve indicaciones al analista o pide otra
búsqueda, hasta max_iteraciones.
"""

import logging
import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from asistente.agentes import prompts
from asistente.agentes.esquemas import Plan, Verificacion
from asistente.llm import ClienteLLM
from asistente.rag.recuperador import RecuperadorHibrido

log = logging.getLogger(__name__)

_CITA = re.compile(r"\[(\d+)\]")


class Estado(TypedDict, total=False):
    pregunta: str
    plan: dict
    consultas: list[str]
    fragmentos: list[dict]
    respuesta: str
    verificacion: dict
    iteracion: int
    consulta_extra: str             # consulta pedida por el verificador, pendiente de buscar
    busquedas_extra: int
    estado: str                     # respondida | fuera_de_alcance | no_verificada | error
    usos: Annotated[list[dict], operator.add]
    pasos: Annotated[list[dict], operator.add]   # traza: nodo, duración, detalle


def citas_invalidas(respuesta: str, n_fragmentos: int) -> list[int]:
    """Citas a números de documento que no existen (control determinista, sin LLM)."""
    return sorted({int(n) for n in _CITA.findall(respuesta) if not 1 <= int(n) <= n_fragmentos})


def construir_grafo(llm: ClienteLLM, recuperador: RecuperadorHibrido, max_iteraciones: int = 2,
                    usar_verificador: bool = True):
    """usar_verificador=False arma la variante sin verificador, para medir cuánto aporta."""

    def planificar(e: Estado) -> dict:
        t0 = time.perf_counter()
        plan, uso = llm.estructurado("planificador", prompts.PLANIFICADOR,
                                     f"<pregunta>{e['pregunta']}</pregunta>", Plan)
        consultas = [c.strip() for c in plan.consultas if c.strip()][:3]
        return {"plan": plan.model_dump(), "consultas": consultas, "iteracion": 0,
                "usos": [uso.a_dict()],
                "pasos": [{"nodo": "planificar", "ms": (time.perf_counter() - t0) * 1000,
                           "en_alcance": plan.en_alcance, "consultas": consultas}]}

    def recuperar(e: Estado) -> dict:
        # Se busca con la pregunta original y con las reformulaciones (la combinación que mejor
        # rindió en la evaluación), y el reranker, entrenado en inglés, solo ve consultas en inglés.
        # En una búsqueda pedida por el verificador se busca solo con la consulta nueva (mezclada con
        # las demás se diluye en la fusión) y sus fragmentos se suman a los que ya había.
        t0 = time.perf_counter()
        extra = e.get("consulta_extra", "")
        if extra:
            r = recuperador.buscar([extra], top_final=4)
        else:
            en_ingles = e["plan"].get("idioma", "es") == "en"
            r = recuperador.buscar([e["pregunta"]] + e["consultas"],
                                   consultas_rerank=e["consultas"] + ([e["pregunta"]] if en_ingles else []))
        nuevos = [f.a_dict() | {"puntaje": round(p, 4)} for f, p in zip(r.fragmentos, r.puntajes)]
        if extra:
            previos = e.get("fragmentos", [])
            vistos = {f["id"] for f in previos}
            nuevos = (previos + [f for f in nuevos if f["id"] not in vistos])[:10]
        return {"fragmentos": nuevos, "consulta_extra": "",
                "busquedas_extra": e.get("busquedas_extra", 0) + (1 if extra else 0),
                "pasos": [{"nodo": "recuperar", "ms": (time.perf_counter() - t0) * 1000,
                           "ids": [f.id for f in r.fragmentos], "tiempos_ms": r.tiempos_ms,
                           "consulta_extra": extra, "descartados_sospechosos": r.descartados_sospechosos}]}

    def redactar(e: Estado) -> dict:
        t0 = time.perf_counter()
        idioma = e["plan"].get("idioma", "es")
        mensaje = (f"{prompts.bloque_documentos(e['fragmentos'])}\n\n"
                   f"<pregunta>{e['pregunta']}</pregunta>\n\nIdioma de la respuesta: {idioma}")
        v = e.get("verificacion")
        if v and v.get("veredicto") == "corregir" and e.get("respuesta"):
            observadas = [a for a in v["afirmaciones"] if not a["respaldada"]]
            detalle = "\n".join(f"- {a['texto']}: {a['comentario']}" for a in observadas)
            mensaje += (f"\n\nTu respuesta anterior:\n<respuesta>{e['respuesta']}</respuesta>\n\n"
                        f"El verificador la observó:\n{v['comentario_general']}\n{detalle}\n\n"
                        "Reescribí la respuesta corrigiendo esos puntos.")
        texto, uso = llm.texto("analista", prompts.ANALISTA, mensaje)
        return {"respuesta": texto, "usos": [uso.a_dict()],
                "pasos": [{"nodo": "redactar", "ms": (time.perf_counter() - t0) * 1000,
                           "iteracion": e.get("iteracion", 0)}]}

    def verificar(e: Estado) -> dict:
        t0 = time.perf_counter()
        invalidas = citas_invalidas(e["respuesta"], len(e["fragmentos"]))
        mensaje = (f"{prompts.bloque_documentos(e['fragmentos'])}\n\n"
                   f"<pregunta>{e['pregunta']}</pregunta>\n\n<respuesta>{e['respuesta']}</respuesta>")
        if invalidas:
            mensaje += f"\n\nAtención: la respuesta cita documentos inexistentes: {invalidas}."
        if e.get("busquedas_extra", 0):
            mensaje += "\n\nYa se hizo una búsqueda adicional."
        v, uso = llm.estructurado("verificador", prompts.VERIFICADOR, mensaje, Verificacion)
        v = v.model_dump()
        if invalidas and v["veredicto"] == "aprobada":
            v["veredicto"] = "corregir"
            v["comentario_general"] += f" Se citan documentos inexistentes: {invalidas}."
        return {"verificacion": v, "iteracion": e.get("iteracion", 0) + 1, "usos": [uso.a_dict()],
                "pasos": [{"nodo": "verificar", "ms": (time.perf_counter() - t0) * 1000,
                           "veredicto": v["veredicto"], "citas_invalidas": invalidas,
                           "no_respaldadas": sum(not a["respaldada"] for a in v["afirmaciones"])}]}

    def preparar_busqueda(e: Estado) -> dict:
        nueva = e["verificacion"].get("consulta_adicional", "").strip() or e["consultas"][0]
        return {"consulta_extra": nueva,
                "pasos": [{"nodo": "preparar_busqueda", "ms": 0.0, "consulta_nueva": nueva}]}

    def finalizar(e: Estado) -> dict:
        if not e["plan"]["en_alcance"]:
            return {"estado": "fuera_de_alcance",
                    "respuesta": "La pregunta queda fuera del alcance del asistente, que responde sobre la "
                                 "literatura de conectividad EEG, confiabilidad test-retest y bilingüismo. "
                                 f"Motivo: {e['plan']['motivo']}"}
        if not usar_verificador:
            return {"estado": "respondida"}
        aprobada = e["verificacion"]["veredicto"] == "aprobada"
        return {"estado": "respondida" if aprobada else "no_verificada"}

    def tras_planificar(e: Estado) -> str:
        return "recuperar" if e["plan"]["en_alcance"] and e["consultas"] else "finalizar"

    def tras_verificar(e: Estado) -> str:
        veredicto = e["verificacion"]["veredicto"]
        if veredicto == "aprobada" or e["iteracion"] >= max_iteraciones:
            return "finalizar"
        return "redactar" if veredicto == "corregir" else "preparar_busqueda"

    g = StateGraph(Estado)
    for nombre, nodo in [("planificar", planificar), ("recuperar", recuperar), ("redactar", redactar),
                         ("verificar", verificar), ("preparar_busqueda", preparar_busqueda),
                         ("finalizar", finalizar)]:
        g.add_node(nombre, nodo)
    g.add_edge(START, "planificar")
    g.add_conditional_edges("planificar", tras_planificar, ["recuperar", "finalizar"])
    g.add_edge("recuperar", "redactar")
    g.add_edge("redactar", "verificar" if usar_verificador else "finalizar")
    g.add_conditional_edges("verificar", tras_verificar, ["finalizar", "redactar", "preparar_busqueda"])
    g.add_edge("preparar_busqueda", "recuperar")
    g.add_edge("finalizar", END)
    return g.compile()
