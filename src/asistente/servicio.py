"""Servicio de consulta: une la guardia de entrada, la caché, el grafo de agentes, la validación
de la salida y la auditoría. Lo usan la API y los scripts de evaluación.
"""

import logging
import re
import time
import uuid

from asistente.agentes.grafo import citas_invalidas, construir_grafo
from asistente.cache import CacheRespuestas
from asistente.config import cargar_config, ruta
from asistente.llm import ClienteLLM, ErrorLLM
from asistente.rag.indice import Codificador, Indice
from asistente.rag.recuperador import RecuperadorHibrido, Reordenador
from asistente.seguridad.auditoria import Auditoria
from asistente.seguridad.inyeccion import DetectorInyeccion, confirmador_llm
from asistente.seguridad.sanitizar import limpiar

log = logging.getLogger(__name__)

_CITA = re.compile(r"\[(\d+)\]")
MENSAJE_BLOQUEO = ("La consulta fue bloqueada por el control de seguridad porque parece contener "
                   "instrucciones dirigidas al sistema. Reformulala como una pregunta sobre la literatura.")


class ErrorEntrada(ValueError):
    """La pregunta no cumple los límites de la API (vacía o demasiado larga)."""


class Servicio:
    def __init__(self, cfg: dict | None = None, llm: ClienteLLM | None = None, cargar_modelos: bool = True,
                 usar_verificador: bool = True):
        self.cfg = cfg or cargar_config()
        c_ind, c_rec, c_seg = self.cfg["indice"], self.cfg["recuperacion"], self.cfg["seguridad"]
        t0 = time.perf_counter()
        self.indice = Indice.cargar(ruta(c_ind["dir"]))
        self.codificador = Codificador(self.indice.modelo_embeddings)
        usa_reranker = c_rec.get("modo", "hibrido_rerank") == "hibrido_rerank"
        self.reordenador = Reordenador(c_rec["modelo_reranker"]) if cargar_modelos and usa_reranker else None
        self.recuperador = RecuperadorHibrido(self.indice, self.codificador, self.reordenador, c_rec)
        self.llm = llm or ClienteLLM(self.cfg["agentes"], self.cfg["precios_usd_por_millon"])
        self.detector = DetectorInyeccion(c_seg["modelo_inyeccion"], c_seg["umbral_inyeccion"],
                                          confirmador=confirmador_llm(self.llm)) if cargar_modelos else None
        self.grafo = construir_grafo(self.llm, self.recuperador, self.cfg["agentes"]["max_iteraciones"],
                                     usar_verificador)
        c_cache = self.cfg["cache"]
        self.cache = CacheRespuestas(ruta(c_cache["archivo"]), c_cache["ttl_horas"], c_cache["habilitada"])
        self.auditoria = Auditoria(ruta(c_seg["archivo_auditoria"]))
        log.info("Servicio listo en %.1f s (%d fragmentos)", time.perf_counter() - t0, len(self.indice.fragmentos))

    def consultar(self, pregunta: str, usuario: str = "local", usar_cache: bool = True) -> dict:
        id_consulta = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        limpia, invisibles = limpiar(pregunta or "")
        if not limpia:
            raise ErrorEntrada("La pregunta está vacía")
        if len(limpia) > self.cfg["seguridad"]["max_caracteres_pregunta"]:
            raise ErrorEntrada(f"La pregunta supera los {self.cfg['seguridad']['max_caracteres_pregunta']} caracteres")

        evento = {"id": id_consulta, "usuario": usuario, "pregunta": limpia[:300]}

        # La caché va antes de la guardia: solo guarda respuestas a preguntas que ya pasaron la
        # guardia con el mismo texto normalizado, y así una pregunta repetida no paga ninguna llamada.
        if usar_cache and (previa := self.cache.obtener(limpia)):
            salida = previa | {"id": id_consulta, "desde_cache": True,
                               "metricas": {"latencia_ms": (time.perf_counter() - t0) * 1000, "costo_usd": 0.0,
                                            "tokens_entrada": 0, "tokens_salida": 0, "llamadas_llm": 0}}
            self.auditoria.registrar(evento | {"estado": salida["estado"], "desde_cache": True})
            return salida

        guardia = self.detector.evaluar(limpia) if self.detector else {"bloquear": False}
        guardia["caracteres_invisibles"] = invisibles
        evento["guardia"] = guardia

        if guardia["bloquear"]:
            salida = {"id": id_consulta, "estado": "bloqueada", "respuesta": MENSAJE_BLOQUEO, "referencias": [],
                      "advertencias": [], "metricas": {"latencia_ms": (time.perf_counter() - t0) * 1000}}
            self.auditoria.registrar(evento | {"estado": "bloqueada"})
            return salida


        try:
            e = self.grafo.invoke({"pregunta": limpia, "usos": [], "pasos": []})
        except ErrorLLM as exc:
            log.error("Consulta %s: %s", id_consulta, exc)
            self.auditoria.registrar(evento | {"estado": "error", "error": str(exc)})
            return {"id": id_consulta, "estado": "error", "referencias": [], "advertencias": [],
                    "respuesta": "No se pudo generar la respuesta por un error del modelo. Probá de nuevo.",
                    "metricas": {"latencia_ms": (time.perf_counter() - t0) * 1000}}

        if guardia.get("uso"):
            e["usos"] = [guardia["uso"]] + e.get("usos", [])
        salida = self._armar_salida(id_consulta, e, t0)
        if salida["estado"] in ("respondida", "fuera_de_alcance"):
            self.cache.guardar(limpia, salida)
        self.auditoria.registrar(evento | {
            "estado": salida["estado"], "consultas": e.get("consultas"),
            "fragmentos": [f["id"] for f in e.get("fragmentos", [])],
            "veredicto": (e.get("verificacion") or {}).get("veredicto"),
            "iteraciones": e.get("iteracion", 0), "metricas": salida["metricas"],
            "usos": e.get("usos", []), "pasos": e.get("pasos", []),
        })
        return salida

    def _armar_salida(self, id_consulta: str, e: dict, t0: float) -> dict:
        """Validación de la salida: las referencias salen de los metadatos del índice, no del modelo."""
        fragmentos = e.get("fragmentos", [])
        respuesta = e.get("respuesta", "")
        advertencias = []
        invalidas = citas_invalidas(respuesta, len(fragmentos))
        if invalidas:
            respuesta = _CITA.sub(lambda m: "" if int(m.group(1)) in invalidas else m.group(0), respuesta)
            advertencias.append(f"Se quitaron citas a documentos inexistentes: {invalidas}")
        citadas = sorted({int(n) for n in _CITA.findall(respuesta)})
        referencias = []
        for n in citadas:
            f = fragmentos[n - 1]
            referencias.append({"n": n, "pmid": f["pmid"], "titulo": f["titulo"], "anio": f["anio"],
                                "revista": f["revista"], "seccion": f["seccion"], "fragmento": f["id"],
                                "url": f"https://pubmed.ncbi.nlm.nih.gov/{f['pmid']}/"})
        v = e.get("verificacion") or {}
        if e.get("estado") == "no_verificada":
            sin_respaldo = [a["texto"] for a in v.get("afirmaciones", []) if not a["respaldada"]]
            advertencias.append("El verificador no aprobó la respuesta después de "
                                f"{e.get('iteracion', 0)} revisiones. Afirmaciones sin respaldo: {sin_respaldo}")
        usos = e.get("usos", [])
        costos = [u["costo_usd"] for u in usos if u["costo_usd"] is not None]
        metricas = {
            "latencia_ms": round((time.perf_counter() - t0) * 1000),
            "tokens_entrada": sum(u["tokens_entrada"] for u in usos),
            "tokens_salida": sum(u["tokens_salida"] for u in usos),
            "costo_usd": round(sum(costos), 6) if costos else None,
            "llamadas_llm": len(usos),
            "iteraciones": e.get("iteracion", 0),
            "por_nodo_ms": _sumar_por_nodo(e.get("pasos", [])),
        }
        return {"id": id_consulta, "estado": e.get("estado", "error"), "respuesta": respuesta,
                "referencias": referencias, "advertencias": advertencias,
                "consultas": e.get("consultas", []), "veredicto": v.get("veredicto"),
                "fragmentos_recuperados": [f["id"] for f in fragmentos], "metricas": metricas,
                "desde_cache": False}


def _sumar_por_nodo(pasos: list[dict]) -> dict:
    total: dict[str, float] = {}
    for p in pasos:
        total[p["nodo"]] = round(total.get(p["nodo"], 0) + p["ms"])
    return total
