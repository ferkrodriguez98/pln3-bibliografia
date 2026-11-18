"""Acceso a los modelos de Claude con contabilidad de tokens, costo y latencia por llamada.

Optimización de costos: cada rol usa el modelo más barato que alcanza para su tarea. El
analista (redacta la respuesta) usa el modelo grande; el planificador y el verificador usan
uno chico, con salida estructurada validada por Pydantic.
"""

import logging
import time
from dataclasses import asdict, dataclass

import anthropic
from pydantic import BaseModel, ValidationError

from asistente.config import variable_entorno

log = logging.getLogger(__name__)

# Modelos para los que se piden reintentos del lado del servidor si un clasificador de
# seguridad rechaza la consulta (un falso positivo no corta el servicio).
_CON_FALLBACK = {"claude-opus-5", "claude-opus-5-5", "claude-fable-5-1"}


class ErrorLLM(RuntimeError):
    """Falló la llamada al modelo (red, límite de uso, rechazo o salida inválida)."""


@dataclass
class Uso:
    rol: str
    modelo: str
    tokens_entrada: int
    tokens_salida: int
    tokens_cache_lectura: int
    tokens_cache_escritura: int
    costo_usd: float | None
    latencia_ms: float
    id_solicitud: str | None

    def a_dict(self) -> dict:
        return asdict(self)


class ClienteLLM:
    def __init__(self, cfg_agentes: dict, precios: dict, cliente: anthropic.Anthropic | None = None):
        self.cfg = cfg_agentes
        self.precios = precios
        self.cliente = cliente or anthropic.Anthropic(
            api_key=variable_entorno("ANTHROPIC_API_KEY"),
            timeout=cfg_agentes.get("timeout_s", 120), max_retries=2)

    def _modelo(self, rol: str) -> tuple[str, str, int]:
        if rol == "analista":
            return self.cfg["modelo_analista"], self.cfg["esfuerzo_analista"], self.cfg["max_tokens_analista"]
        esfuerzo = self.cfg.get(f"esfuerzo_{rol}", self.cfg["esfuerzo_auxiliar"])
        return self.cfg["modelo_auxiliar"], esfuerzo, self.cfg["max_tokens_auxiliar"]

    def _costo(self, modelo: str, u) -> float | None:
        p = self.precios.get(modelo)
        if not p or p.get("entrada") is None:
            return None
        lectura = getattr(u, "cache_read_input_tokens", 0) or 0
        escritura = getattr(u, "cache_creation_input_tokens", 0) or 0
        return (u.input_tokens * p["entrada"] + u.output_tokens * p["salida"]
                + lectura * p.get("cache_lectura", p["entrada"])
                + escritura * p.get("cache_escritura", p["entrada"])) / 1e6

    def _uso(self, rol: str, respuesta, t0: float) -> Uso:
        u = respuesta.usage
        return Uso(rol=rol, modelo=respuesta.model, tokens_entrada=u.input_tokens,
                   tokens_salida=u.output_tokens,
                   tokens_cache_lectura=getattr(u, "cache_read_input_tokens", 0) or 0,
                   tokens_cache_escritura=getattr(u, "cache_creation_input_tokens", 0) or 0,
                   costo_usd=self._costo(respuesta.model, u),
                   latencia_ms=(time.perf_counter() - t0) * 1000,
                   id_solicitud=getattr(respuesta, "_request_id", None))

    @staticmethod
    def _revisar_corte(respuesta) -> None:
        if respuesta.stop_reason == "refusal":
            detalle = getattr(respuesta, "stop_details", None)
            raise ErrorLLM(f"El modelo rechazó la consulta ({getattr(detalle, 'category', None)})")
        if respuesta.stop_reason == "max_tokens":
            raise ErrorLLM("La respuesta se cortó por max_tokens")

    def texto(self, rol: str, sistema: str, mensaje: str) -> tuple[str, Uso]:
        """Respuesta en texto libre."""
        modelo, esfuerzo, max_tokens = self._modelo(rol)
        params = dict(model=modelo, max_tokens=max_tokens, system=sistema,
                      output_config={"effort": esfuerzo},
                      messages=[{"role": "user", "content": mensaje}])
        t0 = time.perf_counter()
        try:
            if modelo in _CON_FALLBACK:
                r = self.cliente.beta.messages.create(
                    **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                r = self.cliente.messages.create(**params)
        except anthropic.APIError as exc:
            raise ErrorLLM(f"Falló la llamada a {modelo}: {exc}") from exc
        self._revisar_corte(r)
        texto = "".join(b.text for b in r.content if b.type == "text").strip()
        uso = self._uso(rol, r, t0)
        log.info("%s %s: %d+%d tokens, %.0f ms", rol, uso.modelo, uso.tokens_entrada, uso.tokens_salida,
                 uso.latencia_ms)
        return texto, uso

    def estructurado(self, rol: str, sistema: str, mensaje: str, esquema: type[BaseModel]) -> tuple[BaseModel, Uso]:
        """Respuesta validada contra un esquema Pydantic (salida estructurada de la API)."""
        modelo, esfuerzo, max_tokens = self._modelo(rol)
        t0 = time.perf_counter()
        try:
            r = self.cliente.messages.parse(
                model=modelo, max_tokens=max_tokens, system=sistema,
                messages=[{"role": "user", "content": mensaje}],
                output_format=esquema, output_config={"effort": esfuerzo})
        except anthropic.APIError as exc:
            raise ErrorLLM(f"Falló la llamada a {modelo}: {exc}") from exc
        except ValidationError as exc:
            # El SDK valida el JSON antes de devolver la respuesta: si llega cortado o mal formado,
            # se informa como error del modelo y no como excepción interna.
            raise ErrorLLM(f"{modelo} devolvió un JSON inválido para {esquema.__name__}") from exc
        self._revisar_corte(r)
        if r.parsed_output is None:
            raise ErrorLLM(f"{modelo} no devolvió una salida válida para {esquema.__name__}")
        uso = self._uso(rol, r, t0)
        log.info("%s %s: %d+%d tokens, %.0f ms", rol, uso.modelo, uso.tokens_entrada, uso.tokens_salida,
                 uso.latencia_ms)
        return r.parsed_output, uso
