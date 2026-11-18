"""API del asistente bibliográfico.

Uso: uvicorn app.main:app --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from asistente.config import cargar_config, configurar_logging, ruta, variable_entorno
from asistente.seguridad.acceso import ControlAcceso
from asistente.seguridad.auditoria import hash_usuario
from asistente.servicio import ErrorEntrada, Servicio

log = logging.getLogger("api")


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    configurar_logging()
    cfg = cargar_config()
    claves = [c.strip() for c in (variable_entorno("APP_API_KEYS") or "").split(",") if c.strip()]
    app.state.acceso = ControlAcceso(claves, cfg["seguridad"]["consultas_por_minuto"])
    app.state.servicio = getattr(app.state, "servicio", None) or Servicio(cfg)
    yield


app = FastAPI(title="Asistente bibliográfico EEG", version="0.1.0", lifespan=ciclo_de_vida)


class Consulta(BaseModel):
    pregunta: str = Field(min_length=3, max_length=1000,
                          examples=["¿Qué tan confiable es el PLI entre sesiones con ojos cerrados?"])


def usuario_autorizado(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    acceso: ControlAcceso = request.app.state.acceso
    if not acceso.autenticar(x_api_key):
        raise HTTPException(status_code=401, detail="Clave de acceso inválida o ausente (header X-API-Key)")
    if not acceso.permitir(x_api_key):
        raise HTTPException(status_code=429, detail="Se superó el límite de consultas por minuto")
    return hash_usuario(x_api_key)


@app.get("/salud")
def salud(request: Request) -> dict:
    s: Servicio = request.app.state.servicio
    return {"estado": "ok", "fragmentos": len(s.indice.fragmentos),
            "articulos": len({f.pmid for f in s.indice.fragmentos}),
            "modelos": {"analista": s.cfg["agentes"]["modelo_analista"],
                        "auxiliar": s.cfg["agentes"]["modelo_auxiliar"]}}


@app.get("/corpus")
def corpus(usuario: str = Depends(usuario_autorizado)) -> list[dict]:
    with open(ruta(cargar_config()["corpus"]["archivo_corpus"]), encoding="utf-8") as f:
        return [{k: a[k] for k in ("pmid", "titulo", "anio", "revista", "texto_completo", "consultas")}
                for a in map(json.loads, f)]


@app.post("/consultas")
def consultar(consulta: Consulta, request: Request, usuario: str = Depends(usuario_autorizado)) -> dict:
    # Endpoint sincrónico: FastAPI lo corre en un hilo aparte y no bloquea el resto de la API.
    try:
        return request.app.state.servicio.consultar(consulta.pregunta, usuario=usuario)
    except ErrorEntrada as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        log.exception("Error no controlado en /consultas")
        raise HTTPException(status_code=500, detail="Error interno; quedó registrado en el log")
