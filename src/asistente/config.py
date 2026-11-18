"""Carga de la configuración externa (config/config.json) y de las variables de entorno (.env)."""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[2]
ARCHIVO_CONFIG = Path(os.getenv("ASISTENTE_CONFIG", RAIZ / "config" / "config.json"))


class ErrorConfiguracion(RuntimeError):
    """Falta un parámetro o una variable de entorno obligatoria."""


@lru_cache
def cargar_config() -> dict:
    """Lee config.json una sola vez por proceso y carga el .env de la raíz del proyecto."""
    load_dotenv(RAIZ / ".env")
    try:
        with open(ARCHIVO_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise ErrorConfiguracion(f"No se encontró {ARCHIVO_CONFIG}") from exc


def ruta(relativa: str) -> Path:
    """Convierte una ruta de config.json (relativa a la raíz del proyecto) en absoluta."""
    p = Path(relativa)
    return p if p.is_absolute() else RAIZ / p


def variable_entorno(nombre: str, obligatoria: bool = True) -> str | None:
    """Devuelve una variable de entorno; las credenciales nunca se escriben en el código."""
    cargar_config()
    valor = os.getenv(nombre)
    if obligatoria and not valor:
        raise ErrorConfiguracion(f"Falta la variable de entorno {nombre} (ver .env.example)")
    return valor


def configurar_logging(nivel: str | None = None) -> None:
    """Logging con niveles para todo el paquete (INFO por defecto, LOG_LEVEL para cambiarlo)."""
    logging.basicConfig(
        level=(nivel or os.getenv("LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for ruidoso in ("httpx", "httpx2", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
