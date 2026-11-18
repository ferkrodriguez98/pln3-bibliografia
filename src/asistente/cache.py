"""Caché de respuestas en disco: una pregunta repetida no vuelve a llamar a los modelos.

Es una de las acciones de optimización de costo y latencia del sistema.
"""

import hashlib
import json
import threading
import time
from pathlib import Path


def clave(pregunta: str) -> str:
    normalizada = " ".join(pregunta.lower().split()).strip(" ?¿!¡.")
    return hashlib.sha256(normalizada.encode()).hexdigest()[:32]


class CacheRespuestas:
    def __init__(self, archivo: Path, ttl_horas: float, habilitada: bool = True):
        self.archivo = archivo
        self.ttl_s = ttl_horas * 3600
        self.habilitada = habilitada
        self._lock = threading.Lock()
        self._datos = json.loads(archivo.read_text()) if habilitada and archivo.exists() else {}

    def obtener(self, pregunta: str) -> dict | None:
        if not self.habilitada:
            return None
        item = self._datos.get(clave(pregunta))
        if item and time.time() - item["guardado"] < self.ttl_s:
            return item["respuesta"]
        return None

    def guardar(self, pregunta: str, respuesta: dict) -> None:
        if not self.habilitada:
            return
        with self._lock:
            self._datos[clave(pregunta)] = {"guardado": time.time(), "respuesta": respuesta}
            self.archivo.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.archivo.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._datos, ensure_ascii=False))
            tmp.replace(self.archivo)
