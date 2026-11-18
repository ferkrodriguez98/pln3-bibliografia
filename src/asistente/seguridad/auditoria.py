"""Registro de auditoría: una línea JSON por consulta, con lo necesario para reconstruir qué pasó.

No guarda la clave de API del usuario (solo un hash) ni el texto completo de los documentos.
"""

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def hash_usuario(credencial: str) -> str:
    return hashlib.sha256(credencial.encode()).hexdigest()[:12]


class Auditoria:
    def __init__(self, archivo: Path):
        self.archivo = archivo
        self.archivo.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def registrar(self, evento: dict) -> None:
        linea = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **evento}
        with self._lock, open(self.archivo, "a", encoding="utf-8") as f:
            f.write(json.dumps(linea, ensure_ascii=False, default=str) + "\n")
