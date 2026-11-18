"""Control de acceso de la API: claves por usuario y límite de consultas por minuto.

El límite de tasa mitiga el consumo no acotado de recursos (OWASP LLM10): cada consulta
cuesta llamadas a modelos pagos.
"""

import hmac
import threading
import time
from collections import defaultdict, deque


class ControlAcceso:
    def __init__(self, claves_validas: list[str], consultas_por_minuto: int):
        if not claves_validas:
            raise ValueError("No hay claves de acceso configuradas (APP_API_KEYS)")
        self.claves = claves_validas
        self.limite = consultas_por_minuto
        self._historial: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def autenticar(self, clave: str | None) -> bool:
        # Comparación en tiempo constante para no filtrar información por la latencia.
        return bool(clave) and any(hmac.compare_digest(clave, c) for c in self.claves)

    def permitir(self, clave: str, ahora: float | None = None) -> bool:
        """Ventana deslizante de 60 s por clave."""
        ahora = time.monotonic() if ahora is None else ahora
        with self._lock:
            h = self._historial[clave]
            while h and ahora - h[0] > 60:
                h.popleft()
            if len(h) >= self.limite:
                return False
            h.append(ahora)
            return True
