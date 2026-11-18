"""Limpieza de texto de entrada: se aplica a las preguntas y a los documentos antes de indexarlos.

Los caracteres invisibles (ancho cero, marcas de dirección, etiquetas Unicode) son una vía
conocida para esconder instrucciones dentro de un texto (OWASP LLM01, inyección indirecta).
"""

import re
import unicodedata

# Rango de "tag characters" de Unicode, usado para esconder texto ASCII invisible.
_ETIQUETAS_UNICODE = re.compile("[\U000E0000-\U000E007F]")
_ESPACIOS = re.compile(r"\s+")


def limpiar(texto: str) -> tuple[str, int]:
    """Normaliza (NFKC), quita caracteres de formato invisibles y colapsa espacios.

    Devuelve el texto limpio y la cantidad de caracteres invisibles eliminados, para auditoría.
    """
    texto = unicodedata.normalize("NFKC", texto)
    texto, n_etiquetas = _ETIQUETAS_UNICODE.subn("", texto)
    limpio = []
    n_invisibles = 0
    for c in texto:
        # Cf = formato (ancho cero, marcas bidireccionales); Co/Cn = uso privado / sin asignar.
        if unicodedata.category(c) in ("Cf", "Co", "Cn") or (unicodedata.category(c) == "Cc" and c not in "\n\t"):
            n_invisibles += 1
            continue
        limpio.append(c)
    return _ESPACIOS.sub(" ", "".join(limpio)).strip(), n_invisibles + n_etiquetas
