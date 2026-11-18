"""Extracción de secciones de texto desde el XML JATS de Europe PMC."""

import re

from lxml import etree

# Elementos que no aportan texto útil para el RAG o que meten ruido (fórmulas en LaTeX/MathML,
# tablas completas, referencias cruzadas a figuras, material suplementario).
_DESCARTAR = ("tex-math", "{http://www.w3.org/1998/Math/MathML}math", "inline-formula",
              "disp-formula", "table", "supplementary-material", "fig")
_SECCIONES_EXCLUIDAS = re.compile(
    r"^(acknowledg|funding|author contribution|conflict|competing interest|data availability|"
    r"ethics|supplementary|abbreviation|declaration)", re.I)


def _texto(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def secciones(xml: bytes) -> list[tuple[str, str]]:
    """Devuelve [(nombre_de_sección, texto)] con el resumen primero y el cuerpo después."""
    raiz = etree.fromstring(xml)
    for tag in _DESCARTAR:
        for el in list(raiz.iter(tag)):
            _quitar(el)
    salida = []
    resumen = raiz.find(".//front//abstract")
    if resumen is not None:
        salida.append(("Abstract", _texto(resumen)))
    for sec in raiz.findall(".//body//sec"):
        # Solo los párrafos directos: las subsecciones se procesan en su propia iteración.
        parrafos = [_texto(p) for p in sec.findall("p")]
        parrafos = [p for p in parrafos if p]
        if not parrafos:
            continue
        nombre = _nombre_seccion(sec)
        if _SECCIONES_EXCLUIDAS.match(nombre):
            continue
        salida.append((nombre, " ".join(parrafos)))
    return salida


def _quitar(el) -> None:
    """Elimina un elemento sin perder el texto que lo sigue (en lxml, la 'cola' viaja con él)."""
    madre = el.getparent()
    if madre is None:
        return
    cola = el.tail or ""
    previo = el.getprevious()
    if previo is not None:
        previo.tail = (previo.tail or "") + " " + cola
    else:
        madre.text = (madre.text or "") + " " + cola
    madre.remove(el)


def _nombre_seccion(sec) -> str:
    """Título de la sección precedido por el de la sección madre (p. ej. 'Methods > EEG recording')."""
    nombres = []
    actual = sec
    while actual is not None and actual.tag == "sec":
        titulo = _texto(actual.find("title"))
        if titulo:
            nombres.append(titulo)
        actual = actual.getparent()
    return " > ".join(reversed(nombres))[:120] or "Body"
