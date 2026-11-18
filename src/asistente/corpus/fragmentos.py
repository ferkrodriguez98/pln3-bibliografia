"""División de los artículos en fragmentos con metadatos para la trazabilidad de las fuentes."""

from dataclasses import asdict, dataclass

from asistente.seguridad.sanitizar import limpiar


@dataclass
class Fragmento:
    id: str            # "<pmid>:<n>", estable entre corridas
    pmid: str
    titulo: str
    anio: int | None
    revista: str
    seccion: str
    texto: str
    fuente: str        # "texto_completo" o "resumen"
    sospechoso: bool = False   # marcado por el detector de inyección al indexar

    def a_dict(self) -> dict:
        return asdict(self)


def ventanas(palabras: list[str], max_palabras: int, solape: int) -> list[list[str]]:
    """Ventanas deslizantes de max_palabras con solape; la última no queda más corta que el solape."""
    if len(palabras) <= max_palabras:
        return [palabras]
    paso = max_palabras - solape
    salida = []
    for inicio in range(0, len(palabras), paso):
        ventana = palabras[inicio:inicio + max_palabras]
        salida.append(ventana)
        if inicio + max_palabras >= len(palabras):
            break
    return salida


def fragmentar(articulo: dict, secciones: list[tuple[str, str]], fuente: str, max_palabras: int,
               solape: int, min_palabras: int) -> list[Fragmento]:
    """Fragmenta sección por sección (nunca mezcla secciones) y limpia cada texto."""
    frags = []
    for nombre, texto in secciones:
        texto, _ = limpiar(texto)
        for ventana in ventanas(texto.split(), max_palabras, solape):
            if len(ventana) < min_palabras and frags:
                continue
            frags.append(Fragmento(
                id=f"{articulo['pmid']}:{len(frags):03d}",
                pmid=articulo["pmid"],
                titulo=articulo["titulo"],
                anio=articulo.get("anio"),
                revista=articulo.get("revista", ""),
                seccion=nombre,
                texto=" ".join(ventana),
                fuente=fuente,
            ))
    return frags
