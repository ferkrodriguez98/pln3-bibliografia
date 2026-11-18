"""Clientes de las APIs públicas de PubMed (E-utilities de NCBI) y de Europe PMC.

Solo se usan servicios que ofrecen acceso programático: los resúmenes salen de PubMed y
los textos completos, del subconjunto de acceso abierto de Europe PMC.
"""

import logging
import re
import time

import requests
from lxml import etree

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
HERRAMIENTA = "asistente-bibliografico-pln3"


class ErrorFuente(RuntimeError):
    """Una API externa respondió con error o con un formato inesperado."""


class ClientePubMed:
    """Búsqueda y descarga de metadatos respetando el límite de NCBI (3 consultas por segundo)."""

    def __init__(self, email: str | None = None, pausa_s: float = 0.4, timeout_s: float = 30):
        self.email = email
        self.pausa_s = pausa_s
        self.timeout_s = timeout_s
        self.sesion = requests.Session()

    def _get(self, url: str, params: dict) -> requests.Response:
        params = {**params, "tool": HERRAMIENTA}
        if self.email:
            params["email"] = self.email
        time.sleep(self.pausa_s)
        try:
            r = self.sesion.get(url, params=params, timeout=self.timeout_s)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ErrorFuente(f"Falló la consulta a {url}: {exc}") from exc
        return r

    def buscar(self, consulta: str, max_resultados: int = 200) -> list[str]:
        """Devuelve los PMID que cumplen la ecuación de búsqueda."""
        r = self._get(f"{EUTILS}/esearch.fcgi", {
            "db": "pubmed", "term": consulta, "retmax": max_resultados, "retmode": "json",
        })
        try:
            return r.json()["esearchresult"]["idlist"]
        except (KeyError, ValueError) as exc:
            raise ErrorFuente("Respuesta inesperada de esearch") from exc

    def metadatos(self, pmids: list[str]) -> list[dict]:
        """Título, resumen, revista, año, autores e identificadores de cada artículo."""
        if not pmids:
            return []
        r = self._get(f"{EUTILS}/efetch.fcgi", {
            "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
        })
        raiz = etree.fromstring(r.content)
        return [_parsear_articulo(a) for a in raiz.findall(".//PubmedArticle")]

    def texto_completo(self, pmcid: str) -> bytes | None:
        """XML JATS del texto completo si el artículo está en el subconjunto abierto de Europe PMC."""
        time.sleep(self.pausa_s)
        try:
            r = self.sesion.get(f"{EUROPEPMC}/{pmcid}/fullTextXML", timeout=self.timeout_s)
        except requests.RequestException as exc:
            log.warning("Europe PMC no respondió para %s: %s", pmcid, exc)
            return None
        if r.status_code != 200 or b"<body" not in r.content:
            return None
        return r.content


def _texto(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _parsear_articulo(art) -> dict:
    cita = art.find("MedlineCitation")
    articulo = cita.find("Article")
    partes = []
    for ab in articulo.findall("Abstract/AbstractText"):
        etiqueta = ab.get("Label")
        partes.append(f"{etiqueta}: {_texto(ab)}" if etiqueta else _texto(ab))
    fecha = articulo.find("Journal/JournalIssue/PubDate")
    anio = _texto(fecha.find("Year")) if fecha is not None else ""
    if not anio and fecha is not None:
        m = re.search(r"\d{4}", _texto(fecha.find("MedlineDate")))
        anio = m.group(0) if m else ""
    ids = {i.get("IdType"): _texto(i) for i in art.findall("PubmedData/ArticleIdList/ArticleId")}
    autores = [_texto(a.find("LastName")) for a in articulo.findall("AuthorList/Author")
               if a.find("LastName") is not None]
    return {
        "pmid": _texto(cita.find("PMID")),
        "titulo": _texto(articulo.find("ArticleTitle")),
        "resumen": " ".join(partes),
        "revista": _texto(articulo.find("Journal/ISOAbbreviation")) or _texto(articulo.find("Journal/Title")),
        "anio": int(anio) if anio.isdigit() else None,
        "autores": autores,
        "doi": ids.get("doi"),
        "pmcid": ids.get("pmc"),
    }
