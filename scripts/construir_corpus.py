"""Arma el corpus: corre las ecuaciones de búsqueda en PubMed, baja metadatos y resúmenes,
y el texto completo de los artículos que están en el subconjunto abierto de Europe PMC.

Uso: python scripts/construir_corpus.py
"""

import json
import logging

from asistente.config import cargar_config, configurar_logging, ruta, variable_entorno
from asistente.corpus.pubmed import ClientePubMed

log = logging.getLogger("construir_corpus")


def main() -> None:
    configurar_logging()
    cfg = cargar_config()["corpus"]
    cliente = ClientePubMed(email=variable_entorno("NCBI_EMAIL", obligatoria=False),
                            pausa_s=cfg["pausa_ncbi_s"])

    consultas_por_pmid: dict[str, list[str]] = {}
    for nombre, consulta in cfg["consultas_pubmed"].items():
        pmids = cliente.buscar(consulta, cfg["max_resultados"])
        log.info("%s: %d resultados", nombre, len(pmids))
        for p in pmids:
            consultas_por_pmid.setdefault(p, []).append(nombre)

    articulos = cliente.metadatos(sorted(consultas_por_pmid))
    dir_crudos = ruta(cfg["dir_crudos"])
    dir_crudos.mkdir(parents=True, exist_ok=True)
    con_texto = 0
    for a in articulos:
        a["consultas"] = consultas_por_pmid.get(a["pmid"], [])
        a["texto_completo"] = False
        if a["pmcid"]:
            xml = cliente.texto_completo(a["pmcid"])
            if xml:
                (dir_crudos / f"{a['pmid']}.xml").write_bytes(xml)
                a["texto_completo"] = True
                con_texto += 1

    destino = ruta(cfg["archivo_corpus"])
    with open(destino, "w", encoding="utf-8") as f:
        for a in articulos:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    log.info("Corpus: %d artículos (%d con texto completo, %d solo con resumen) -> %s",
             len(articulos), con_texto, len(articulos) - con_texto, destino)


if __name__ == "__main__":
    main()
