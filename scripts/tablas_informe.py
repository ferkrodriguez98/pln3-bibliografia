"""Genera las tablas LaTeX del informe a partir de config.json y de eval/resultados/*.json.

Uso: python scripts/tablas_informe.py
"""

import json

from asistente.config import RAIZ, cargar_config

DESTINO = RAIZ / "docs" / "informe" / "tablas"
NOMBRES_MODO = {"bm25": "BM25", "denso": "Denso (MiniLM)", "hibrido": "Híbrido (RRF)",
                "hibrido_rerank": "Híbrido + reranker"}
NOMBRES_CONSULTA = {"original": "original", "planificador": "planificador",
                    "original+planificador": "ambas",
                    "original+planificador, rerank en inglés": "ambas, reranker en inglés"}


def pct(x, dec: int = 0) -> str:
    return "--" if x is None else f"{100 * x:.{dec}f}".replace(".", ",")


def num(x, dec=2) -> str:
    return "--" if x is None else f"{x:.{dec}f}".replace(".", ",")


def tabla(nombre: str, epigrafe: str, columnas: str, encabezado: list[str], filas: list[list[str]],
          etiqueta: str, nota: str = "", chica: bool = False) -> None:
    tamanio = [r"\footnotesize", r"\setlength{\tabcolsep}{3pt}"] if chica else [r"\small"]
    lineas = [r"\begin{table}[htbp]", r"\centering", *tamanio, rf"\caption{{{epigrafe}}}",
              rf"\label{{{etiqueta}}}", rf"\begin{{tabular}}{{{columnas}}}", r"\toprule",
              " & ".join(rf"\textbf{{{h}}}" for h in encabezado) + r" \\", r"\midrule"]
    lineas += [" & ".join(f) + r" \\" for f in filas]
    lineas += [r"\bottomrule", r"\end{tabular}"]
    if nota:
        lineas.append(rf"\par\smallskip\parbox{{0.92\linewidth}}{{\footnotesize {nota}}}")
    lineas.append(r"\end{table}")
    (DESTINO / f"{nombre}.tex").write_text("\n".join(lineas) + "\n")


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    cfg = cargar_config()
    a, precios = cfg["agentes"], cfg["precios_usd_por_millon"]
    filas = []
    for rol, modelo, esfuerzo in [("Planificador", a["modelo_auxiliar"], a.get("esfuerzo_planificador", a["esfuerzo_auxiliar"])),
                                  ("Analista", a["modelo_analista"], a["esfuerzo_analista"]),
                                  ("Verificador", a["modelo_auxiliar"], a.get("esfuerzo_verificador", a["esfuerzo_auxiliar"])),
                                  ("Guardia (confirmación)", a["modelo_auxiliar"], a.get("esfuerzo_guardia", a["esfuerzo_auxiliar"]))]:
        p = precios[modelo]
        filas.append([rol, rf"\texttt{{{modelo}}}", {"low": "bajo", "medium": "medio", "high": "alto"}[esfuerzo],
                      f"{num(p['entrada'])} / {num(p['salida'])}"])
    filas.append(["Embeddings", r"\texttt{all-MiniLM-L6-v2}", "--", "local"])
    filas.append(["Reranker", r"\texttt{ms-marco-MiniLM-L-6-v2}", "--", "local"])
    filas.append(["Detector de inyección", r"\texttt{deberta-v3-base-prompt-injection-v2}", "--", "local"])
    tabla("modelos", "Modelos por rol.", "llll", ["Rol", "Modelo", "Esfuerzo", "USD por millón"],
          filas, "tab:modelos", "USD por millón de tokens de entrada / de salida, según la documentación de "
          "Anthropic consultada el 6 de noviembre de 2026.")

    rec = RAIZ / "eval" / "resultados" / "recuperacion.json"
    if rec.exists():
        filas = []
        for r in json.loads(rec.read_text()):
            pool = r.get("fragmento_pool") or {}
            filas.append([NOMBRES_MODO[r["modo"]], NOMBRES_CONSULTA[r["consultas"]],
                          pct(r["articulo"]["Hit@1"]), pct(r["articulo"]["Hit@3"]),
                          pct(r["fragmento"]["Hit@5"]), pct(pool.get("P@5")), num(pool.get("nDCG@5")),
                          num(pool.get("MRR")), f"{r['latencia_p50_ms']:.0f}"])
        tabla("recuperacion", "Recuperación sobre las 40 preguntas respondibles.", "llrrrrrrr",
              ["Modo", "Consultas", "Art.@1", "Art.@3", "Origen@5", "P@5", "nDCG@5", "MRR", "ms"], filas,
              "tab:recuperacion",
              "Consultas: la pregunta original, las del planificador o ambas. "
              "Art.@k: porcentaje de preguntas con el artículo de origen entre los k primeros. Origen@5: el "
              "fragmento de origen entre los cinco primeros. P@5, nDCG@5 y MRR usan los juicios de relevancia "
              "por pooling. ms: mediana de la latencia de recuperación.", chica=True)

        filas = []
        for r in json.loads(rec.read_text()):
            if r["modo"] not in ("bm25", "hibrido") or r["consultas"] not in ("original", "planificador",
                                                                              "original+planificador"):
                continue
            for idioma in ("es", "en"):
                f = r.get(f"fragmento_{idioma}")
                if f:
                    filas.append([NOMBRES_MODO[r["modo"]], NOMBRES_CONSULTA[r["consultas"]],
                                  {"es": "español", "en": "inglés"}[idioma],
                                  pct(r[f"articulo_{idioma}"]["Hit@3"]), pct(f["Hit@5"])])
        tabla("recuperacion_idioma", "Recuperación según el idioma de la pregunta (30 en español, 10 en inglés).",
              "lllrr", ["Modo", "Consultas", "Idioma", "Art.@3", "Origen@5"], filas, "tab:idioma")

    ctx = RAIZ / "eval" / "resultados" / "contexto.json"
    if ctx.exists():
        filas = [[str(c["top"]), str(c["max_por_articulo"]), NOMBRES_MODO[c["modo"]], pct(c["con_relevante"]),
                  num(c["relevantes_medios"]), pct(c["con_origen"])]
                 for c in json.loads(ctx.read_text()) if (c["top"], c["max_por_articulo"]) in ((6, 2), (6, 3), (10, 3))]
        tabla("contexto", "Cobertura del contexto que recibe el analista.", "rrlrrr",
              ["Fragmentos", "Por artículo", "Modo", "Con relevante", "Relevantes", "Con origen"], filas,
              "tab:contexto", "Con relevante: preguntas con al menos un fragmento relevante (pooling) en el contexto. "
              "Relevantes: cantidad media de fragmentos relevantes. Con origen: el fragmento de origen está en el contexto.")

    filas = []
    for archivo, nombre in (("sistema.json", "Completo"), ("sistema_sin_verificador.json", "Sin verificador")):
        p = RAIZ / "eval" / "resultados" / archivo
        if not p.exists():
            continue
        s = json.loads(p.read_text())["resumen"]
        r, c = s["respondibles"], s["costo_latencia"]
        filas.append([nombre, pct(r["recupero_articulo"], 1), pct(r["cito_articulo_correcto"], 1),
                      num(r["correccion_media_1a5"]), pct(r["fidelidad_afirmaciones"], 1),
                      pct(s["fuera_de_alcance_rechazadas"]), pct(s["sin_respuesta_reconocidas"]),
                      num(c["costo_medio_usd"], 3), f"{c['latencia_p50_ms'] / 1000:.0f}",
                      f"{c['latencia_p95_ms'] / 1000:.0f}"])
    if filas:
        tabla("sistema", "Sistema completo y variante sin verificador.", "lrrrrrrrrr",
              ["Variante", "Recup.", "Cita", "Corr.", "Fidel.", "Fuera", "Sin resp.", "USD", "p50 s", "p95 s"],
              filas, "tab:sistema",
              "Recup.: artículo de origen entre los recuperados. Cita: la respuesta cita el artículo de origen. "
              "Corr.: corrección de 1 a 5 según el juez. Fidel.: afirmaciones respaldadas por los documentos "
              "citados. Fuera: preguntas fuera de alcance rechazadas. Sin resp.: preguntas sin respuesta en el "
              "corpus en las que el sistema lo reconoce. USD: costo medio por consulta.")

    seg = RAIZ / "eval" / "resultados" / "seguridad.json"
    if seg.exists():
        s = json.loads(seg.read_text())
        d, i = s["inyeccion_directa"], s["inyeccion_indirecta"]
        filas = [
            ["Ataques directos bloqueados por la guardia", f"{d['bloqueados_por_guardia']} de {d['n']}"],
            ["Ataques directos rechazados por el planificador", f"{d['rechazados_por_planificador']} de {d['n']}"],
            ["Ataques con filtración del prompt", f"{d['con_fuga_de_prompt']} de {d['n']}"],
            ["Preguntas legítimas bloqueadas", f"{s['falsos_positivos_preguntas']['bloqueadas']} de {s['falsos_positivos_preguntas']['n']}"],
            ["Fragmentos del corpus excluidos", f"{s['fragmentos_del_corpus_marcados']['marcados']} de {s['fragmentos_del_corpus_marcados']['n']}"],
            ["Documento plantado detectado al indexar", "sí" if i["detector_lo_marca"] else "no"],
            ["Analista adopta la afirmación plantada (filtro salteado)", "sí" if i["adopta_la_afirmacion_plantada"] else "no"],
            ["Analista recomienda el enlace plantado (filtro salteado)", "sí" if i["recomienda_el_enlace"] else "no"],
            ["Analista advierte la inyección (filtro salteado)", "sí" if i["advierte_la_inyeccion"] else "no"],
        ]
        tabla("seguridad", "Pruebas adversariales.", "lr", ["Prueba", "Resultado"], filas, "tab:seguridad")
    print("Tablas ->", DESTINO)


if __name__ == "__main__":
    main()
