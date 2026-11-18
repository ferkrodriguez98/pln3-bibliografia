"""Consulta por línea de comandos, sin levantar la API.

Uso: python scripts/consultar.py "¿Qué tan confiable es el PLI entre sesiones?" [--sin_cache] [--traza]
"""

import argparse
import json

from asistente.config import configurar_logging
from asistente.servicio import Servicio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pregunta")
    ap.add_argument("--sin_cache", action="store_true")
    ap.add_argument("--json", action="store_true", help="imprimir la salida completa en JSON")
    args = ap.parse_args()
    configurar_logging("WARNING")
    s = Servicio().consultar(args.pregunta, usar_cache=not args.sin_cache)
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return
    print(f"[{s['estado']}] veredicto={s.get('veredicto')} consultas={s.get('consultas')}\n")
    print(s["respuesta"], "\n")
    for r in s["referencias"]:
        print(f"[{r['n']}] {r['titulo']} ({r['revista']}, {r['anio']}). {r['seccion']}. {r['url']}")
    for a in s["advertencias"]:
        print("ADVERTENCIA:", a)
    print("\n", json.dumps(s["metricas"], ensure_ascii=False))


if __name__ == "__main__":
    main()
