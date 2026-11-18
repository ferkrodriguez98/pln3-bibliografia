"""Esquemas de las salidas estructuradas de los agentes (validados por la API y por Pydantic)."""

from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Salida del planificador."""
    en_alcance: bool = Field(description="True si la pregunta se puede responder con literatura científica "
                                         "sobre EEG, conectividad funcional, confiabilidad o bilingüismo")
    motivo: str = Field(description="Una oración que justifica la decisión")
    idioma: str = Field(description="Código ISO 639-1 del idioma de la pregunta, p. ej. 'es' o 'en'")
    consultas: list[str] = Field(description="Entre 1 y 3 consultas de búsqueda en inglés técnico; "
                                             "vacía si la pregunta está fuera de alcance")


class Afirmacion(BaseModel):
    texto: str = Field(description="La afirmación tal como aparece en la respuesta, resumida si es larga")
    citas: list[int] = Field(description="Números de documento citados para esta afirmación")
    respaldada: bool = Field(description="True si los documentos citados dicen lo que afirma")
    comentario: str = Field(description="Qué falta o qué está mal, si no está respaldada; vacío si lo está")


class Verificacion(BaseModel):
    """Salida del verificador."""
    afirmaciones: list[Afirmacion]
    veredicto: Literal["aprobada", "corregir", "buscar_mas"] = Field(
        description="aprobada: todo respaldado; corregir: hay afirmaciones sin respaldo que el analista "
                    "puede arreglar con estos documentos; buscar_mas: los documentos no alcanzan para "
                    "responder y hace falta otra búsqueda")
    comentario_general: str = Field(description="Indicaciones concretas para el analista")
    consulta_adicional: str = Field(description="Si el veredicto es buscar_mas, una consulta de búsqueda "
                                                "en inglés; si no, vacío")
