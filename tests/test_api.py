"""Tests de la API con un servicio falso (sin modelos ni llamadas a Claude)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from asistente.servicio import ErrorEntrada


class ServicioFalso:
    def __init__(self):
        self.indice = SimpleNamespace(fragmentos=[SimpleNamespace(pmid="1"), SimpleNamespace(pmid="2")])
        self.cfg = {"agentes": {"modelo_analista": "a", "modelo_auxiliar": "b"}}

    def consultar(self, pregunta, usuario="local"):
        if "larga" in pregunta:
            raise ErrorEntrada("demasiado larga")
        return {"estado": "respondida", "respuesta": f"eco: {pregunta}", "usuario": usuario}


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setenv("APP_API_KEYS", "clave-test")
    app.state.servicio = ServicioFalso()
    with TestClient(app) as c:
        yield c
    del app.state.servicio


def test_salud_no_pide_clave(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200 and r.json()["articulos"] == 2


def test_consulta_sin_clave_es_rechazada(cliente):
    assert cliente.post("/consultas", json={"pregunta": "¿Qué es el PLI?"}).status_code == 401
    assert cliente.post("/consultas", json={"pregunta": "¿Qué es el PLI?"},
                        headers={"X-API-Key": "otra"}).status_code == 401


def test_consulta_valida_no_expone_la_clave(cliente):
    r = cliente.post("/consultas", json={"pregunta": "¿Qué es el PLI?"}, headers={"X-API-Key": "clave-test"})
    assert r.status_code == 200 and r.json()["respuesta"] == "eco: ¿Qué es el PLI?"
    assert r.json()["usuario"] != "clave-test"


def test_validacion_de_entrada(cliente):
    h = {"X-API-Key": "clave-test"}
    assert cliente.post("/consultas", json={"pregunta": "x" * 1001}, headers=h).status_code == 422
    assert cliente.post("/consultas", json={"pregunta": "pregunta larga"}, headers=h).status_code == 422


def test_limite_de_tasa(cliente):
    h = {"X-API-Key": "clave-test"}
    codigos = [cliente.post("/consultas", json={"pregunta": f"pregunta {i}"}, headers=h).status_code
               for i in range(25)]
    assert codigos.count(200) == 20 and codigos[-1] == 429
