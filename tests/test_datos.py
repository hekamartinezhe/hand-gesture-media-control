"""Tests unitarios para la carga, limpieza y partición estratificada del dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.utils import (
    CLASES_GESTOS,
    COLUMNAS_CSV,
    COLUMNAS_FEATURES,
    cargar_dataset,
    construir_matriz_x,
    dividir_estratificado,
    limpiar_dataset,
)

RUTA_DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"


def _fila(
    label: str = "palma",
    handedness: str = "Right",
    semilla: int = 7,
) -> dict[str, object]:
    """Genera una fila sintética con el esquema completo y coordenadas normalizadas."""
    rng = np.random.default_rng(semilla)
    puntos = rng.uniform(low=0.2, high=0.8, size=(21, 3)).astype(np.float32)
    puntos[0] = 0.0
    puntos[9] = [0.0, -1.0, 0.0]  # nudillo medio a distancia unitaria de la muñeca

    vector = np.concatenate([puntos[:, 0], puntos[:, 1], puntos[:, 2]])
    fila = {col: float(valor) for col, valor in zip(COLUMNAS_FEATURES, vector)}
    fila.update(
        label=label,
        handedness=handedness,
        session_id="sesion_test",
        timestamp=0.0,
    )
    return fila


def _dataframe(filas: list[dict[str, object]]) -> pd.DataFrame:
    """Construye un DataFrame respetando el orden estricto de columnas del esquema."""
    return pd.DataFrame(filas, columns=COLUMNAS_CSV)


def test_carga_dataset_real_con_esquema_completo() -> None:
    """El CSV real debe tener 2400 filas, 67 columnas y 400 muestras por clase."""
    df = cargar_dataset(RUTA_DATASET)

    assert df.shape == (2400, 67)
    assert list(df.columns) == COLUMNAS_CSV
    assert df["label"].value_counts().to_dict() == {clase: 400 for clase in CLASES_GESTOS}


def test_carga_dataset_inexistente_lanza_error(tmp_path: Path) -> None:
    """Una ruta inexistente debe fallar con FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No existe el dataset"):
        cargar_dataset(tmp_path / "no_existe.csv")


def test_esquema_incompleto_lanza_error() -> None:
    """Un DataFrame sin todas las columnas del esquema debe rechazarse."""
    df_incompleto = pd.DataFrame({"label": ["palma"], "handedness": ["Right"]})

    with pytest.raises(ValueError, match="Faltan columnas"):
        limpiar_dataset(df_incompleto)
    with pytest.raises(ValueError, match="Faltan columnas"):
        construir_matriz_x(df_incompleto)


def test_limpieza_elimina_registros_invalidos_sin_mutar_el_original() -> None:
    """Descarta nulos, infinitos, vectores degenerados, etiquetas y duplicados."""
    valida_1 = _fila(label="palma", handedness="Right", semilla=1)
    valida_2 = _fila(label="paz", handedness="Left", semilla=2)
    valida_3 = _fila(label="neutro", handedness="Left", semilla=3)

    duplicada = valida_1.copy()

    etiqueta_invalida = _fila(label="otro", semilla=4)
    mano_invalida = _fila(handedness="Both", semilla=5)
    nula = _fila(semilla=6)
    nula["x3"] = np.nan
    infinita = _fila(semilla=8)
    infinita["y5"] = np.inf
    degenerada = {col: 0.0 for col in COLUMNAS_FEATURES}
    degenerada.update(
        label="palma", handedness="Right", session_id="sesion_test", timestamp=0.0
    )
    fuera_de_rango = _fila(semilla=9)
    fuera_de_rango["x1"] = 11.0

    df = _dataframe(
        [
            valida_1,
            valida_2,
            etiqueta_invalida,
            nula,
            valida_3,
            infinita,
            degenerada,
            fuera_de_rango,
            mano_invalida,
            duplicada,
        ]
    )

    limpio = limpiar_dataset(df)

    assert len(df) == 10  # la función no altera el DataFrame de entrada
    assert len(limpio) == 3
    assert set(limpio["label"]) == {"palma", "paz", "neutro"}
    assert limpio[COLUMNAS_FEATURES].duplicated().sum() == 0


def test_limpieza_del_dataset_real_no_pierde_muestras() -> None:
    """El dataset capturado debe pasar todos los filtros sin perder filas."""
    df = cargar_dataset(RUTA_DATASET)
    limpio = limpiar_dataset(df)

    assert len(limpio) == len(df) == 2400
    assert limpio["label"].value_counts().to_dict() == {clase: 400 for clase in CLASES_GESTOS}


def test_construir_matriz_x_espeja_manos_izquierdas() -> None:
    """La mano izquierda debe quedar reflejada en x respecto a la derecha."""
    derecha = _fila(handedness="Right", semilla=11)
    izquierda = derecha.copy()
    izquierda["handedness"] = "Left"

    X = construir_matriz_x(_dataframe([derecha, izquierda]))

    assert X.shape == (2, 63)
    assert X.dtype == np.float32
    assert np.allclose(X[0, :21], -X[1, :21], atol=1e-6)
    assert np.allclose(X[0, 21:], X[1, 21:], atol=1e-6)


def test_construir_matriz_x_vacia_devuelve_forma_correcta() -> None:
    """Un dataset vacío debe producir una matriz (0, 63) sin lanzar excepciones."""
    X = construir_matriz_x(pd.DataFrame(columns=COLUMNAS_CSV))

    assert X.shape == (0, 63)


def test_particion_70_15_15_estratificada() -> None:
    """Las particiones deben respetar tamaños y proporciones por clase."""
    df = limpiar_dataset(cargar_dataset(RUTA_DATASET))
    split = dividir_estratificado(df)

    assert len(split["X_train"]) == 1680
    assert len(split["X_val"]) == 360
    assert len(split["X_test"]) == 360
    assert split["X_train"].shape == (1680, 63)

    for nombre in ("train", "val", "test"):
        y = split[f"y_{nombre}"]
        conteo = pd.Series(y).value_counts()
        assert set(conteo.index) == set(CLASES_GESTOS)
        assert (conteo == {"train": 280, "val": 60, "test": 60}[nombre]).all()


def test_particion_sin_fugas_y_reproducible() -> None:
    """No debe haber filas compartidas entre particiones y la semilla debe ser estable."""
    df = limpiar_dataset(cargar_dataset(RUTA_DATASET))
    split_a = dividir_estratificado(df)
    split_b = dividir_estratificado(df)

    for llave in split_a:
        assert np.array_equal(split_a[llave], split_b[llave])

    conjuntos = {
        nombre: {fila.tobytes() for fila in split_a[f"X_{nombre}"]}
        for nombre in ("train", "val", "test")
    }
    assert conjuntos["train"].isdisjoint(conjuntos["val"])
    assert conjuntos["train"].isdisjoint(conjuntos["test"])
    assert conjuntos["val"].isdisjoint(conjuntos["test"])


def test_balance_de_clases_en_cada_particion() -> None:
    """Ninguna clase debe salir del rango 14%-19% esperado."""
    df = limpiar_dataset(cargar_dataset(RUTA_DATASET))
    split = dividir_estratificado(df)

    for nombre in ("train", "val", "test"):
        proporciones = pd.Series(split[f"y_{nombre}"]).value_counts(normalize=True)
        assert proporciones.min() >= 0.14
        assert proporciones.max() <= 0.19
