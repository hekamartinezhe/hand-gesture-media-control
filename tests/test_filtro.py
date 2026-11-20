"""Tests unitarios de la máquina de estados del filtro de disparo."""

import pytest

from scripts.filtro import FiltroDisparo, PestilloGesto


def _alimentar(
    filtro: FiltroDisparo,
    gesto: str | None,
    inicio: float = 0.0,
    cantidad: int = 5,
    intervalo: float = 0.02,
) -> list[str | None]:
    """Envía `cantidad` predicciones consecutivas y devuelve sus salidas."""
    return [
        filtro.actualizar(gesto, inicio + i * intervalo)
        for i in range(cantidad)
    ]


def test_no_dispara_antes_del_quinto_cuadro() -> None:
    """Cuatro detecciones consecutivas todavía no confirman el gesto."""
    filtro = FiltroDisparo()

    salidas = _alimentar(filtro, "palma", cantidad=4)

    assert salidas == [None, None, None, None]
    assert filtro.progreso == pytest.approx(4 / 5)
    assert filtro.candidato == "palma"


def test_dispara_exactamente_en_el_quinto_cuadro() -> None:
    """El disparo ocurre en la quinta llamada, ni antes ni después."""
    filtro = FiltroDisparo()

    salidas = _alimentar(filtro, "palma", cantidad=5)

    assert salidas[:4] == [None, None, None, None]
    assert salidas[4] == "palma"
    assert filtro.progreso == 0.0
    assert filtro.candidato is None


def test_cambio_de_gesto_reinicia_el_conteo() -> None:
    """Un gesto distinto descarta el progreso anterior y empieza desde uno."""
    filtro = FiltroDisparo()
    _alimentar(filtro, "palma", cantidad=4)
    assert filtro.progreso == pytest.approx(4 / 5)

    assert filtro.actualizar("paz", 0.10) is None
    assert filtro.progreso == pytest.approx(1 / 5)
    assert filtro.candidato == "paz"

    salidas = _alimentar(filtro, "paz", inicio=0.12, cantidad=4)
    assert salidas[:3] == [None, None, None]
    assert salidas[3] == "paz"


def test_neutro_nunca_dispara() -> None:
    """La clase neutro no debe emitir ningún comando aunque se repita."""
    filtro = FiltroDisparo()

    salidas = _alimentar(filtro, "neutro", cantidad=50)

    assert all(salida is None for salida in salidas)
    assert filtro.progreso == 0.0


def test_sin_mano_nunca_dispara() -> None:
    """La ausencia de detección tampoco debe emitir comandos."""
    filtro = FiltroDisparo()

    salidas = _alimentar(filtro, None, cantidad=20)

    assert all(salida is None for salida in salidas)
    assert filtro.progreso == 0.0


def test_gesto_desconocido_nunca_dispara() -> None:
    """Una etiqueta fuera del dominio se ignora y reinicia el candidato."""
    filtro = FiltroDisparo()

    salidas = _alimentar(filtro, "desconocido", cantidad=20)

    assert all(salida is None for salida in salidas)
    assert filtro.candidato is None


def test_neutro_reinicia_el_progreso() -> None:
    """Intercalar neutro debe borrar el avance de un candidato activo."""
    filtro = FiltroDisparo()
    _alimentar(filtro, "pulgar", cantidad=3, intervalo=0.02)
    assert filtro.progreso == pytest.approx(3 / 5)

    assert filtro.actualizar("neutro", 0.06) is None

    assert filtro.progreso == 0.0
    assert filtro.candidato is None


def test_cooldown_bloquea_disparos_durante_un_segundo() -> None:
    """Tras disparar, ninguna ráfaga nueva debe emitir hasta cumplir 1.0 s."""
    filtro = FiltroDisparo(cooldown_s=1.0)
    _alimentar(filtro, "palma", inicio=0.00, cantidad=5, intervalo=0.02)
    assert filtro.actualizar("palma", 0.08) is None  # el disparo fue en t=0.08

    # Ráfaga completa dentro del cooldown: se ignora por completo.
    salidas = _alimentar(filtro, "paz", inicio=0.10, cantidad=5, intervalo=0.02)
    assert all(salida is None for salida in salidas)
    assert filtro.progreso == 0.0

    # Justo al cumplirse 1.0 s desde el disparo, vuelve a aceptar detecciones.
    assert filtro.actualizar("paz", 1.08) is None
    salidas = _alimentar(filtro, "paz", inicio=1.10, cantidad=4, intervalo=0.02)
    assert salidas[:3] == [None, None, None]
    assert salidas[3] == "paz"


def test_cooldown_permite_disparar_al_cumplir_el_limite() -> None:
    """En t = ultimo_disparo + cooldown ya se permite el siguiente gesto."""
    filtro = FiltroDisparo(cooldown_s=1.0)
    salidas = _alimentar(filtro, "puno", inicio=0.00, cantidad=5, intervalo=0.01)
    assert salidas[4] == "puno"

    # 0.04 + 1.0 = 1.04: el primer cuadro del nuevo gesto cae en el límite.
    salidas = _alimentar(filtro, "senalar", inicio=1.04, cantidad=5, intervalo=0.01)
    assert salidas[:4] == [None, None, None, None]
    assert salidas[4] == "senalar"


def test_reiniciar_conserva_el_cooldown() -> None:
    """Reiniciar el candidato no debe saltarse el tiempo de enfriamiento."""
    filtro = FiltroDisparo(cooldown_s=1.0)
    salidas = _alimentar(filtro, "palma", inicio=0.0, cantidad=5, intervalo=0.02)
    assert salidas[4] == "palma"

    filtro.reiniciar()

    assert filtro.en_cooldown(0.50) is True
    assert filtro.en_cooldown(1.08) is False


def test_cooldown_restante_decrece_hasta_cero() -> None:
    """El HUD necesita saber cuánto falta para liberar el siguiente gesto."""
    filtro = FiltroDisparo(cooldown_s=1.0)
    assert filtro.cooldown_restante(0.0) == 0.0

    salidas = _alimentar(filtro, "palma", inicio=0.0, cantidad=5, intervalo=0.02)
    assert salidas[4] == "palma"

    assert filtro.cooldown_restante(0.08) == pytest.approx(1.0)
    assert filtro.cooldown_restante(0.58) == pytest.approx(0.5)
    assert filtro.cooldown_restante(1.08) == 0.0
    assert filtro.cooldown_restante(5.0) == 0.0


@pytest.mark.parametrize(
    ("frames", "cooldown"),
    [(0, 1.0), (-1, 1.0), (5, -0.1)],
)
def test_parametros_invalidos_lanzan_error(frames: int, cooldown: float) -> None:
    """La configuración del filtro debe validarse en el constructor."""
    with pytest.raises(ValueError):
        FiltroDisparo(frames_confirmacion=frames, cooldown_s=cooldown)


# --- Pestillo anti-repetición -------------------------------------------------


def test_pestillo_bloquea_el_mismo_gesto_sostenido() -> None:
    """Mantener el gesto no debe repetir la acción."""
    pestillo = PestilloGesto()
    assert pestillo.actualizar("palma") is True

    pestillo.bloquear("palma")

    assert all(pestillo.actualizar("palma") is False for _ in range(50))
    assert pestillo.bloqueado is True
    assert pestillo.gesto == "palma"


def test_pestillo_se_rearma_tras_tres_cuadros_neutros() -> None:
    """Bajar la mano durante tres cuadros habilita el siguiente gesto."""
    pestillo = PestilloGesto()
    pestillo.bloquear("pulgar")

    assert pestillo.actualizar("neutro") is False
    assert pestillo.actualizar("neutro") is False
    assert pestillo.actualizar("neutro") is False
    assert pestillo.bloqueado is False
    assert pestillo.actualizar("pulgar") is True


def test_pestillo_se_rearma_sin_mano() -> None:
    """La ausencia de detección también cuenta como bajar la mano."""
    pestillo = PestilloGesto()
    pestillo.bloquear("puno")

    assert pestillo.actualizar(None) is False
    assert pestillo.actualizar(None) is False
    assert pestillo.actualizar(None) is False
    assert pestillo.bloqueado is False


def test_pestillo_exige_cinco_cuadros_de_otro_gesto() -> None:
    """Cambiar de gesto rearma solo tras cinco cuadros sostenidos."""
    pestillo = PestilloGesto()
    pestillo.bloquear("palma")

    assert all(pestillo.actualizar("paz") is False for _ in range(4))
    assert pestillo.bloqueado is True

    assert pestillo.actualizar("paz") is True
    assert pestillo.bloqueado is False


def test_pestillo_ignora_un_flicker_de_un_cuadro() -> None:
    """Una predicción errónea aislada no debe rearmar el pestillo."""
    pestillo = PestilloGesto()
    pestillo.bloquear("palma")

    assert pestillo.actualizar("paz") is False
    assert pestillo.actualizar("palma") is False
    assert all(pestillo.actualizar("paz") is False for _ in range(4))
    assert pestillo.bloqueado is True

    assert pestillo.actualizar("paz") is True


@pytest.mark.parametrize(("neutro", "cambio"), [(0, 5), (3, 0), (-1, 5)])
def test_pestillo_parametros_invalidos(neutro: int, cambio: int) -> None:
    """Los umbrales del pestillo deben ser positivos."""
    with pytest.raises(ValueError):
        PestilloGesto(cuadros_neutro=neutro, cuadros_cambio=cambio)


def test_demo_combinado_dispara_una_vez_por_presentacion() -> None:
    """La demo con frames_confirmacion=1 no repite mientras se mantiene el gesto."""
    filtro = FiltroDisparo(frames_confirmacion=1, cooldown_s=0.5)
    pestillo = PestilloGesto()
    salidas: list[str] = []
    ahora = 0.0

    secuencia = ["palma"] * 20 + ["neutro"] * 3 + ["palma"] * 20
    for gesto in secuencia:
        ahora += 0.05
        if pestillo.actualizar(gesto):
            disparo = filtro.actualizar(gesto, ahora)
        else:
            filtro.reiniciar()
            disparo = None
        if disparo is not None:
            pestillo.bloquear(disparo)
            salidas.append(disparo)

    # Dos presentaciones separadas por bajar la mano: dos acciones, no cuarenta.
    assert salidas == ["palma", "palma"]


def test_demo_combinado_cambia_de_gesto_tras_cinco_cuadros() -> None:
    """Cambiar de gesto dispara el nuevo sin repetir el anterior."""
    filtro = FiltroDisparo(frames_confirmacion=1, cooldown_s=0.5)
    pestillo = PestilloGesto()
    salidas: list[str] = []
    ahora = 0.0

    for gesto in ["puno"] * 20 + ["paz"] * 20:
        ahora += 0.05
        if pestillo.actualizar(gesto):
            disparo = filtro.actualizar(gesto, ahora)
        else:
            filtro.reiniciar()
            disparo = None
        if disparo is not None:
            pestillo.bloquear(disparo)
            salidas.append(disparo)

    assert salidas == ["puno", "paz"]
