"""Tests unitarios para el despachador multimedia multiplataforma."""

from unittest.mock import MagicMock, patch
import pytest

from scripts.utils import (
    CODIGOS_VK_WINDOWS,
    COMANDOS_LINUX,
    GESTO_A_ACCION,
    despachar_gesto,
    ejecutar_comando_multimedia,
)


def test_gesto_neutro_no_ejecuta_accion() -> None:
    """Verifica que la clase neutro jamás despacha ninguna acción."""
    assert GESTO_A_ACCION["neutro"] is None
    assert despachar_gesto("neutro") is False


def test_gesto_invalido_retorna_falso() -> None:
    """Verifica que un gesto desconocido no ejecuta nada."""
    assert despachar_gesto("inexistente") is False


@patch("subprocess.run")
def test_despacho_linux_exitoso(mock_run: MagicMock) -> None:
    """Verifica que en Linux se despachan los comandos playerctl correctos."""
    mock_run.return_value = MagicMock(returncode=0)

    resultado = ejecutar_comando_multimedia("play_pause", sistema="Linux")
    assert resultado is True
    mock_run.assert_called_once_with(
        ["playerctl", "play-pause"],
        check=False,
        capture_output=True,
        text=True,
    )


@patch("subprocess.run")
def test_despacho_linux_playerctl_no_encontrado(mock_run: MagicMock) -> None:
    """Verifica manejo elegante si playerctl no está instalado en Linux."""
    mock_run.side_effect = FileNotFoundError

    resultado = ejecutar_comando_multimedia("next", sistema="Linux")
    assert resultado is False


def test_despacho_windows_sintetiza_eventos() -> None:
    """Verifica que en Windows se invocan las Virtual Keys correctas."""
    mock_ctypes = MagicMock()

    with patch.dict("sys.modules", {"ctypes": mock_ctypes}):
        resultado = ejecutar_comando_multimedia("play_pause", sistema="Windows")
        assert resultado is True
        # Debe llamar presionar (flag 0) y soltar (flag 2)
        mock_ctypes.windll.user32.keybd_event.assert_any_call(
            CODIGOS_VK_WINDOWS["play_pause"], 0, 0, 0
        )
        mock_ctypes.windll.user32.keybd_event.assert_any_call(
            CODIGOS_VK_WINDOWS["play_pause"], 0, 2, 0
        )


def test_mapeo_completo_gestos_activos() -> None:
    """Verifica que los 5 gestos activos tienen acciones asignadas."""
    gestos_activos = ["palma", "senalar", "paz", "pulgar", "puno"]
    for g in gestos_activos:
        accion = GESTO_A_ACCION[g]
        assert accion is not None
        assert accion in COMANDOS_LINUX
        assert accion in CODIGOS_VK_WINDOWS
