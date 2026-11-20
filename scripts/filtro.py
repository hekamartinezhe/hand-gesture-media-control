"""Máquina de estados del filtro temporal anti-rebote (debounce + cooldown)."""

from __future__ import annotations

from typing import Optional

from scripts.utils import CLASES_GESTOS


class PestilloGesto:
    """Evita repetir la misma acción mientras el gesto se mantiene.

    Tras disparar, el pestillo se cierra. Se rearma cuando la mano baja
    (`cuadros_neutro` cuadros sin gesto o con `neutro`) o cuando aparece otro
    gesto distinto durante `cuadros_cambio` cuadros, lo que evita disparos por
    una clasificación errónea de un solo cuadro.
    """

    def __init__(
        self,
        cuadros_neutro: int = 3,
        cuadros_cambio: int = 5,
    ) -> None:
        if cuadros_neutro < 1 or cuadros_cambio < 1:
            raise ValueError("Los umbrales deben ser al menos 1")

        self.cuadros_neutro = cuadros_neutro
        self.cuadros_cambio = cuadros_cambio
        self._gesto: Optional[str] = None
        self._neutro = 0
        self._cambio: Optional[str] = None
        self._contador_cambio = 0

    @property
    def bloqueado(self) -> bool:
        """Indica si el último gesto sigue bloqueando nuevas acciones."""
        return self._gesto is not None

    @property
    def gesto(self) -> Optional[str]:
        """Gesto que disparó y mantiene cerrado el pestillo."""
        return self._gesto

    def bloquear(self, gesto: str) -> None:
        """Cierra el pestillo tras ejecutar la acción de `gesto`."""
        self._gesto = gesto
        self._neutro = 0
        self._cambio = None
        self._contador_cambio = 0

    def actualizar(self, gesto: Optional[str]) -> bool:
        """Actualiza el pestillo y devuelve True si permite un nuevo disparo."""
        if self._gesto is None:
            return True

        if gesto in (None, "neutro"):
            self._neutro += 1
            self._cambio = None
            self._contador_cambio = 0
            if self._neutro >= self.cuadros_neutro:
                self._gesto = None
                self._neutro = 0
            return False

        self._neutro = 0
        if gesto == self._gesto:
            self._cambio = None
            self._contador_cambio = 0
            return False

        if gesto == self._cambio:
            self._contador_cambio += 1
        else:
            self._cambio = gesto
            self._contador_cambio = 1

        if self._contador_cambio >= self.cuadros_cambio:
            self._gesto = None
            self._cambio = None
            self._contador_cambio = 0
            return True
        return False


class FiltroDisparo:
    """Confirma un gesto tras N cuadros consecutivos y bloquea el spam con cooldown.

    El tiempo se recibe como parámetro para poder testear la máquina de estados
    sin depender del reloj real.
    """

    def __init__(
        self,
        frames_confirmacion: int = 5,
        cooldown_s: float = 1.0,
    ) -> None:
        if frames_confirmacion < 1:
            raise ValueError("frames_confirmacion debe ser al menos 1")
        if cooldown_s < 0:
            raise ValueError("cooldown_s no puede ser negativo")

        self.frames_confirmacion = frames_confirmacion
        self.cooldown_s = cooldown_s
        self._candidato: Optional[str] = None
        self._contador = 0
        self._ultimo_disparo: Optional[float] = None

    @property
    def progreso(self) -> float:
        """Fracción del umbral de confirmación ya acumulada (0.0 a 1.0)."""
        return self._contador / self.frames_confirmacion

    @property
    def candidato(self) -> Optional[str]:
        """Gesto que está acumulando cuadros, o None si no hay candidato."""
        return self._candidato

    def cooldown_restante(self, ahora: float) -> float:
        """Segundos que faltan para que termine el cooldown (0.0 si no aplica)."""
        if self._ultimo_disparo is None:
            return 0.0
        return max(0.0, self.cooldown_s - (ahora - self._ultimo_disparo))

    def en_cooldown(self, ahora: float) -> bool:
        """Indica si todavía no se cumple el tiempo de enfriamiento."""
        return self.cooldown_restante(ahora) > 0.0

    def reiniciar(self) -> None:
        """Descarta el candidato y el contador, conservando el cooldown."""
        self._candidato = None
        self._contador = 0

    def actualizar(self, gesto: Optional[str], ahora: float) -> Optional[str]:
        """Procesa una predicción y devuelve el gesto solo cuando se confirma.

        Args:
            gesto: Clase predicha o None si no se detectó mano.
            ahora: Marca de tiempo monotónica en segundos.

        Returns:
            El gesto confirmado en este cuadro, o None si no hay disparo.
        """
        # "neutro", "sin mano" y clases desconocidas nunca disparan.
        if gesto not in CLASES_GESTOS or gesto == "neutro":
            self.reiniciar()
            return None

        # Durante el cooldown se ignoran las detecciones y se pierde el progreso.
        if self.en_cooldown(ahora):
            self.reiniciar()
            return None

        if gesto == self._candidato:
            self._contador += 1
        else:
            self._candidato = gesto
            self._contador = 1

        if self._contador < self.frames_confirmacion:
            return None

        # Quinto cuadro consecutivo: disparar una sola vez y arrancar el cooldown.
        self._candidato = None
        self._contador = 0
        self._ultimo_disparo = ahora
        return gesto
