---
name: capturar-dataset
description: Guía operativa paso a paso para la captura física humana de muestras gestuales con cámara y validación estadística del archivo CSV generado.
---

# Guía de Captura y Validación del Dataset

Este skill asiste al operador humano en el protocolo de captura física para garantizar un conjunto de datos robusto y balanceado.

## Protocolo de Preparación Física

1. **Iluminación y Fondo:**
   - Ubícate frente a una fuente de luz difusa uniforme (evitar contraluz directo de ventanas).
   - Mantén un fondo despejado o neutro para facilitar la segmentación de MediaPipe.
2. **Distancia y Encuadre:**
   - Posiciona la cámara (smartphone vía `scrcpy` o webcam) a una distancia de **40 a 70 cm**.
   - Asegúrate de que la mano completa (desde la muñeca hasta las puntas de los dedos) permanezca dentro del cuadro visual.
3. **Variabilidad Controlada:**
   - Graba tanto con la **mano derecha** como con la **mano izquierda** (aproximadamente 50% de muestras por lado).
   - Realiza ligeras variaciones de ángulo ($\pm 15^\circ$) y pequeñas variaciones de distancia dentro del rango útil.
   - En la clase `neutro`, relaja la mano, abre y cierra dedos aleatoriamente o mantenla en postura de descanso natural.

## Ejecución de Captura por Gesto

Ejecuta el script interactivo indicando la clase a capturar:
```bash
python scripts/capturar.py --clase palma --muestras 400
```
Repite el comando para cada una de las 6 clases:
1. `palma`
2. `senalar`
3. `paz`
4. `pulgar`
5. `puno`
6. `neutro`

## Validación de Calidad del CSV Resultante

Al terminar la sesión de captura, corre el siguiente comando de inspección:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/dataset.csv')
print('=== Resumen de Dataset ===')
print('Total instancias:', len(df))
print('Distribución por clase:\n', df['label'].value_counts())
print('Valores nulos:', df.isnull().sum().sum())
print('Manos detectadas:\n', df['handedness'].value_counts())
assert df.isnull().sum().sum() == 0, 'Hay valores nulos en el CSV'
assert all(df['label'].value_counts() >= 300), 'Hay clases con menos de 300 muestras'
print('>>> DATASET VÁLIDO Y LISTO PARA ENTRENAMIENTO <<<')
"
```
