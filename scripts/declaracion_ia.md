# Declaración de uso de IA y contribución del equipo

### Uso de herramientas de IA

| Herramienta utilizada | Propósito de uso | Responsabilidad: qué revisó o validó el equipo |
|---|---|---|
| Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande].<br>https://claude.ai/ | Generar código de la libreta (carga de datos, aprendizaje de métrica, ensambles, optimización con Optuna y gráficas), búsqueda de un conjunto público con la misma estructura que la galería de Bnext, apoyo en la redacción de las interpretaciones y del reporte. | El equipo definió los modelos, las pérdidas y los ensambles a partir de la sesión con Bnext del 28 de octubre, revisó el código generado, ejecutó la libreta completa de principio a fin y validó cada cifra e interpretación contra las salidas. |

### Contribución individual

| Integrante | Contribución al avance |
|---|---|
| Oscar Alberto Ramírez Anaya | Incorporación de las recomendaciones de la sesión con Bnext, selección y preparación del conjunto público sustituto, entrenamiento del embedding con aprendizaje de métrica, ensambles homogéneos (bosque aleatorio y gradient boosting), ejecución de la libreta, reporte y repositorio. |
| Jesús Soto Medrano | Propuesta de modelos y pérdidas de métrica en la sesión con Bnext (triplet, multi-similitud, SupCon y fusión con HSV), ensambles heterogéneos (votación, stacking y blending), tabla comparativa, elección del modelo final y análisis de las gráficas. |

### Decisiones principales

<small><b>Decisión 1:</b> Identificar el SKU con una galería por carpeta y un embedding entrenado con aprendizaje de métrica.</small>

- <small>Justificación: un SKU nuevo se da de alta agregando su carpeta, sin reentrenar; el aprendizaje de métrica acerca fotos del mismo producto y separa productos distintos, que es exactamente lo que necesita la comparación contra la galería.</small>
- <small>Alternativa descartada y motivo: un clasificador con una salida por SKU, porque obliga a reentrenar cada vez que cambia el surtido.</small>

<small><b>Decisión 2:</b> Ajustar los ensambles contra el conjunto de validación y no con validación cruzada dentro de entrenamiento.</small>

- <small>Justificación: las fotos de cada producto se tomaron en ráfagas; dentro de entrenamiento cada foto tiene gemelas casi idénticas y la validación cruzada sobreestima mucho el desempeño (fuga medida en la sección 5).</small>
- <small>Alternativa descartada y motivo: puntajes fuera de partición dentro de entrenamiento, porque daban pesos y meta-modelos ajustados a una señal inflada.</small>

<small><b>Decisión 3:</b> Ejecutar con un conjunto público mientras no hubo acceso a la infraestructura de Bnext.</small>

- <small>Justificación: el servidor y el Jetson no estuvieron disponibles en la ventana de entrega y los datos de Bnext no pueden salir de su infraestructura; el conjunto público tiene la misma estructura (una carpeta por producto) y licencia MIT.</small>
- <small>Alternativa descartada y motivo: entregar el código sin ejecutar, porque la libreta debe ejecutarse de principio a fin.</small>
