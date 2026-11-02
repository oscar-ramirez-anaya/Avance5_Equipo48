"""Genera el reporte del Avance 5 en HTML, PDF y DOCX a partir de un solo contenido.

Las figuras se toman de la libreta ejecutada (Avance5.48.ipynb), de modo que
el reporte y la libreta siempre muestran los mismos resultados. Uso, desde la raíz del repositorio:
    python3 scripts/generar_documento_avance3.py
Requiere Google Chrome (PDF), PyMuPDF (pie de página e índice) y python-docx.
"""
import base64
import html
import io
import json
import pathlib
import re
import subprocess

import fitz
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

RAIZ = pathlib.Path(__file__).resolve().parent.parent
CARPETA = RAIZ
LIBRETA = CARPETA / "Avance5.48.ipynb"
CSS = CARPETA / "assets" / "print.css"
LOGO = CARPETA / "assets" / "logo_tec.png"
HTML = CARPETA / "Avance5.48.html"
PDF = CARPETA / "Avance5.48.pdf"
DOCX = CARPETA / "Avance5.48.docx"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO = "https://github.com/oscar-ramirez-anaya/Avance5_Equipo48"
FECHA = "1 de noviembre de 2026"

AZUL = RGBColor(0x0B, 0x3B, 0x8C)
AZUL_TABLA = "1346BD"


# ------------------------------------------------------------------ figuras de la libreta
def figuras_libreta():
    nb = json.loads(LIBRETA.read_text(encoding="utf-8"))
    figs = {}
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if "data" in o and "image/png" in o["data"]:
                figs[c["id"]] = o["data"]["image/png"]
    return figs


# ------------------------------------------------------------------ contenido
# Bloques: ("h2", id, texto) ("h3", id, texto) ("p", html) ("ul", [html]) ("ol", [html])
#          ("tabla", id, titulo, encabezados, filas) ("fig", id, celda, titulo) ("nota", titulo, html)
B = []
h2 = lambda i, t: B.append(("h2", i, t))
h3 = lambda i, t: B.append(("h3", i, t))
p = lambda t: B.append(("p", t))
ul = lambda xs: B.append(("ul", xs))
ol = lambda xs: B.append(("ol", xs))
tabla = lambda i, t, e, f: B.append(("tabla", i, t, e, f))
fig = lambda i, c, t: B.append(("fig", i, c, t))
nota = lambda t, x: B.append(("nota", t, x))

h2("s1", "1. Introducción")
h3("s11", "1.1 Propósito")
p("Este documento reporta el cierre de la fase de <b>Ingeniería del modelo</b> del proyecto, conforme a CRISP-ML(Q) (Visengeriyeva et al., 2023). Los mejores modelos individuales del Avance 4 se combinan en <b>modelos de ensamble</b> para identificar el SKU de cada producto, se comparan contra los individuales y se elige el modelo final con criterios de negocio. Se cubren los objetivos 3.5 (mejorar el rendimiento combinando las fortalezas de distintos modelos) y 3.6 (evaluar la calidad de las predicciones en datos no vistos).")
p(f"La libreta ejecutada que respalda cada cifra, <code>Avance5.48.ipynb</code>, está en el repositorio <code>{REPO}</code>.")
h3("s12", "1.2 Recomendaciones de la sesión con Bnext")
p("En la revisión del Avance 4 con Bnext, el 28 de octubre de 2026, se acordó cómo cerrar el pipeline del planograma: foto del anaquel, detector de una sola clase, recortes, embedding, comparación contra una galería de SKU y planograma. Las recomendaciones que guían este avance son:")
ol([
    "<b>Galería de SKU por carpeta.</b> El nombre de cada carpeta es el SKU y sus fotos son las referencias. Un recorte se identifica por similitud coseno contra la galería, con un umbral para declarar «desconocido». Un SKU nuevo se da de alta agregando su carpeta, sin reentrenar.",
    "<b>Aprendizaje de métrica</b> en lugar del entrenamiento auto-supervisado del Avance 4: ancla y positivo del mismo SKU, negativo de otro SKU, procesados por el mismo modelo.",
    "<b>DINOv2 ViT-S/14</b> como modelo base, con lotes de 2 a 4 fotos por SKU, distancia coseno y destilación hacia el modelo original.",
    "<b>Comparar tres pérdidas</b>: triplet con minado semi-difícil, multi-similitud y SupCon.",
    "<b>Fusión con color</b>: combinar DINOv2 con el histograma HSV, ganador del Avance 4.",
    "<b>Dejar SKU fuera del entrenamiento</b> para medir el reconocimiento de productos nuevos.",
    "<b>Pendientes:</b> reentrenar el detector con refrigeradores, correr el pipeline completo en uno o dos refrigeradores y solicitar a Bnext el planograma esperado.",
])
h3("s13", "1.3 Datos usados en esta ejecución")
p("Ejecutar con la galería de Bnext requiere el Jetson Orin NX y el servidor de Bnext. El equipo no tuvo acceso a ellos durante la ventana de entrega: el servidor no respondió a través de la VPN y el Jetson apareció fuera de línea. Como los datos de Bnext no pueden salir de su infraestructura, la libreta se ejecutó con un conjunto público que tiene la misma estructura que la galería: el <i>Grocery Store Dataset</i> (Klasson et al., 2019), con licencia MIT. Contiene fotos tomadas con un teléfono en supermercados, organizadas en una carpeta por producto, y una imagen icónica de catálogo por producto (Tabla 1).")
tabla("t1", "Conjunto de datos de esta ejecución", ["Grupo", "Productos", "Entrenamiento (galería)", "Validación", "Prueba"], [
    ["Empaque (leche, jugo, yogur, soya, avena)", "31", "864", "100", "781"],
    ["Fruta", "28", "1 142", "111", "1 117"],
    ["Verdura", "22", "634", "85", "587"],
    ["<b>Total</b>", "<b>81</b>", "<b>2 640</b>", "<b>296</b>", "<b>2 485</b>"],
])
nota("Alcance de las cifras", "Las cifras de este reporte <b>no</b> son resultados sobre datos de Bnext. Son evidencia del método, que se repetirá con la galería real cambiando solo la ruta de los datos. Los ensambles del detector (fusión de cajas de FCOS y Faster R-CNN) requieren los modelos entrenados en el Jetson y quedan para la siguiente ejecución.")
p("<b>Protocolo.</b> Se usa la partición oficial del conjunto, que es por foto. Las fotos de entrenamiento forman la galería, y un recorte se identifica con el SKU cuya carpeta contiene la foto más parecida. Diez productos se dejan fuera del entrenamiento del embedding, pero sí están en la galería: así se mide qué tan bien se reconoce un SKU dado de alta sin reentrenar. La <b>métrica principal es el F1 macro</b>, como en el Avance 3, porque cada SKU pesa igual en el anaquel. Como métricas de apoyo se usan top-1, top-5 (útil para revisión asistida) y el rango recíproco medio (MRR).")

h2("s2", "2. Modelos individuales")
p("Se usan como base los modelos individuales de la fase previa y el embedding recomendado en la sesión (Tabla 2). Los modelos de galería clasifican con kNN coseno: el puntaje de cada SKU es la similitud máxima contra las fotos de su carpeta.")
h3("s21", "2.1 Aprendizaje de métrica")
p("Se ajustan los 4 últimos bloques de DINOv2 ViT-S/14 con AdamW (tasa de 1e-5) durante 8 épocas. Cada lote tiene 64 fotos, 4 por SKU. A la pérdida de métrica se suma una destilación hacia el DINOv2 original (λ = 1). Las tres pérdidas mejoran mucho el top-1 en validación (Figura 1):")
ul([
    "SupCon: de 0.676 a 0.902, en la época 6;",
    "triplet semi-difícil: 0.899, en la época 8;",
    "multi-similitud: 0.885, en la época 7.",
])
p("SupCon y triplet empatan en la práctica: la diferencia es menos de una foto de validación. Se elige SupCon porque converge más rápido. Cada entrenamiento tardó unos 280 segundos en la GPU de una Mac M4 Pro.")
fig("f1", "celda-013", "Top-1 en validación y pérdida de entrenamiento por época para las tres pérdidas de métrica.")
h3("s22", "2.2 Resultados individuales")
tabla("t2", "Modelos individuales en prueba", ["Modelo", "Origen", "F1 macro", "Top-1", "Top-5", "MRR"], [
    ["Histograma HSV + kNN", "Final del Avance 4", "0.387", "0.418", "0.804", "0.585"],
    ["DINOv2 ViT-S/14 + kNN", "Finalista del Avance 4", "0.668", "0.701", "0.948", "0.809"],
    ["DINOv2 + métrica (SupCon) + kNN", "Recomendación de la sesión", "0.940", "0.934", "0.998", "0.965"],
    ["DINOv2 + métrica, solo imagen icónica", "Galería de una foto por SKU", "0.748", "0.751", "0.974", "0.851"],
    ["Regresión logística sobre DINOv2 + métrica", "Algoritmo del Avance 3", "0.954", "0.953", "0.999", "0.973"],
])
p("El aprendizaje de métrica es el cambio de mayor impacto: sube el F1 macro de DINOv2 de 0.668 a 0.940. El histograma HSV cae a 0.387, porque aquí hay frutas y verduras de colores parecidos y fotos con iluminación muy variable. Con los recortes de Bnext, de la misma cámara y el mismo anaquel, el color separaba mejor, así que su papel debe volver a medirse con la galería real.")

h2("s3", "3. Modelos de ensamble")
tabla("t3", "Ensambles construidos", ["#", "Ensamble", "Estrategia", "Modelos base", "Optimización"], [
    ["E1", "Bosque aleatorio (bagging)", "Homogénea", "Árboles sobre [HSV, DINOv2 + métrica]", "Optuna, 15 pruebas"],
    ["E2", "Gradient boosting", "Homogénea", "Árboles secuenciales sobre PCA de [HSV, DINOv2 + métrica]", "Optuna, 6 pruebas"],
    ["E3", "Votación ponderada", "Heterogénea", "HSV, DINOv2, DINOv2 + métrica, icónica", "Optuna sobre los pesos, 80 pruebas"],
    ["E4", "Stacking", "Heterogénea", "Los mismos cuatro, con un meta-modelo de regresión logística", "Rejilla sobre C con validación cruzada agrupada"],
    ["E5", "Blending", "Heterogénea", "Top-5 de DINOv2 + métrica re-ordenado con HSV", "Peso α en validación"],
])
p("Los ensambles heterogéneos usan los mejores modelos individuales: HSV y DINOv2 del Avance 4, y DINOv2 con métrica de la sección 2. El meta-modelo del stacking no aprende un peso por SKU. Recibe, para cada par (recorte, SKU candidato), los puntajes estandarizados de las cuatro bases y su distancia al mejor candidato, y estima la probabilidad de que ese candidato sea el correcto. Así, <b>un SKU nuevo se agrega a la galería sin reentrenar</b>.")
h3("s31", "3.1 Fuga en la validación cruzada dentro de entrenamiento")
p("La primera opción fue ajustar los ensambles con puntajes fuera de partición dentro de entrenamiento. La Tabla 4 muestra que esa estimación tiene fuga. Las fotos de cada producto se tomaron en ráfagas, así que en entrenamiento cada foto tiene gemelas casi idénticas en las otras particiones.")
tabla("t4", "Top-1 de cada modelo base según dónde se mide", ["Modelo base", "Fuera de partición (entrenamiento)", "Validación", "Prueba"], [
    ["HSV", "0.949", "0.358", "0.418"],
    ["DINOv2", "0.951", "0.676", "0.701"],
    ["DINOv2 + métrica", "0.996", "0.902", "0.934"],
    ["Imagen icónica", "0.795", "0.784", "0.751"],
])
p("Por eso <b>todos los hiperparámetros, pesos y meta-modelos se ajustaron con el conjunto de validación</b>, que es de otra sesión de fotos, y la prueba se usó una sola vez. Lo mismo aplica a las carpetas de Bnext, donde las capturas de un SKU son de una misma sesión (Avance 3).")
h3("s32", "3.2 Resultados de la optimización")
ul([
    "<b>E1:</b> 600 árboles, <code>max_features = log2</code>, hoja mínima de 2 y clases balanceadas; F1 en validación de 0.913.",
    "<b>E2:</b> tasa de 0.084, 25 hojas y 200 iteraciones; F1 en validación de 0.688.",
    "<b>E3:</b> pesos de 0.94 para DINOv2 + métrica, 0.43 para DINOv2, 0.12 para icónica y 0.06 para HSV.",
    "<b>E4:</b> C = 10 (F1 de 0.881 en la validación cruzada agrupada).",
    "<b>E5:</b> α = 0.95; HSV casi no cambia el orden del top-5.",
])

h2("s4", "4. Tabla comparativa y elección del modelo final")
tabla("t5", "Comparación en prueba, ordenada por F1 macro", ["#", "Modelo", "Tipo", "F1 macro", "Top-1", "Top-5", "MRR", "Entrenamiento (s)", "Inferencia (ms/img)", "Alta de SKU"], [
    ["1", "Regresión logística (DINOv2 + métrica)", "Individual", "0.954", "0.953", "0.999", "0.973", "0.1", "3.8", "Reentrenar"],
    ["2", "E1 Bosque aleatorio", "Homogéneo", "0.947", "0.945", "0.999", "0.970", "47.2", "3.9", "Reentrenar"],
    ["<b>3</b>", "<b>E4 Stacking</b>", "<b>Heterogéneo</b>", "<b>0.940</b>", "<b>0.937</b>", "<b>1.000</b>", "<b>0.966</b>", "<b>0.1</b>", "<b>7.6</b>", "<b>Sin reentrenar</b>"],
    ["4", "DINOv2 + métrica + kNN", "Individual", "0.940", "0.934", "0.998", "0.965", "283.9", "3.8", "Sin reentrenar"],
    ["5", "E5 Blending", "Heterogéneo", "0.935", "0.931", "0.998", "0.963", "0.0", "3.9", "Sin reentrenar"],
    ["6", "E3 Votación ponderada", "Heterogéneo", "0.930", "0.928", "0.999", "0.961", "0.1", "7.6", "Sin reentrenar"],
    ["7", "E2 Gradient boosting", "Homogéneo", "0.753", "0.771", "0.942", "0.848", "157.5", "4.2", "Reentrenar"],
    ["8", "DINOv2 + métrica, imagen icónica", "Individual", "0.748", "0.751", "0.974", "0.851", "283.9", "3.8", "Sin reentrenar"],
    ["9", "DINOv2 + kNN", "Individual", "0.668", "0.701", "0.948", "0.809", "9.9", "3.8", "Sin reentrenar"],
    ["10", "HSV + kNN", "Individual", "0.387", "0.418", "0.804", "0.585", "0.3", "0.1", "Sin reentrenar"],
])
p("El tiempo de entrenamiento incluye la optimización de hiperparámetros de cada modelo, sin sumar el de sus bases. El embedding con métrica, del que dependen casi todos, tardó 284 s. La latencia se midió en la GPU de una Mac (MPS) y se medirá en el Jetson con FP16.")
fig("f2", "celda-031", "F1 macro en prueba por modelo y relación entre calidad y latencia; el número de cada punto es la posición en la Tabla 5.")
p("<b>Modelo final: E4 Stacking.</b> Todos los modelos, salvo E2 y los individuales sin entrenar, superan la meta de negocio de 0.85. La regla de elección se definió en el Avance 4: dar de alta un SKU nuevo agregando su carpeta, sin reentrenar, porque el surtido de los anaqueles cambia con frecuencia.")
p("Entre los modelos que lo permiten, el stacking tiene el mayor F1 macro (0.940) y el mejor top-1 (0.937). Con top-5 de 1.000, en revisión asistida el SKU correcto siempre está entre los cinco primeros. Además es el más robusto con SKU nuevos (Tabla 6).")
p("La regresión logística (+0.014) y el bosque aleatorio (+0.007) son algo más precisos, pero obligan a reentrenar y volver a validar con cada cambio de catálogo. Una diferencia de una a dos fotos por cada cien no compensa ese costo de operación.")
p("Costos del stacking: su ganancia sobre DINOv2 con métrica sola es pequeña en SKU vistos (+0.003 de top-1), y necesita dos pasadas de DINOv2 (7.6 ms por imagen). Esa latencia es baja frente a los cerca de 100 ms por foto del detector en el Jetson. El gradient boosting fue el ensamble más débil: con 81 clases y unas 30 fotos por clase, los árboles sobre componentes principales pierden la geometría del embedding.")

h2("s5", "5. Análisis del modelo final")
h3("s51", "5.1 SKU nuevos")
tabla("t6", "Top-1 y top-5 en SKU vistos y nuevos (prueba)", ["Modelo", "Top-1 vistos", "Top-1 nuevos", "Top-5 vistos", "Top-5 nuevos"], [
    ["DINOv2 + kNN", "0.708", "0.645", "0.955", "0.900"],
    ["DINOv2 + métrica + kNN", "0.935", "0.928", "0.998", "1.000"],
    ["<b>E4 Stacking</b>", "<b>0.937</b>", "<b>0.941</b>", "<b>1.000</b>", "<b>1.000</b>"],
])
p("El aprendizaje de métrica generaliza a productos que no vio al entrenar: 0.928 de top-1 en SKU nuevos contra 0.935 en vistos. El stacking llega a 0.941 en los SKU nuevos, porque la imagen icónica y el DINOv2 original aportan señal que no depende del entrenamiento. Esto respalda el flujo de alta sin reentrenar. Solo se probaron 10 productos, así que debe repetirse con la galería de Bnext.")
h3("s52", "5.2 Matriz de confusión")
fig("f3", "celda-038", "Matriz de confusión normalizada por fila de los 15 SKU que participan en más errores.")
p("Los errores se concentran en <b>variantes de la misma marca o familia</b>:")
ul([
    "Golden Delicious con Granny Smith (15 errores);",
    "leche Arla semidescremada con su versión deslactosada (12);",
    "leche de soya Alpro de anaquel con la refrigerada (12);",
    "satsumas con naranja (11);",
    "yogur Yoggi de fresa con el de vainilla (9).",
])
p("En los productos en empaque cambia solo un texto o un color pequeño del envase, el mismo riesgo identificado para los refrigeradores de Bnext. Esto orienta dos acciones: usar esas variantes como negativos difíciles en el entrenamiento con métrica, y probar CLIP o SigLIP, que leen texto y logotipos.")
h3("s53", "5.3 Curvas ROC y precisión-recall: productos desconocidos")
p("Para medir si el modelo distingue un producto que no está en la galería, se retiraron de ella los 10 SKU nuevos y sus fotos de prueba se trataron como desconocidas (clase positiva).")
fig("f4", "celda-041", "Curvas ROC y precisión-recall para detectar un SKU que no está en la galería.")
p("Esta es la limitación principal del modelo elegido. La similitud máxima de DINOv2 con métrica separa mejor los desconocidos: AUC de 0.896 y precisión promedio de 0.432, frente a un azar de 0.117. El stacking queda en 0.776 y 0.287, porque su puntaje está construido para elegir entre candidatos y no para medir qué tan familiar es el producto. HSV casi no separa (AUC de 0.594).")
p("Se recomienda un pipeline en dos pasos: primero, un filtro de «desconocido» con umbral sobre la similitud de DINOv2 con métrica; después, la identificación con el stacking. Con una precisión promedio de 0.43, los casos marcados como desconocidos deben ir a revisión humana.")
h3("s54", "5.4 Puntajes y umbral de decisión")
fig("f5", "celda-044", "Distribución del puntaje del candidato elegido en aciertos y errores, y exactitud de lo decidido automáticamente al subir el umbral.")
p("Los aciertos tienen puntajes de 1.5 a 4 y los errores de 0 a 1.5, con una zona de traslape. Al subir el umbral, la exactitud de lo que se decide automáticamente sube de 0.937 (todo) a 0.955 (90 %), 0.972 (80 %) y 0.980 (70 %).")
p("Una política razonable es decidir automáticamente alrededor del 80 % de los productos y mandar el resto a revisión con sus cinco candidatos. El umbral final se fijará con Bnext según el costo de una alerta falsa frente al de una revisión manual.")
h3("s55", "5.5 Importancia de los modelos base")
fig("f6", "celda-047", "Coeficientes del meta-modelo del stacking y caída del F1 macro al retirar cada modelo base.")
p("El stacking depende sobre todo de DINOv2 con métrica. Retirarlo baja el F1 macro en 0.081; retirar la imagen icónica, en 0.006. El DINOv2 original y el HSV casi no cambian el resultado.")
p("Los coeficientes más altos son la distancia al mejor candidato de DINOv2 con métrica (4.5) y su puntaje estandarizado (2.4). Los coeficientes negativos de los puntajes estandarizados deben leerse junto con las distancias al mejor, con las que están correlacionados. El color no aporta en este conjunto, a diferencia del Avance 4 con datos de Bnext; su peso debe volver a estimarse con la galería real.")
h3("s56", "5.6 Desempeño por SKU")
fig("f7", "celda-050", "F1 por SKU con el modelo final, por grupo de producto; el asterisco señala los SKU nuevos.")
p("Las verduras son el grupo más fácil (F1 medio de 0.972) y las frutas siguen (0.935). Los productos en empaque son los más difíciles (0.923; mínimo de 0.683), justamente el grupo más parecido a los productos de refrigerador. Los SKU con menor F1 tienen todos una variante casi idéntica en la galería: yogur Yoggi de vainilla y de fresa, leches de soya Alpro y leche Arla semidescremada. Para Bnext, la prioridad de captura de fotos y de negativos difíciles deben ser las presentaciones de una misma marca.")

h2("s6", "6. Siguiente ejecución con los datos de Bnext")
ol([
    "Copiar al Jetson las carpetas de SKU y las escenas de refrigerador, sin que salgan de la infraestructura de Bnext, y ejecutar la libreta con la ruta de esos datos.",
    "Ensambles del detector: fusión ponderada de cajas (WBF; Solovyev et al., 2021) de FCOS y Faster R-CNN v2, que es heterogénea, y FCOS con aumentos en prueba, que es homogénea. Se ajustarán en validación con el protocolo del Avance 4.",
    "Correr el pipeline completo en uno o dos refrigeradores (detección, SKU, hilera y posición, huecos), reportando solo resultados agregados.",
    "Solicitar a Bnext el planograma esperado y la correspondencia de las carpetas de agrupamiento con productos.",
])

h2("s7", "7. Conclusiones")
ol([
    "<b>Ensambles.</b> Se construyeron cinco con optimización de hiperparámetros: dos homogéneos (bosque aleatorio y gradient boosting) y tres heterogéneos (votación, stacking y blending) sobre los mejores modelos individuales.",
    "<b>Mayor mejora.</b> Viene del aprendizaje de métrica recomendado por Bnext: el F1 macro pasa de 0.668 a 0.940, y los SKU nuevos se reconocen casi igual que los vistos.",
    "<b>Modelo final: stacking.</b> F1 macro de 0.940, top-1 de 0.937 y top-5 de 1.000. Es el mejor de los modelos que permiten dar de alta un SKU sin reentrenar.",
    "<b>Hallazgo metodológico.</b> Ajustar ensambles con validación cruzada dentro de las fotos de entrenamiento tiene fuga, por las ráfagas de fotos. Hay que ajustar con fotos de otra sesión.",
    "<b>Limitaciones.</b> Las cifras son de un conjunto público y no de Bnext. Detectar productos desconocidos todavía no es confiable para automatizar. Los ensambles del detector están pendientes.",
])

h2("anexo", "Anexo A. Declaración de uso de IA y contribución del equipo")
tabla("ta1", "Uso de herramientas de IA", ["Herramienta utilizada", "Propósito de uso", "Responsabilidad: qué revisó o validó el equipo"], [
    ["Anthropic. (2026). Claude (Opus 5.5) [Modelo de lenguaje grande]. https://claude.ai/",
     "Generar código de la libreta (carga de datos, aprendizaje de métrica, ensambles, optimización con Optuna y gráficas), búsqueda de un conjunto público con la misma estructura que la galería de Bnext, apoyo en la redacción de las interpretaciones y del reporte.",
     "El equipo definió los modelos, las pérdidas y los ensambles a partir de la sesión con Bnext del 28 de octubre, revisó el código generado, ejecutó la libreta completa de principio a fin y validó cada cifra e interpretación contra las salidas."],
])
tabla("ta2", "Contribución individual", ["Integrante", "Contribución al avance"], [
    ["Oscar Alberto Ramírez Anaya", "Incorporación de las recomendaciones de la sesión con Bnext, selección y preparación del conjunto público sustituto, entrenamiento del embedding con aprendizaje de métrica, ensambles homogéneos (bosque aleatorio y gradient boosting), ejecución de la libreta, reporte y repositorio."],
    ["Jesús Soto Medrano", "Propuesta de modelos y pérdidas de métrica en la sesión con Bnext (triplet, multi-similitud, SupCon y fusión con HSV), ensambles heterogéneos (votación, stacking y blending), tabla comparativa, elección del modelo final y análisis de las gráficas."],
])

BIBLIO = [
    "Akiba, T., Sano, S., Yanase, T., Ohta, T., y Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. En <i>Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining</i> (pp. 2623–2631). ACM. https://doi.org/10.1145/3292500.3330701",
    "Breiman, L. (1996). Bagging predictors. <i>Machine Learning, 24</i>(2), 123–140. https://doi.org/10.1007/BF00058655",
    "Breiman, L. (2001). Random forests. <i>Machine Learning, 45</i>(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. <i>The Annals of Statistics, 29</i>(5), 1189–1232. https://doi.org/10.1214/aos/1013203451",
    "Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., y Krishnan, D. (2020). Supervised contrastive learning. En <i>Advances in Neural Information Processing Systems 33</i> (pp. 18661–18673).",
    "Klasson, M., Zhang, C., y Kjellström, H. (2019). A hierarchical grocery store image dataset with visual and semantic labels. En <i>2019 IEEE Winter Conference on Applications of Computer Vision</i> (pp. 491–500). IEEE. https://doi.org/10.1109/WACV.2019.00058",
    "Musgrave, K., Belongie, S., y Lim, S.-N. (2020). PyTorch metric learning. arXiv. https://doi.org/10.48550/arXiv.2008.09164",
    "Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without supervision. <i>Transactions on Machine Learning Research</i>.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830.",
    "Schroff, F., Kalenichenko, D., y Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. En <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</i> (pp. 815–823). IEEE. https://doi.org/10.1109/CVPR.2015.7298682",
    "Singh, A. (2023, 14 de julio). <i>Comprehensive guide to ensemble learning (with Python codes)</i>. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/",
    "Solovyev, R., Wang, W., y Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. <i>Image and Vision Computing, 107</i>, 104117. https://doi.org/10.1016/j.imavis.2021.104117",
    "VanderPlas, J. (2022). <i>Python data science handbook</i> (2a ed.). O'Reilly Media.",
    "Visengeriyeva, L., Kammer, A., Bär, I., Kniesz, A., y Plöd, M. (2023). <i>CRISP-ML(Q). The ML lifecycle process</i>. MLOps. INNOQ. https://ml-ops.org/content/crisp-ml",
    "Wang, X., Han, X., Huang, W., Dong, D., y Scott, M. R. (2019). Multi-similarity loss with general pair weighting for deep metric learning. En <i>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition</i> (pp. 5022–5030). IEEE.",
    "Wolpert, D. H. (1992). Stacked generalization. <i>Neural Networks, 5</i>(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1",
]


INTEGRANTES = [("Jesús Soto Medrano", "A01794746"), ("Germán Eduardo Rodríguez Vázquez", "A01796325"),
               ("Oscar Alberto Ramírez Anaya", "A01795438")]
TITULO = "Planograma: IA para detección de productos y ubicación"
SUBTITULO = "Ensambles homogéneos y heterogéneos para identificar el SKU de cada producto contra una galería de referencia, y elección del modelo final"


# ------------------------------------------------------------------ HTML
def generar_html(figs, paginas):
    def pag(ref):
        return paginas.get(ref, "")
    tablas = [b for b in B if b[0] == "tabla"]
    figuras = [b for b in B if b[0] == "fig"]
    num_t = {b[1]: i + 1 for i, b in enumerate(tablas)}
    num_f = {b[1]: i + 1 for i, b in enumerate(figuras)}
    logo = base64.b64encode(LOGO.read_bytes()).decode()
    o = ['<!DOCTYPE html>\n<html lang="es">\n<head>\n<meta charset="utf-8">',
         "<title>Avance 5. Modelo final - Equipo 48</title>",
         '<link rel="stylesheet" href="assets/print.css">',
         "<style>.logo-tec{width:62mm;margin:0 auto 4mm auto;display:block}"
         "figure img{max-width:100%;max-height:150mm}"
         "code{font-family:Menlo,Consolas,monospace;font-size:9pt;hyphens:none}</style>",
         "</head>\n<body>",
         '<section class="portada">\n<div class="portada-encabezado">',
         f'<img class="logo-tec" alt="Tecnológico de Monterrey" src="data:image/png;base64,{logo}">',
         '<p class="programa">Maestría en Inteligencia Artificial Aplicada</p>',
         '<p class="curso">TC5035.10 — Proyecto Integrador (Grupo 10) · Trimestre septiembre – diciembre 2026</p>',
         '<hr class="regla">\n</div>\n<div>',
         '<p class="etiqueta">Avance 5 · Modelo final</p>',
         f'<h1 class="titulo">{TITULO}</h1>', f'<p class="subtitulo">{SUBTITULO}</p>',
         '</div>\n<div>\n<p class="equipo">Equipo 48</p>\n<ul class="integrantes">']
    o += [f'<li><span>{n}</span><span class="matricula">{m}</span></li>' for n, m in INTEGRANTES]
    o += ["</ul>\n</div>", '<div class="portada-pie">',
          "<p><strong>Profesora:</strong> Dra. Grettel Barceló Alonso</p>",
          "<p><strong>Asesor:</strong> Dr. Raúl Valente Ramírez</p>",
          "<p><strong>Sponsor:</strong> Dr. Juan Arturo Nolazco Flores, Director del Hub de Ciencia de Datos, Tecnológico de Monterrey</p>",
          "<p><strong>Organización:</strong> Bnext, Monterrey, Nuevo León</p>",
          f'<p style="margin-top:4mm">{FECHA}</p>', "</div>\n</section>"]
    o.append('<h2>Índice</h2>\n<ul class="indice">')
    for b in B:
        if b[0] in ("h2", "h3"):
            cls = ' class="sub"' if b[0] == "h3" else ""
            o.append(f'<li{cls}><span>{b[2]}</span><span class="relleno"></span><span class="pag">{pag(b[1])}</span></li>')
    o.append(f'<li><span>Bibliografía</span><span class="relleno"></span><span class="pag">{pag("biblio")}</span></li></ul>')
    o.append('<h3 class="lista-indice-titulo">Índice de tablas</h3>\n<ul class="indice">')
    for b in tablas:
        o.append(f'<li class="sub"><span>Tabla {num_t[b[1]]}. {b[2]}</span><span class="relleno"></span><span class="pag">{pag(b[1])}</span></li>')
    o.append('</ul>\n<h3 class="lista-indice-titulo">Índice de figuras</h3>\n<ul class="indice">')
    for b in figuras:
        corto = b[3].split(". ")[0].rstrip(".")
        o.append(f'<li class="sub"><span>Figura {num_f[b[1]]}. {corto}</span><span class="relleno"></span><span class="pag">{pag(b[1])}</span></li>')
    o.append("</ul>")
    primero = True
    for b in B:
        t = b[0]
        if t == "h2":
            salto = ' class="salto"' if primero else ""
            o.append(f'<h2{salto} id="{b[1]}">{b[2]}</h2>'); primero = False
        elif t == "h3":
            o.append(f'<h3 id="{b[1]}">{b[2]}</h3>')
        elif t == "p":
            o.append(f"<p>{b[1]}</p>")
        elif t in ("ul", "ol"):
            o.append(f"<{t}>" + "".join(f"<li>{x}</li>" for x in b[1]) + f"</{t}>")
        elif t == "nota":
            o.append(f'<div class="nota"><p class="titulo-nota">{b[1]}</p><p>{b[2]}</p></div>')
        elif t == "tabla":
            _, i, tit, enc, filas = b
            o.append(f'<table id="{i}"><caption><b>Tabla {num_t[i]}.</b> {tit}</caption><thead><tr>'
                     + "".join(f"<th>{e}</th>" for e in enc) + "</tr></thead><tbody>"
                     + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in f) + "</tr>" for f in filas)
                     + "</tbody></table>")
        elif t == "fig":
            _, i, celda, tit = b
            o.append(f'<figure id="{i}"><img alt="Figura {num_f[i]}" src="data:image/png;base64,{figs[celda]}">'
                     f"<figcaption><b>Figura {num_f[i]}.</b> {tit} Elaboración propia con datos del Grocery Store Dataset (Klasson et al., 2019).</figcaption></figure>")
    o.append('<h2 id="biblio">Bibliografía</h2>\n<div class="biblio">')
    o += [f"<p>{x}</p>" for x in BIBLIO]
    o.append("</div>\n</body>\n</html>\n")
    HTML.write_text("\n".join(o), encoding="utf-8")


def imprimir_pdf():
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True)


def localizar_paginas():
    """Página (1-based) donde aparece cada encabezado, tabla y figura, después del índice."""
    doc = fitz.open(PDF)
    textos = [pg.get_text() for pg in doc]
    paginas = {}
    tablas = [b for b in B if b[0] == "tabla"]
    figuras = [b for b in B if b[0] == "fig"]
    # El índice repite los títulos: la búsqueda empieza después de su última entrada
    ultima = f"Figura {len(figuras)}."
    desde = next(n for n in range(2, len(textos) + 1) if ultima in textos[n - 1]) + 1
    num_t = {b[1]: i + 1 for i, b in enumerate(tablas)}
    num_f = {b[1]: i + 1 for i, b in enumerate(figuras)}
    def buscar(clave):
        for n in range(desde, len(textos) + 1):
            if clave in textos[n - 1]:
                return n
        return ""
    for b in B:
        if b[0] in ("h2", "h3"):
            paginas[b[1]] = buscar(b[2])
        elif b[0] == "tabla":
            paginas[b[1]] = buscar(f"Tabla {num_t[b[1]]}.")
        elif b[0] == "fig":
            paginas[b[1]] = buscar(f"Figura {num_f[b[1]]}.")
    paginas["biblio"] = buscar("Bibliografía")
    return paginas


def pie_pdf():
    doc = fitz.open(PDF)
    total = doc.page_count
    for i, pg in enumerate(doc, start=1):
        if i == 1:
            continue
        w, h = pg.rect.width, pg.rect.height
        y = h - 42.5
        pg.draw_line(fitz.Point(56.7, y - 9), fitz.Point(w - 56.7, y - 9), color=(0.79, 0.81, 0.84), width=0.5)
        pg.insert_text(fitz.Point(56.7, y + 2), "TC5035.10 Proyecto Integrador  ·  Avance 5  ·  Equipo 48",
                       fontname="helv", fontsize=7.5, color=(0.42, 0.42, 0.42))
        der = f"Página {i} de {total}"
        pg.insert_text(fitz.Point(w - 56.7 - fitz.get_text_length(der, fontname="helv", fontsize=7.5), y + 2),
                       der, fontname="helv", fontsize=7.5, color=(0.42, 0.42, 0.42))
    doc.set_metadata({"title": "Avance 5. Modelo final — Equipo 48", "author": "Equipo 48",
                      "subject": "TC5035.10 Proyecto Integrador — Avance 5",
                      "keywords": "planograma, ensambles, stacking, aprendizaje de metrica, DINOv2, F1 macro, CRISP-ML(Q)"})
    doc.saveIncr()
    return total


# ------------------------------------------------------------------ DOCX
def runs_desde_html(par, texto, tam=None, color=None, fuente=None):
    """Convierte el marcado mínimo (<b>, <i>, <code>) a runs de Word."""
    estado = {"b": False, "i": False, "code": False}
    for trozo in re.split(r"(</?(?:b|i|code)>)", texto):
        m = re.fullmatch(r"<(/?)(b|i|code)>", trozo)
        if m:
            estado[m.group(2)] = m.group(1) == ""
            continue
        if not trozo:
            continue
        r = par.add_run(html.unescape(trozo))
        r.bold = estado["b"] or None
        r.italic = estado["i"] or None
        if estado["code"]:
            r.font.name = "Consolas"; r.font.size = Pt(9)
        elif fuente:
            r.font.name = fuente
        if tam and not estado["code"]:
            r.font.size = Pt(tam)
        if color:
            r.font.color.rgb = color


def sombrear(celda, hexcolor):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def campo(par, instruccion):
    for tipo, texto in (("begin", None), (None, instruccion), ("separate", None), (None, "1"), ("end", None)):
        r = par.add_run()
        if tipo:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), tipo); r._r.append(fc)
        elif texto == instruccion:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instruccion; r._r.append(it)
        else:
            r.text = texto


def generar_docx(figs, paginas):
    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.4)
    normal = d.styles["Normal"]
    normal.font.name = "Georgia"; normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.15
    for nombre, tam in (("Heading 1", 15), ("Heading 2", 12)):
        st = d.styles[nombre]
        st.font.name = "Arial"; st.font.size = Pt(tam); st.font.color.rgb = AZUL; st.font.bold = True
        rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), "Arial")
    def centrado(texto, tam=None, color=None, fuente=None, negrita=False, antes=0):
        par = d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(antes)
        runs_desde_html(par, f"<b>{texto}</b>" if negrita else texto, tam, color, fuente)
        return par
    # portada
    par = d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(str(LOGO), width=Cm(6.2))
    centrado("Maestría en Inteligencia Artificial Aplicada", 11, RGBColor(0x44, 0x44, 0x44), antes=30)
    centrado("TC5035.10 — Proyecto Integrador (Grupo 10) · Trimestre septiembre – diciembre 2026", 10, RGBColor(0x44, 0x44, 0x44))
    centrado("AVANCE 3 · BASELINE", 8.5, RGBColor(0x66, 0x66, 0x66), "Arial", antes=40)
    centrado(TITULO, 20, AZUL, "Arial", negrita=True, antes=6)
    centrado(f"<i>{SUBTITULO}</i>", 10.5, RGBColor(0x55, 0x55, 0x55))
    centrado("Equipo 48", 13, None, "Arial", negrita=True, antes=40)
    t = d.add_table(rows=0, cols=2); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for n, m in INTEGRANTES:
        fila = t.add_row().cells
        fila[0].text = n; fila[1].text = m
        fila[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    centrado("<b>Profesora:</b> Dra. Grettel Barceló Alonso", 9.5, antes=50)
    centrado("<b>Asesor:</b> Dr. Raúl Valente Ramírez", 9.5)
    centrado("<b>Sponsor:</b> Dr. Juan Arturo Nolazco Flores, Director del Hub de Ciencia de Datos, Tecnológico de Monterrey", 9.5)
    centrado("<b>Organización:</b> Bnext, Monterrey, Nuevo León", 9.5)
    centrado(FECHA, 9.5, RGBColor(0x66, 0x66, 0x66), antes=8)
    # índice (campo TOC; Word lo actualiza al abrir)
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    par = d.add_paragraph(); runs_desde_html(par, "Índice", 15, AZUL, "Arial")
    campo(d.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
    ajustes = d.settings.element
    uf = OxmlElement("w:updateFields"); uf.set(qn("w:val"), "true"); ajustes.append(uf)
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    # cuerpo
    tablas = [b for b in B if b[0] == "tabla"]; figuras = [b for b in B if b[0] == "fig"]
    num_t = {b[1]: i + 1 for i, b in enumerate(tablas)}; num_f = {b[1]: i + 1 for i, b in enumerate(figuras)}
    for b in B:
        tp = b[0]
        if tp == "h2":
            d.add_heading(b[2], level=1)
        elif tp == "h3":
            d.add_heading(b[2], level=2)
        elif tp == "p":
            par = d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; runs_desde_html(par, b[1])
        elif tp in ("ul", "ol"):
            for x in b[1]:
                par = d.add_paragraph(style="List Number" if tp == "ol" else "List Bullet"); runs_desde_html(par, x)
        elif tp == "nota":
            tn = d.add_table(rows=1, cols=1); c = tn.rows[0].cells[0]; sombrear(c, "F3F6FB")
            runs_desde_html(c.paragraphs[0], f"<b>{b[1]}</b>", 9.5, AZUL, "Arial")
            runs_desde_html(c.add_paragraph(), b[2], 9.5)
            d.add_paragraph()
        elif tp == "tabla":
            _, i, tit, enc, filas = b
            par = d.add_paragraph(); par.paragraph_format.keep_with_next = True
            runs_desde_html(par, f"<b>Tabla {num_t[i]}.</b> {tit}", 9, AZUL, "Arial")
            tb = d.add_table(rows=1, cols=len(enc)); tb.style = "Table Grid"
            for j, e in enumerate(enc):
                c = tb.rows[0].cells[j]; sombrear(c, AZUL_TABLA)
                runs_desde_html(c.paragraphs[0], f"<b>{e}</b>", 9, RGBColor(0xFF, 0xFF, 0xFF), "Arial")
            for f in filas:
                cs = tb.add_row().cells
                for j, v in enumerate(f):
                    runs_desde_html(cs[j].paragraphs[0], v, 9)
            d.add_paragraph()
        elif tp == "fig":
            _, i, celda, tit = b
            par = d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.keep_with_next = True
            par.add_run().add_picture(io.BytesIO(base64.b64decode(figs[celda])), width=Cm(16))
            par = d.add_paragraph()
            runs_desde_html(par, f"<b>Figura {num_f[i]}.</b> {tit} Elaboración propia con datos del Grocery Store Dataset (Klasson et al., 2019).", 9, None, "Arial")
    d.add_heading("Bibliografía", level=1)
    for x in BIBLIO:
        par = d.add_paragraph(); par.paragraph_format.left_indent = Cm(1); par.paragraph_format.first_line_indent = Cm(-1)
        runs_desde_html(par, x, 10)
    # pie con número de página
    sec.different_first_page_header_footer = True
    pie = sec.footer.paragraphs[0]
    runs_desde_html(pie, "TC5035.10 Proyecto Integrador  ·  Avance 5  ·  Equipo 48        Página ", 7.5, RGBColor(0x6B, 0x6B, 0x6B), "Arial")
    campo(pie, "PAGE")
    d.core_properties.author = "Equipo 48"
    d.core_properties.title = "Avance 5. Modelo final — Equipo 48"
    d.save(DOCX)


if __name__ == "__main__":
    figs = figuras_libreta()
    generar_html(figs, {})
    imprimir_pdf()
    paginas = localizar_paginas()
    generar_html(figs, paginas)
    imprimir_pdf()
    assert localizar_paginas() == paginas, "La paginación cambió al insertar los números del índice"
    total = pie_pdf()
    generar_docx(figs, paginas)
    print(f"{PDF.name}: {total} páginas | {HTML.name} | {DOCX.name}")
