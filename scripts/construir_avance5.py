"""Genera Avance5.48.ipynb a partir de las celdas definidas aquí.

La libreta se mantiene como código fuente plano para revisar diffs y regenerarla:
    python3 scripts/construir_avance5.py
Las interpretaciones se escriben después de revisar la ejecución, en scripts/avance5_textos.json.
"""
import json
import pathlib

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "Avance5.48.ipynb"
TEXTOS = RAIZ / "scripts" / "avance5_textos.json"

celdas = []


def md(texto):
    celdas.append(("markdown", texto.strip("\n")))


def code(texto):
    celdas.append(("code", texto.strip("\n")))


def interpretacion(clave):
    """Texto interpretativo escrito tras revisar los resultados de la ejecución."""
    textos = json.loads(TEXTOS.read_text(encoding="utf-8")) if TEXTOS.exists() else {}
    md(textos.get(clave, f"**Interpretación ({clave}):** pendiente."))


# ====================================================================== portada
md("""
# Avance 5. Modelo final

**Proyecto Integrador (TC5035.10) · Equipo 48 · Planograma: IA para detección de productos y ubicación**

| Integrante | Matrícula |
|---|---|
| Oscar Alberto Ramírez Anaya | A01795438 |
| Germán Eduardo Rodríguez Vázquez | A01796325 |
| Jesús Soto Medrano | A01794746 |

Profesora: Dra. Grettel Barceló Alonso · Asesor: Dr. Raúl Valente Ramírez · Patrocinador: Dr. Juan Arturo Nolazco Flores · Organización: empresa sponsor

---

**Fase CRISP-ML(Q):** Ingeniería del modelo, cierre. En este avance se combinan los mejores modelos
individuales del Avance 4 en **modelos de ensamble** para identificar el SKU de cada producto, se
comparan contra los individuales y se elige el modelo final con criterios de negocio. Objetivos del
curso que se cubren:

- **3.5** Mejorar el rendimiento aprovechando las fortalezas de distintos modelos y reduciendo sus
  debilidades (secciones 4 a 6).
- **3.6** Evaluar la calidad de las predicciones en datos no vistos (secciones 6 y 7).

| Instrucción | Dónde |
|---|---|
| A. Ensambles homogéneos y heterogéneos, con optimización de hiperparámetros; stacking y blending con los mejores individuales | Secciones 4 y 5 |
| B. Tabla comparativa ordenada por la métrica principal, con otras métricas y tiempos de entrenamiento; elección del modelo final | Sección 6 |
| C. Gráficas del modelo elegido con su interpretación | Sección 7 |
| Declaración de uso de IA | Al final de la libreta |
""")

md("""
## 1. Contexto: sesión con la empresa sponsor del 28 de octubre y datos de este avance

### 1.1 Recomendaciones de la sesión

En la revisión del Avance 4 con la empresa sponsor (28 de octubre de 2026) se acordó cómo cerrar el pipeline
de la Figura 1 del Avance 4 (foto del anaquel, detector de una sola clase, recortes, embedding,
comparación contra una galería de SKU y planograma). Las recomendaciones que guían este avance son:

1. **Galería de SKU por carpeta.** El nombre de cada carpeta es el SKU y sus fotos son las
   referencias. La identificación se hace comparando el embedding de cada recorte contra la
   galería (similitud coseno), con un umbral para declarar un producto como «desconocido».
   Con galería, **un SKU nuevo se agrega con su carpeta, sin reentrenar**.
2. **Entrenar el embedding con aprendizaje de métrica** en lugar del entrenamiento
   auto-supervisado (InfoNCE) del Avance 4: ancla = foto 1 del SKU X, positivo = foto 2 del
   mismo SKU, negativo = foto de otro SKU. El mismo modelo procesa las tres imágenes y la pérdida
   acerca ancla y positivo y separa ancla y negativo.
3. **Modelo base:** DINOv2 ViT-S/14, con lotes de 2 a 4 fotos por SKU (`MPerClassSampler`),
   distancia coseno sobre vectores normalizados y **destilación** hacia el DINOv2 original para
   no perder lo que ya sabe: pérdida = pérdida de métrica + λ · destilación.
4. **Pérdidas a comparar** (`pytorch-metric-learning`): `TripletMarginLoss` con minado
   semi-difícil, `MultiSimilarityLoss` y `SupConLoss` (la continuación supervisada del InfoNCE).
   Se elige por top-1 en validación.
5. **Fusión con color:** el histograma HSV, ganador del Avance 4, se combina con DINOv2
   (los primeros candidatos por DINOv2 y desempate por color).
6. **Dejar SKU fuera del entrenamiento** para medir el reconocimiento de productos nuevos.
7. **Pendientes de la sesión:** reentrenar el detector con escenas de refrigerador, aplicar el
   pipeline completo en uno o dos refrigeradores y solicitar a la empresa sponsor el planograma esperado para
   señalar faltantes de un SKU concreto.

### 1.2 Datos usados en esta ejecución

> **Esta libreta se ejecutó con datos dummy.** Durante la ventana de entrega no hubo acceso a la
> infraestructura de la empresa sponsor, y por el acuerdo de confidencialidad sus datos no pueden
> salir de ella. **El lunes 2 de noviembre de 2026 la misma libreta se ejecutará con los datos de la
> empresa sponsor, dentro de su infraestructura.** El repositorio se actualizará solo con métricas y
> gráficas agregadas, sin imágenes ni información confidencial.

Como datos dummy se usó un conjunto público con la misma estructura que la galería de SKU: el
*Grocery Store Dataset* (Klasson et al., 2019, licencia MIT). Tiene 5 125 fotos de productos
tomadas con un teléfono en supermercados, **una carpeta por producto** (81 productos finos, de
ellos 31 en empaque, como leche, jugo y yogur, típicos de refrigerador) y una imagen icónica de
catálogo por producto.

- El código no depende del conjunto: para los datos de la empresa sponsor basta con cambiar
  `A5_DATA_DIR` y las listas de partición. Las cifras de esta libreta **no** son resultados sobre
  datos de la empresa; se reportan como evidencia del método.
- Los ensambles del detector (fusión de cajas de FCOS y Faster R-CNN) requieren los modelos
  entrenados con los datos de la empresa y se harán en esa misma ejecución (sección 8).
- La tarea que se evalúa es la del paso 5 del pipeline: dado un recorte de producto, identificar
  su SKU contra la galería.
""")

# ====================================================================== 0. configuración
md("""
## 2. Configuración

Las rutas se leen de variables de entorno: `A5_DATA_DIR` (datos) y `A5_OUT_DIR` (embeddings,
modelos y resultados intermedios, que no se versionan). `A5_PERFIL=prueba` corre una versión
reducida para verificar el código; la entrega usa `completo`. La semilla fija todo componente
estocástico. Los resultados pesados se guardan en disco y se reutilizan si ya existen.
""")
code(r"""
import os, io, json, time, copy, math, random, warnings, pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import cv2
from PIL import Image as PILImage
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

import torch
import torch.nn.functional as F
import sklearn
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GroupKFold
from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
from pytorch_metric_learning import losses, miners, samplers, distances
import importlib.metadata

SEED = 48
def fijar_semilla(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
fijar_semilla()

PERFIL = os.environ.get('A5_PERFIL', 'completo')        # 'prueba' | 'completo'
P = {'prueba':   dict(max_img={'train': 12, 'val': 3, 'test': 6}, epocas=1, trials_rf=2, trials_hgb=1, trials_voto=10),
     'completo': dict(max_img=None, epocas=8, trials_rf=15, trials_hgb=6, trials_voto=80)}[PERFIL]

DATA_DIR = pathlib.Path(os.environ.get('A5_DATA_DIR', '../../datasets_publicos/GroceryStoreDataset/dataset'))
OUT_DIR = pathlib.Path(os.environ.get('A5_OUT_DIR', 'outputs')) / PERFIL
FIG_DIR = pathlib.Path('figuras')
for d in (OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
assert (DATA_DIR / 'train.txt').exists(), 'Definir A5_DATA_DIR con la ubicación de los datos'

dev = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
N_JOBS = int(os.environ.get('A5_N_JOBS', 8))
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams.update({'figure.dpi': 100, 'axes.titleweight': 'bold', 'axes.spines.top': False,
                     'axes.spines.right': False})
# color por tipo de modelo (fijo en todas las gráficas)
COLOR_TIPO = {'Individual': '#8c8c8c', 'Homogéneo': '#2a78d6', 'Heterogéneo': '#eb6834'}

def guardar(fig, nombre):
    fig.savefig(FIG_DIR / nombre, dpi=200, bbox_inches='tight')
    plt.show()

print(f'perfil={PERFIL} | dispositivo={dev} | torch {torch.__version__} | scikit-learn {sklearn.__version__}'
      f' | optuna {optuna.__version__} | pytorch-metric-learning {importlib.metadata.version('pytorch-metric-learning')}')
""")

# ====================================================================== 3. datos
md("""
## 3. Datos, partición y protocolo de evaluación

- **Partición oficial del conjunto** (por foto): entrenamiento, validación y prueba. La prueba se
  usa una sola vez por modelo, al final.
- **Galería:** las fotos de entrenamiento de cada carpeta. Un recorte se identifica con el SKU
  de la carpeta más parecida (máxima similitud coseno contra las fotos de esa carpeta).
- **SKU nuevos:** 10 productos se dejan fuera del entrenamiento del embedding (recomendación 6).
  Sus fotos sí están en la galería, como ocurriría al dar de alta un SKU sin reentrenar.
- **Métrica principal: F1 macro**, la misma del Avance 3. Cada SKU pesa igual en el anaquel y
  un error de identificación produce una alerta falsa de producto mal ubicado. Métricas de apoyo:
  exactitud top-1, top-5 (útil para revisión asistida) y MRR.
""")
code(r"""
clases = pd.read_csv(DATA_DIR / 'classes.csv')
clases.columns = ['clase', 'id', 'familia', 'id_familia', 'iconica', 'descripcion']
C = len(clases)
NOMBRE = dict(zip(clases.id, clases.clase))
GRUPO = {r.id: ('Empaque' if r.iconica.startswith('/iconic-images-and-descriptions/Packages') else
                'Fruta' if '/Fruit/' in r.iconica else 'Verdura') for r in clases.itertuples()}

def leer_lista(nombre):
    df = pd.read_csv(DATA_DIR / f'{nombre}.txt', header=None, names=['ruta', 'y', 'yf'], skipinitialspace=True)
    df['split'] = nombre
    lim = P['max_img']
    if lim:
        df = df.groupby('y', group_keys=False).head(lim[nombre])
    return df

datos = pd.concat([leer_lista(s) for s in ('train', 'val', 'test')], ignore_index=True)
rng = np.random.default_rng(SEED)
NUEVOS = np.sort(rng.choice(C, size=10, replace=False))
datos['grupo'] = datos.y.map(GRUPO)
datos['sku_nuevo'] = datos.y.isin(NUEVOS)

resumen = datos.pivot_table(index='grupo', columns='split', values='ruta', aggfunc='count').fillna(0).astype(int)
resumen['productos'] = datos.groupby('grupo').y.nunique()
resumen.loc['total'] = resumen.sum()
display(resumen[['productos', 'train', 'val', 'test']])
por_clase = datos[datos.split == 'train'].groupby('y').size()
print(f'Fotos de galería por SKU: mínimo {por_clase.min()}, mediana {por_clase.median():.0f}, máximo {por_clase.max()}'
      f' (desbalance {por_clase.max() / por_clase.min():.1f}x)')
print(f'Clases con fotos de validación: {datos[datos.split == "val"].y.nunique()} de {C}')
print('SKU nuevos (fuera del entrenamiento del embedding):', ', '.join(NOMBRE[i] for i in NUEVOS))
""")
interpretacion('datos')

code(r"""
# Carga única de imágenes a 256 x 256 (RGB, uint8). Entrenamiento: recorte aleatorio de 224;
# evaluación: recorte central de 224 (protocolo estándar de los modelos de ImageNet y DINOv2).
LADO = 256
def leer(ruta):
    im = PILImage.open(ruta).convert('RGB').resize((LADO, LADO), PILImage.BILINEAR)
    return np.asarray(im, dtype=np.uint8)

t0 = time.time()
with ThreadPoolExecutor(N_JOBS) as ex:
    IMG = np.stack(list(ex.map(leer, [DATA_DIR / r for r in datos.ruta])))
    ICONO = np.stack(list(ex.map(leer, [DATA_DIR / r.lstrip('/') for r in clases.iconica])))
print(f'{len(IMG)} fotos y {len(ICONO)} imágenes icónicas cargadas en {time.time() - t0:.1f} s')

idx = {s: np.where(datos.split == s)[0] for s in ('train', 'val', 'test')}
y = datos.y.to_numpy()
yG = y[idx['train']]                       # etiquetas de la galería
""")

# ====================================================================== 4. individuales
md("""
## 4. Modelos individuales (fase previa y recomendaciones de la sesión)

| Modelo | Origen | Cómo identifica el SKU |
|---|---|---|
| **Histograma HSV** | Embedding final del Avance 4 | kNN coseno contra la galería |
| **DINOv2 ViT-S/14 sin entrenar** | Finalista profundo del Avance 4 | kNN coseno contra la galería |
| **DINOv2 ViT-S/14 + aprendizaje de métrica** | Recomendación de la sesión (pérdida elegida en validación) | kNN coseno contra la galería |
| **DINOv2 + métrica, solo imagen icónica** | Galería de una sola foto de catálogo por SKU | Coseno contra la imagen icónica |
| **Regresión logística** sobre el embedding entrenado | Algoritmo baseline del Avance 3 | Clasificador por SKU (requiere reentrenar al agregar SKU) |

El histograma HSV usa 16 × 4 × 4 contenedores, raíz cuadrada (distancia de Hellinger) y
normalización L2. En el Avance 4 se aplicaba sobre recortes con margen de 8 %; aquí las fotos ya
son del producto, por lo que se usa la imagen completa.
""")
code(r"""
def hsv_hist(img):
    hsv = cv2.cvtColor(img[16:240, 16:240], cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    h = np.sqrt(h / h.sum())
    return (h / np.linalg.norm(h)).astype(np.float32)

def puntajes_clase(Q, G, yg, n=C):
    # puntaje de cada SKU = máxima similitud coseno contra las fotos de su carpeta (-1 si no hay fotos)
    S = Q @ G.T
    out = np.full((len(Q), n), -1.0, dtype=np.float32)
    for c in np.unique(yg):
        out[:, c] = S[:, yg == c].max(1)
    return out

def metricas(scores, yv):
    pred = scores.argmax(1)
    rango = (scores > scores[np.arange(len(yv)), yv][:, None]).sum(1) + 1
    return {'F1 macro': f1_score(yv, pred, average='macro', labels=np.unique(yv)),
            'Top-1': float((rango == 1).mean()), 'Top-5': float((rango <= 5).mean()),
            'MRR': float((1 / rango).mean())}

TIEMPOS = {}       # segundos de entrenamiento o construcción de cada modelo
MS_IMG = {}        # milisegundos de inferencia por imagen (extracción + comparación)

t0 = time.time()
HSV = np.stack([hsv_hist(im) for im in IMG])
ms_hsv = (time.time() - t0) / len(IMG) * 1e3
TIEMPOS['HSV + kNN'] = ms_hsv * len(idx['train']) / 1e3      # construir la galería
print(f'HSV: {HSV.shape[1]} dimensiones, {ms_hsv:.2f} ms por imagen')
""")

code(r"""
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def a_tensor(lote_uint8, entrenamiento=False):
    x = torch.from_numpy(lote_uint8).to(dev).permute(0, 3, 1, 2).float() / 255
    if entrenamiento:
        # recorte aleatorio de 224, volteo horizontal y brillo leve (el color identifica al producto)
        i, j = np.random.randint(0, LADO - 224 + 1, size=2)
        x = x[:, :, i:i + 224, j:j + 224]
        if np.random.rand() < 0.5:
            x = x.flip(3)
        x = (x * torch.empty(len(x), 1, 1, 1, device=dev).uniform_(0.85, 1.15)).clamp(0, 1)
    else:
        x = x[:, :, 16:240, 16:240]
    return (x - MEAN) / STD

@torch.no_grad()
def embeber(modelo, imgs, lote=128):
    modelo.eval()
    out = [F.normalize(modelo(a_tensor(imgs[i:i + lote])).float(), dim=1).cpu()
           for i in range(0, len(imgs), lote)]
    return torch.cat(out).numpy()

DINO_BASE = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).to(dev).eval()
for p in DINO_BASE.parameters():
    p.requires_grad_(False)

t0 = time.time()
DINO = embeber(DINO_BASE, IMG)
if dev.type == 'mps': torch.mps.synchronize()
ms_dino = (time.time() - t0) / len(IMG) * 1e3
ICO_DINO = embeber(DINO_BASE, ICONO)
TIEMPOS['DINOv2 + kNN'] = ms_dino * len(idx['train']) / 1e3
print(f'DINOv2 ViT-S/14: {DINO.shape[1]} dimensiones, {ms_dino:.2f} ms por imagen en {dev}')
""")

md("""
### 4.1 Aprendizaje de métrica: comparación de pérdidas

Se ajustan los 4 últimos bloques del transformer (los 8 primeros quedan congelados) con AdamW
(tasa 1e-5, decaimiento 0.05) y lotes de 64 fotos con 4 fotos por SKU. A cada pérdida se suma
la destilación λ · (1 − cos(e, e₀)), con λ = 1, donde e₀ es el embedding del DINOv2 original.
Cada época se mide el top-1 en validación contra la galería y se conserva la mejor época.
Los SKU nuevos no participan en el entrenamiento.
""")
code(r"""
def nueva_perdida(nombre):
    cos = distances.CosineSimilarity()
    if nombre == 'Triplet (semi-difícil)':
        return (losses.TripletMarginLoss(margin=0.2, distance=cos),
                miners.TripletMarginMiner(margin=0.2, type_of_triplets='semihard', distance=cos))
    if nombre == 'Multi-similitud':
        return losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5), miners.MultiSimilarityMiner(epsilon=0.1)
    if nombre == 'SupCon':
        return losses.SupConLoss(temperature=0.1), None
    raise ValueError(nombre)

def top1_val(modelo):
    G = embeber(modelo, IMG[idx['train']]); Q = embeber(modelo, IMG[idx['val']])
    return metricas(puntajes_clase(Q, G, yG), y[idx['val']])['Top-1']

def entrenar_metrica(nombre, epocas, lam=1.0, lote=64, m=4):
    ruta = OUT_DIR / f'metrica_{nombre.split()[0].lower()}.pt'
    if ruta.exists():
        return torch.load(ruta, weights_only=False)
    fijar_semilla()
    modelo = copy.deepcopy(DINO_BASE)
    for i, blk in enumerate(modelo.blocks):
        for p in blk.parameters():
            p.requires_grad_(i >= 8)
    for p in modelo.norm.parameters():
        p.requires_grad_(True)
    entrenables = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(entrenables, lr=1e-5, weight_decay=0.05)
    perdida, minero = nueva_perdida(nombre)
    sel = idx['train'][~datos.sku_nuevo.to_numpy()[idx['train']]]
    etiquetas = y[sel]
    muestreador = samplers.MPerClassSampler(etiquetas, m=m, batch_size=lote, num_batches=max(1, len(etiquetas) // lote),
                                             generator=torch.Generator().manual_seed(SEED))
    hist = [{'época': 0, 'pérdida': np.nan, 'top-1 val': top1_val(modelo)}]
    mejor = (hist[0]['top-1 val'], copy.deepcopy(modelo.state_dict()), 0)
    t0 = time.time()
    for ep in range(1, epocas + 1):
        modelo.train()
        acum = []
        for pos in muestreador:                       # cada lote trae m fotos de cada SKU muestreado
            pos = np.asarray(pos)
            x = a_tensor(IMG[sel[pos]], entrenamiento=True)
            yy = torch.as_tensor(etiquetas[pos], device=dev)
            e = F.normalize(modelo(x), dim=1)
            with torch.no_grad():
                e0 = F.normalize(DINO_BASE(x), dim=1)
            pares = minero(e, yy) if minero is not None else None
            l = perdida(e, yy, pares) + lam * (1 - (e * e0).sum(1)).mean()
            opt.zero_grad(); l.backward(); opt.step()
            acum.append(l.item())
        t1 = top1_val(modelo)
        hist.append({'época': ep, 'pérdida': float(np.mean(acum)), 'top-1 val': t1})
        if t1 > mejor[0]:
            mejor = (t1, copy.deepcopy(modelo.state_dict()), ep)
    res = {'nombre': nombre, 'historia': hist, 'mejor_top1': mejor[0], 'mejor_epoca': mejor[2],
           'estado': mejor[1], 'segundos': time.time() - t0}
    torch.save(res, ruta)
    return res

PERDIDAS = ['Triplet (semi-difícil)', 'Multi-similitud', 'SupCon']
corridas = {n: entrenar_metrica(n, P['epocas']) for n in PERDIDAS}
tabla_perdidas = pd.DataFrame([{'pérdida': n, 'top-1 val (sin entrenar)': r['historia'][0]['top-1 val'],
                                'mejor top-1 val': r['mejor_top1'], 'mejor época': r['mejor_epoca'],
                                'segundos': r['segundos']} for n, r in corridas.items()])
display(tabla_perdidas.sort_values('mejor top-1 val', ascending=False).round(3))
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for n, r in corridas.items():
    h = pd.DataFrame(r['historia'])
    ax[0].plot(h['época'], h['top-1 val'], marker='o', label=n)
    ax[1].plot(h['época'][1:], h['pérdida'][1:], marker='o', label=n)
ax[0].set(title='Top-1 en validación por época', xlabel='época', ylabel='top-1')
ax[1].set(title='Pérdida de entrenamiento (métrica + destilación)', xlabel='época', ylabel='pérdida')
ax[0].legend(); plt.tight_layout()
guardar(fig, '01_aprendizaje_metrica.png')
""")
interpretacion('perdidas')

code(r"""
MEJOR_PERDIDA = max(corridas, key=lambda n: corridas[n]['mejor_top1'])
DINO_M_MODELO = copy.deepcopy(DINO_BASE)
DINO_M_MODELO.load_state_dict(corridas[MEJOR_PERDIDA]['estado'])
DINOM = embeber(DINO_M_MODELO, IMG)
ICO_DINOM = embeber(DINO_M_MODELO, ICONO)
TIEMPOS['DINOv2 + métrica + kNN'] = corridas[MEJOR_PERDIDA]['segundos']
print(f'Pérdida elegida: {MEJOR_PERDIDA} (época {corridas[MEJOR_PERDIDA]["mejor_epoca"]})')

# Regresión logística sobre el embedding entrenado (algoritmo baseline del Avance 3)
t0 = time.time()
LR = LogisticRegression(C=10, max_iter=3000).fit(DINOM[idx['train']], yG)
TIEMPOS['Regresión logística (DINOv2 + métrica)'] = time.time() - t0

def todos_los_puntajes(Q_ids, G_ids=None, y_gal=None):
    # puntajes por SKU de cada modelo base para las consultas Q_ids contra la galería G_ids
    G_ids = idx['train'] if G_ids is None else G_ids
    y_gal = y[G_ids] if y_gal is None else y_gal
    return {'HSV': puntajes_clase(HSV[Q_ids], HSV[G_ids], y_gal),
            'DINOv2': puntajes_clase(DINO[Q_ids], DINO[G_ids], y_gal),
            'DINOv2+M': puntajes_clase(DINOM[Q_ids], DINOM[G_ids], y_gal),
            'Icónica': DINOM[Q_ids] @ ICO_DINOM.T}

S_TEST = todos_los_puntajes(idx['test'])
S_VAL = todos_los_puntajes(idx['val'])
yT, yV = y[idx['test']], y[idx['val']]
t0 = time.time(); P_LR_TEST = LR.predict_proba(DINOM[idx['test']]); ms_lr = (time.time() - t0) / len(yT) * 1e3

INDIVIDUALES = {'HSV + kNN': ('HSV', ms_hsv), 'DINOv2 + kNN': ('DINOv2', ms_dino),
                'DINOv2 + métrica + kNN': ('DINOv2+M', ms_dino),
                'DINOv2 + métrica, imagen icónica': ('Icónica', ms_dino)}
TIEMPOS['DINOv2 + métrica, imagen icónica'] = TIEMPOS['DINOv2 + métrica + kNN']
resultados = []
for nombre, (k, ms) in INDIVIDUALES.items():
    t0 = time.time(); _ = S_TEST[k].argmax(1); ms_cmp = (time.time() - t0) / len(yT) * 1e3
    MS_IMG[nombre] = ms + ms_cmp
    resultados.append({'modelo': nombre, 'tipo': 'Individual', **metricas(S_TEST[k], yT)})
MS_IMG['Regresión logística (DINOv2 + métrica)'] = ms_dino + ms_lr
resultados.append({'modelo': 'Regresión logística (DINOv2 + métrica)', 'tipo': 'Individual', **metricas(P_LR_TEST, yT)})
display(pd.DataFrame(resultados).set_index('modelo').round(3))
""")
interpretacion('individuales')

# ====================================================================== 5. ensambles
md("""
## 5. Modelos de ensamble

| # | Ensamble | Estrategia | Modelos base | Optimización |
|---|---|---|---|---|
| E1 | **Bosque aleatorio** (bagging) | Homogénea | Árboles de decisión sobre [HSV, DINOv2 + métrica] | Optuna, F1 macro en validación |
| E2 | **Gradient boosting** (`HistGradientBoosting`) | Homogénea | Árboles secuenciales sobre PCA de [HSV, DINOv2 + métrica] | Optuna, F1 macro en validación |
| E3 | **Votación ponderada** (*soft voting*) | Heterogénea | HSV, DINOv2, DINOv2 + métrica, icónica | Optuna sobre los pesos, F1 macro en validación |
| E4 | **Stacking** | Heterogénea | Los mismos cuatro, con un meta-modelo de regresión logística | Meta-modelo entrenado con las predicciones de validación; rejilla sobre C con validación cruzada agrupada por consulta |
| E5 | **Blending** (re-ordenamiento) | Heterogénea | Top-5 de DINOv2 + métrica re-ordenado con HSV | Peso α ajustado en el conjunto de validación |

Los ensambles heterogéneos usan los mejores modelos individuales: el histograma HSV (final del
Avance 4), DINOv2 (finalista profundo del Avance 4) y DINOv2 con aprendizaje de métrica (mejor
individual de la sección 4).

**¿Dónde se ajustan los ensambles?** La primera opción era usar puntajes fuera de partición
dentro de entrenamiento: dividir la galería en 5 particiones y comparar cada foto solo contra las
otras cuatro. La celda siguiente muestra que esa estimación **tiene fuga**: HSV obtiene un top-1
fuera de partición cercano a 0.95 y en prueba de 0.42. Las fotos de cada producto se tomaron en
ráfagas, y dentro de entrenamiento cada foto tiene "gemelas" casi idénticas en las otras
particiones. Lo mismo ocurre en las cuentas de la empresa sponsor, donde las capturas de una carpeta son de la
misma sesión (Avance 3).

Por eso **todos los hiperparámetros, pesos y meta-modelos de los ensambles se ajustan contra el
conjunto de validación**, que es de otra sesión de fotos. Los modelos base se entrenan solo con
entrenamiento y la prueba se usa una sola vez. Es la misma razón por la que el Avance 3 partió por
bloques temporales.
""")
code(r"""
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
tr = idx['train']
OOF = {k: np.zeros((len(tr), C), dtype=np.float32) for k in ('HSV', 'DINOv2', 'DINOv2+M', 'Icónica')}
for a, b in skf.split(tr, yG):
    s = todos_los_puntajes(tr[b], tr[a], yG[a])
    for k in OOF:
        OOF[k][b] = s[k]
diag = pd.DataFrame({'top-1 fuera de partición (entrenamiento)': {k: metricas(v, yG)['Top-1'] for k, v in OOF.items()},
                     'top-1 en validación': {k: metricas(S_VAL[k], yV)['Top-1'] for k in OOF},
                     'top-1 en prueba': {k: metricas(S_TEST[k], yT)['Top-1'] for k in OOF}})
display(diag.round(3))

def z_filas(s):
    # estandariza los puntajes de cada consulta para que las escalas de los modelos sean comparables
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)

BASES = ['HSV', 'DINOv2', 'DINOv2+M', 'Icónica']
Z_OOF = {k: z_filas(OOF[k]) for k in BASES}
Z_TEST = {k: z_filas(S_TEST[k]) for k in BASES}
Z_VAL = {k: z_filas(S_VAL[k]) for k in BASES}
""")

md("### 5.1 E1. Bosque aleatorio (bagging, homogéneo)")
code(r"""
X_TR = np.hstack([HSV[tr], DINOM[tr]]); X_TE = np.hstack([HSV[idx['test']], DINOM[idx['test']]])
X_VA = np.hstack([HSV[idx['val']], DINOM[idx['val']]])

def objetivo_rf(trial):
    rf = RandomForestClassifier(n_estimators=trial.suggest_int('n_estimators', 200, 600, step=100),
                                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.05, 0.1]),
                                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 4),
                                class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
                                n_jobs=N_JOBS, random_state=SEED)
    return metricas(rf.fit(X_TR, yG).predict_proba(X_VA), yV)['F1 macro']

t0 = time.time()
est_rf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
est_rf.optimize(objetivo_rf, n_trials=P['trials_rf'])
RF = RandomForestClassifier(**est_rf.best_params, n_jobs=N_JOBS, random_state=SEED).fit(X_TR, yG)
TIEMPOS['E1 Bosque aleatorio'] = time.time() - t0
t0 = time.time(); P_RF = RF.predict_proba(X_TE); MS_IMG['E1 Bosque aleatorio'] = ms_hsv + ms_dino + (time.time() - t0) / len(yT) * 1e3
print('mejores hiperparámetros:', est_rf.best_params, f'| F1 macro en validación {est_rf.best_value:.3f}')
resultados.append({'modelo': 'E1 Bosque aleatorio', 'tipo': 'Homogéneo', **metricas(P_RF, yT)})
""")

md("### 5.2 E2. Gradient boosting (homogéneo)")
code(r"""
t0 = time.time()
pca = PCA(n_components=48, random_state=SEED).fit(X_TR)
XP_TR, XP_TE = pca.transform(X_TR), pca.transform(X_TE)
XP_VA = pca.transform(X_VA)

def hgb(params):
    return HistGradientBoostingClassifier(max_iter=200, early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=10, random_state=SEED, **params)

def objetivo_hgb(trial):
    params = dict(learning_rate=trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
                  max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 7, 31),
                  l2_regularization=trial.suggest_float('l2_regularization', 1e-3, 10, log=True))
    m = hgb(params).fit(XP_TR, yG)
    return f1_score(yV, m.predict(XP_VA), average='macro', labels=np.unique(yV))

est_hgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
est_hgb.optimize(objetivo_hgb, n_trials=P['trials_hgb'])
HGB = hgb(est_hgb.best_params).fit(XP_TR, yG)
TIEMPOS['E2 Gradient boosting'] = time.time() - t0
t0 = time.time(); P_HGB = HGB.predict_proba(XP_TE)
P_HGB_FULL = np.zeros((len(yT), C)); P_HGB_FULL[:, HGB.classes_] = P_HGB
MS_IMG['E2 Gradient boosting'] = ms_hsv + ms_dino + (time.time() - t0) / len(yT) * 1e3
print('mejores hiperparámetros:', {k: round(v, 4) for k, v in est_hgb.best_params.items()},
      f'| F1 macro en validación {est_hgb.best_value:.3f} | iteraciones {HGB.n_iter_}')
resultados.append({'modelo': 'E2 Gradient boosting', 'tipo': 'Homogéneo', **metricas(P_HGB_FULL, yT)})
""")

md("### 5.3 E3. Votación ponderada (heterogéneo)")
code(r"""
def combinar(Z, w):
    return sum(w[k] * Z[k] for k in BASES)

def objetivo_voto(trial):
    w = {k: trial.suggest_float(k, 0.0, 1.0) for k in BASES}
    return metricas(combinar(Z_VAL, w), yV)['F1 macro']

t0 = time.time()
est_voto = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
est_voto.enqueue_trial({k: 1.0 for k in BASES})          # votación simple como punto de partida
est_voto.optimize(objetivo_voto, n_trials=P['trials_voto'])
W_VOTO = est_voto.best_params
TIEMPOS['E3 Votación ponderada'] = time.time() - t0
t0 = time.time(); S_VOTO = combinar(Z_TEST, W_VOTO)
MS_IMG['E3 Votación ponderada'] = ms_hsv + ms_dino + (time.time() - t0) / len(yT) * 1e3 + MS_IMG['DINOv2 + kNN']
print('pesos:', {k: round(v, 3) for k, v in W_VOTO.items()}, f'| F1 macro en validación {est_voto.best_value:.3f}')
resultados.append({'modelo': 'E3 Votación ponderada', 'tipo': 'Heterogéneo', **metricas(S_VOTO, yT)})
""")

md("""
### 5.4 E4. Stacking (heterogéneo)

El meta-modelo no aprende un peso por SKU: recibe, para cada par (recorte, SKU candidato), los
puntajes de los cuatro modelos base, estandarizados y como distancia al mejor candidato, y
estima la probabilidad de que el candidato sea el correcto. El SKU elegido es el de mayor
probabilidad. Como el meta-modelo no depende de qué SKU son, **un SKU nuevo se agrega a la galería
sin reentrenar**, lo mismo que con los modelos de galería.
""")
code(r"""
def rasgos_pares(Z):
    cols = []
    for k in BASES:
        cols += [Z[k], Z[k] - Z[k].max(1, keepdims=True)]
    return np.stack(cols, axis=-1)                     # (consultas, SKU, rasgos)

NOMBRES_RASGO = [f'{k} {t}' for k in BASES for t in ('puntaje z', 'distancia al mejor')]
# el meta-modelo aprende de las predicciones de los modelos base sobre validación (fotos que no vieron)
R_META = rasgos_pares(Z_VAL)
X_META = R_META.reshape(-1, R_META.shape[-1])
Y_META = (np.arange(C)[None, :] == yV[:, None]).reshape(-1).astype(int)
GRUPOS = np.repeat(np.arange(len(yV)), C)

def meta(Cr):
    return LogisticRegression(C=Cr, max_iter=2000)

def f1_meta(Cr, folds=3):
    f1s = []
    for a, b in GroupKFold(folds).split(X_META, Y_META, GRUPOS):
        m = meta(Cr).fit(X_META[a], Y_META[a])
        q = np.unique(GRUPOS[b])
        f1s.append(metricas(m.decision_function(R_META[q].reshape(-1, X_META.shape[1])).reshape(len(q), C), yV[q])['F1 macro'])
    return float(np.mean(f1s))

t0 = time.time()
rejilla_C = {Cr: f1_meta(Cr) for Cr in (0.01, 0.1, 1.0, 10.0)}
C_META = max(rejilla_C, key=rejilla_C.get)
META = meta(C_META).fit(X_META, Y_META)
TIEMPOS['E4 Stacking'] = time.time() - t0

def stacking(Z):
    R = rasgos_pares(Z)
    return META.decision_function(R.reshape(-1, R.shape[-1])).reshape(R.shape[0], R.shape[1])

t0 = time.time(); S_STACK = stacking(Z_TEST)
MS_IMG['E4 Stacking'] = ms_hsv + ms_dino + (time.time() - t0) / len(yT) * 1e3 + MS_IMG['DINOv2 + kNN']
print('F1 por valor de C:', {k: round(v, 3) for k, v in rejilla_C.items()}, '| C elegido:', C_META)
resultados.append({'modelo': 'E4 Stacking', 'tipo': 'Heterogéneo', **metricas(S_STACK, yT)})
""")

md("""
### 5.5 E5. Blending (heterogéneo)

Recomendación 5 de la sesión. DINOv2 con métrica propone los 5 SKU más parecidos y se reordenan
con la mezcla α · DINOv2 + métrica + (1 − α) · HSV. El peso α se elige en el **conjunto de
validación**, que no se usó para entrenar los modelos base. Por eso es un blending y no un stacking.
""")
code(r"""
def blending(Z, a, k=5):
    base = Z['DINOv2+M']
    top = np.argsort(-base, axis=1)[:, :k]
    mezcla = a * base + (1 - a) * Z['HSV']
    out = np.full_like(base, -1e9)
    filas = np.arange(len(base))[:, None]
    out[filas, top] = mezcla[filas, top] + 1e3          # los candidatos del top-k van primero
    return out

t0 = time.time()
curva_alfa = {round(a, 2): metricas(blending(Z_VAL, a), yV)['F1 macro'] for a in np.linspace(0, 1, 21)}
ALFA = max(curva_alfa, key=curva_alfa.get)
TIEMPOS['E5 Blending'] = time.time() - t0
t0 = time.time(); S_BLEND = blending(Z_TEST, ALFA)
MS_IMG['E5 Blending'] = ms_hsv + ms_dino + (time.time() - t0) / len(yT) * 1e3
print(f'alfa elegido en validación: {ALFA} (F1 val {curva_alfa[ALFA]:.3f}; solo DINOv2+M: {curva_alfa[1.0]:.3f})')
resultados.append({'modelo': 'E5 Blending', 'tipo': 'Heterogéneo', **metricas(S_BLEND, yT)})
""")

# ====================================================================== 6. comparación
md("""
## 6. Tabla comparativa y elección del modelo final

Todos los modelos se evaluaron una sola vez en la partición de prueba. La tabla está ordenada por
el F1 macro y muestra además top-1, top-5, MRR, el tiempo de entrenamiento y la latencia de
inferencia por imagen.

- **Tiempo de entrenamiento:** lo que tarda cada modelo en construirse, incluida su
  optimización de hiperparámetros. En los ensambles no se suma el entrenamiento de sus bases;
  la columna «depende de» indica qué bases usa.
- **Latencia:** se midió en la GPU de una Mac (MPS); en el Jetson se medirá con FP16.

La columna «alta de SKU» indica si agregar un SKU requiere reentrenar.
""")
code(r"""
DEPENDE = {'E1 Bosque aleatorio': 'HSV, DINOv2+M', 'E2 Gradient boosting': 'HSV, DINOv2+M',
           'E3 Votación ponderada': 'HSV, DINOv2, DINOv2+M', 'E4 Stacking': 'HSV, DINOv2, DINOv2+M',
           'E5 Blending': 'HSV, DINOv2+M', 'DINOv2 + métrica, imagen icónica': 'DINOv2+M',
           'Regresión logística (DINOv2 + métrica)': 'DINOv2+M'}
SIN_REENTRENAR = {'HSV + kNN', 'DINOv2 + kNN', 'DINOv2 + métrica + kNN', 'DINOv2 + métrica, imagen icónica',
                  'E3 Votación ponderada', 'E4 Stacking', 'E5 Blending'}
comp = pd.DataFrame(resultados)
comp['entrenamiento (s)'] = comp.modelo.map(TIEMPOS)
comp['inferencia (ms/img)'] = comp.modelo.map(MS_IMG)
comp['depende de'] = comp.modelo.map(DEPENDE).fillna('-')
comp['alta de SKU'] = np.where(comp.modelo.isin(SIN_REENTRENAR), 'sin reentrenar', 'reentrenar')
comp = comp.sort_values('F1 macro', ascending=False).reset_index(drop=True)
comp.index += 1
comp.to_csv(OUT_DIR / 'comparacion.csv')
display(comp.round(3))
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={'width_ratios': [1.5, 1]})
c = comp.iloc[::-1]
ax[0].barh([f'{i}. {m}' for i, m in zip(c.index, c.modelo)], c['F1 macro'], color=c.tipo.map(COLOR_TIPO))
for yy, v in enumerate(c['F1 macro']):
    ax[0].text(v + 0.004, yy, f'{v:.3f}', va='center', fontsize=9)
ax[0].axvline(0.85, ls='--', color='#333333', lw=1)
ax[0].text(0.848, len(c) - 0.45, 'meta 0.85', fontsize=8, ha='right')
ax[0].set(title='F1 macro en prueba (número = posición en la tabla)', xlim=(max(0, c['F1 macro'].min() - 0.1), 1.02), xlabel='F1 macro')
ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in COLOR_TIPO.values()], labels=list(COLOR_TIPO),
             loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3)
for t, g in comp.groupby('tipo'):
    ax[1].scatter(g['inferencia (ms/img)'], g['F1 macro'], s=60, color=COLOR_TIPO[t], label=t)
for i, r in comp.iterrows():
    ax[1].annotate(str(i), (r['inferencia (ms/img)'], r['F1 macro']), fontsize=8, xytext=(5, -3), textcoords='offset points')
ax[1].set(title='Calidad contra latencia', xlabel='inferencia (ms por imagen)', ylabel='F1 macro')
plt.tight_layout()
guardar(fig, '02_comparacion_modelos.png')
""")
interpretacion('comparacion')

code(r"""
# Modelo final: se elige en la celda de interpretación con criterios de negocio; aquí se fija para las gráficas
# Regla de negocio: entre los ensambles que permiten dar de alta un SKU sin reentrenar, el de mayor F1 macro
cand = comp[comp.modelo.str.startswith('E') & (comp['alta de SKU'] == 'sin reentrenar')]
FINAL = os.environ.get('A5_FINAL', cand.iloc[0].modelo)
PUNTAJE_FINAL = {'E3 Votación ponderada': lambda Z: combinar(Z, W_VOTO), 'E4 Stacking': stacking,
                 'E5 Blending': lambda Z: blending(Z, ALFA)}[FINAL]
S_FINAL = PUNTAJE_FINAL(Z_TEST)
print('Modelo final:', FINAL, {k: round(v, 3) for k, v in metricas(S_FINAL, yT).items()})
""")

# ====================================================================== 7. gráficas
md("""
## 7. Análisis del modelo final

### 7.1 SKU nuevos: productos que el embedding no vio al entrenar

Los 10 SKU nuevos están en la galería pero no participaron en el entrenamiento del embedding. Si
su top-1 se mantiene cerca del de los SKU vistos, dar de alta un producto con solo agregar su
carpeta es viable.
""")
code(r"""
nuevo_T = np.isin(yT, NUEVOS)
filas = []
for nombre, s in [('DINOv2 + kNN', S_TEST['DINOv2']), ('DINOv2 + métrica + kNN', S_TEST['DINOv2+M']), (FINAL, S_FINAL)]:
    for etiqueta, m in [('SKU vistos', ~nuevo_T), ('SKU nuevos', nuevo_T)]:
        filas.append({'modelo': nombre, 'grupo': etiqueta, 'fotos': int(m.sum()),
                      **{k: v for k, v in metricas(s[m], yT[m]).items() if k in ('Top-1', 'Top-5')}})
display(pd.DataFrame(filas).pivot(index='modelo', columns='grupo', values=['Top-1', 'Top-5']).round(3))
""")
interpretacion('nuevos')

md("""
### 7.2 Matriz de confusión de los SKU más confundidos

La matriz completa tiene 81 × 81 celdas. Se muestran los 15 SKU que participan en más errores,
con las filas normalizadas (proporción de fotos de cada SKU real).
""")
code(r"""
pred = S_FINAL.argmax(1)
err = pd.Series(np.concatenate([yT[pred != yT], pred[pred != yT]])).value_counts()
top = err.index[:15].to_numpy()
m = np.isin(yT, top)
cm = pd.crosstab(pd.Series(yT[m]).map(NOMBRE), pd.Series(pred[m]).map(NOMBRE), normalize='index')
orden = [NOMBRE[i] for i in top]
cm = cm.reindex(index=orden, columns=[c for c in orden if c in cm.columns] + [c for c in cm.columns if c not in orden]).fillna(0)
cm = cm[[c for c in orden if c in cm.columns]]
fig, ax = plt.subplots(figsize=(9.5, 7.5))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1, cbar_kws={'label': 'proporción'}, ax=ax,
            annot_kws={'fontsize': 7})
ax.set(title=f'Matriz de confusión, 15 SKU con más errores ({FINAL})', xlabel='SKU predicho', ylabel='SKU real')
plt.xticks(rotation=60, ha='right', fontsize=8); plt.yticks(fontsize=8); plt.tight_layout()
guardar(fig, '03_matriz_confusion.png')
pares = (pd.DataFrame({'real': yT[pred != yT], 'predicho': pred[pred != yT]})
         .value_counts().head(8).rename('errores').reset_index())
pares[['real', 'predicho']] = pares[['real', 'predicho']].apply(lambda s: s.map(NOMBRE))
display(pares)
""")
interpretacion('confusion')

md("""
### 7.3 Detección de productos desconocidos: curvas ROC y precisión-recall

El planograma debe distinguir un producto que no está en la galería (por ejemplo, de otro
proveedor) de uno conocido. Para medirlo, se retiran de la galería los 10 SKU nuevos y sus fotos
de prueba se tratan como **desconocidas**. Si el puntaje del mejor candidato queda bajo un
umbral, el producto se declara desconocido. Se evalúan las curvas ROC y precisión-recall de
esa decisión, con «desconocido» como clase positiva.
""")
code(r"""
G_conocida = idx['train'][~np.isin(yG, NUEVOS)]
S_ab = todos_los_puntajes(idx['test'], G_conocida, y[G_conocida])
for k in ('Icónica',):                                  # la galería icónica también excluye a los SKU nuevos
    S_ab[k][:, NUEVOS] = -1.0
def z_filas_presentes(s):
    # igual que z_filas, pero ignora los SKU retirados de la galería (puntaje -1) y los deja al final
    v = np.where(s < -0.5, np.nan, s)
    z = (v - np.nanmean(v, 1, keepdims=True)) / (np.nanstd(v, 1, keepdims=True) + 1e-9)
    return np.nan_to_num(z, nan=-10.0)
Z_ab = {k: z_filas_presentes(S_ab[k]) for k in BASES}
desconocido = np.isin(yT, NUEVOS).astype(int)

curvas = {}
for nombre, conf in [('DINOv2 + métrica (similitud máxima)', S_ab['DINOv2+M'].max(1)),
                     ('HSV (similitud máxima)', S_ab['HSV'].max(1)),
                     (f'{FINAL} (puntaje máximo)', PUNTAJE_FINAL(Z_ab).max(1))]:
    fpr, tpr, _ = roc_curve(desconocido, -conf)
    pr, rc, _ = precision_recall_curve(desconocido, -conf)
    curvas[nombre] = dict(fpr=fpr, tpr=tpr, pr=pr, rc=rc, auc=auc(fpr, tpr), ap=average_precision_score(desconocido, -conf))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for n, c in curvas.items():
    ax[0].plot(c['fpr'], c['tpr'], label=f"{n}: AUC {c['auc']:.3f}")
    ax[1].plot(c['rc'], c['pr'], label=f"{n}: AP {c['ap']:.3f}")
ax[0].plot([0, 1], [0, 1], ls=':', color='#888888')
ax[1].axhline(desconocido.mean(), ls=':', color='#888888', label=f'azar ({desconocido.mean():.3f})')
ax[0].set(title='ROC: detectar SKU desconocido', xlabel='tasa de falsos positivos (conocidos marcados)', ylabel='tasa de verdaderos positivos')
ax[1].set(title='Precisión-recall: detectar SKU desconocido', xlabel='recall', ylabel='precisión')
ax[0].legend(fontsize=8, loc='lower right'); ax[1].legend(fontsize=8)
plt.tight_layout()
guardar(fig, '04_roc_pr_desconocidos.png')
""")
interpretacion('roc_pr')

md("""
### 7.4 Separación de puntajes: aciertos contra errores

Distribución del puntaje del candidato elegido cuando el modelo acierta y cuando se equivoca
(SKU conocidos, galería completa). Un buen umbral deja la mayoría de los errores por debajo y
casi todos los aciertos por encima. Las predicciones bajo el umbral se envían a revisión humana
en lugar de generar una alerta automática.
""")
code(r"""
conf = S_FINAL.max(1)
acierto = S_FINAL.argmax(1) == yT
umbrales = np.quantile(conf, np.linspace(0, 0.5, 51))
tabla_umbral = pd.DataFrame([{'umbral': u, 'cobertura': (conf >= u).mean(),
                              'exactitud en lo automático': acierto[conf >= u].mean()} for u in umbrales])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
sns.histplot(x=conf, hue=np.where(acierto, 'acierto', 'error'), stat='density', common_norm=False,
             bins=40, ax=ax[0], palette={'acierto': '#2a78d6', 'error': '#d62728'})
ax[0].set(title=f'Puntaje del candidato elegido ({FINAL})', xlabel='puntaje', ylabel='densidad')
ax[1].plot(tabla_umbral['cobertura'], tabla_umbral['exactitud en lo automático'], marker='.')
ax[1].set(title='Exactitud contra cobertura al subir el umbral', xlabel='proporción decidida automáticamente',
          ylabel='exactitud de lo decidido')
ax[1].invert_xaxis()
plt.tight_layout()
guardar(fig, '05_puntajes_umbral.png')
display(tabla_umbral.iloc[[0, 5, 10, 20, 30]].round(3))
""")
interpretacion('puntajes')

md("""
### 7.5 Importancia de cada modelo base

Se mide de dos formas. Los **coeficientes** del meta-modelo indican cuánto pesa cada rasgo,
porque los rasgos están estandarizados. La **ablación** reentrena el stacking sin un modelo base
y mide cuánto cae el F1 macro en prueba.
""")
code(r"""
coef = pd.Series(META.coef_[0], index=NOMBRES_RASGO).sort_values()
abl = []
for quitar in BASES:
    usar = [k for k in BASES if k != quitar]
    cols = [i for i, n in enumerate(NOMBRES_RASGO) if n.split(' ')[0] in usar]
    m = meta(C_META).fit(X_META[:, cols], Y_META)
    R = rasgos_pares(Z_TEST)[..., cols]
    s = m.decision_function(R.reshape(-1, len(cols))).reshape(len(yT), C)
    abl.append({'modelo base retirado': quitar, 'caída de F1 macro': metricas(S_STACK, yT)['F1 macro'] - metricas(s, yT)['F1 macro']})
abl = pd.DataFrame(abl).sort_values('caída de F1 macro')
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].barh(coef.index, coef.values, color=np.where(coef.values >= 0, '#2a78d6', '#d62728'))
ax[0].set(title='Coeficientes del meta-modelo (stacking)', xlabel='coeficiente')
ax[1].barh(abl['modelo base retirado'], abl['caída de F1 macro'], color='#eb6834')
ax[1].set(title='Ablación: caída de F1 macro al retirar cada base', xlabel='caída de F1 macro en prueba')
plt.tight_layout()
guardar(fig, '06_importancia_bases.png')
""")
interpretacion('importancia')

md("""
### 7.6 Desempeño por SKU

F1 de cada SKU con el modelo final, coloreado por grupo de producto. Los productos en empaque
son los más cercanos a los productos de refrigerador de la empresa sponsor.
""")
code(r"""
from sklearn.metrics import precision_recall_fscore_support
_, _, f1c, sop = precision_recall_fscore_support(yT, S_FINAL.argmax(1), labels=np.arange(C), zero_division=0)
porsku = pd.DataFrame({'SKU': [NOMBRE[i] for i in range(C)], 'grupo': [GRUPO[i] for i in range(C)],
                       'F1': f1c, 'fotos prueba': sop, 'nuevo': np.isin(np.arange(C), NUEVOS)})
porsku = porsku[porsku['fotos prueba'] > 0].sort_values('F1').reset_index(drop=True)
COLOR_GRUPO = {'Empaque': '#2a78d6', 'Fruta': '#eb6834', 'Verdura': '#1baf7a'}
fig, ax = plt.subplots(figsize=(13, 3.8))
ax.bar(range(len(porsku)), porsku.F1, color=porsku.grupo.map(COLOR_GRUPO), width=0.85)
for i in np.where(porsku.nuevo)[0]:
    ax.text(i, porsku.F1[i] + 0.01, '*', ha='center', fontsize=12)
ax.set(title=f'F1 por SKU ({FINAL}); * = SKU nuevo', xlabel='SKU (ordenados por F1)', ylabel='F1', xticks=[])
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in COLOR_GRUPO.values()], labels=list(COLOR_GRUPO), loc='lower right')
plt.tight_layout()
guardar(fig, '07_f1_por_sku.png')
display(porsku.groupby('grupo').F1.agg(['count', 'mean', 'min']).round(3))
display(porsku.head(6).round(3))
""")
interpretacion('porsku')

# ====================================================================== 8. siguiente ejecución
md("""
## 8. Ejecución del 2 de noviembre de 2026 con los datos de la empresa sponsor

El código de esta libreta corre sin cambios sobre la galería de la empresa sponsor: cada carpeta es un SKU y
se reemplaza la partición oficial por una partición por foto dentro de cada carpeta. Al
recuperar el acceso al Jetson y al servidor, faltan cuatro tareas:

1. Copiar al Jetson las carpetas de SKU y las escenas de refrigerador, sin que salgan de la
   infraestructura de la empresa sponsor, y ejecutar la libreta con `A5_DATA_DIR` apuntando a ellas.
2. **Ensambles del detector:** fusión ponderada de cajas (WBF) de FCOS y Faster R-CNN v2, que es
   un ensamble heterogéneo, y FCOS con aumentos en prueba (volteo y multiescala), que es
   homogéneo. Se ajustarán los pesos y los umbrales de IoU en validación con el protocolo del
   Avance 4 (mAP@[.5:.95], recall y error de conteo).
3. Ejecutar el pipeline completo en uno o dos refrigeradores (detección, SKU, hilera y posición,
   huecos), con resultados solo agregados conforme a `POLITICA_DATOS.md`.
4. Solicitar a la empresa sponsor el planograma esperado para señalar faltantes de un SKU concreto, y la
   correspondencia de las carpetas `cluster_NNN` con productos.
""")

md("""
## 9. Conclusiones (CRISP-ML(Q): ingeniería del modelo)
""")
interpretacion('conclusiones')

# ====================================================================== referencias
md("""
## Referencias

Breiman, L. (1996). Bagging predictors. *Machine Learning, 24*(2), 123–140.
https://doi.org/10.1007/BF00058655

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
https://doi.org/10.1023/A:1010933404324

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of
Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Akiba, T., Sano, S., Yanase, T., Ohta, T., y Koyama, M. (2019). Optuna: A next-generation
hyperparameter optimization framework. En *Proceedings of the 25th ACM SIGKDD International Conference
on Knowledge Discovery & Data Mining* (pp. 2623–2631). ACM. https://doi.org/10.1145/3292500.3330701

Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., y
Krishnan, D. (2020). Supervised contrastive learning. En *Advances in Neural Information Processing
Systems 33* (pp. 18661–18673).

Klasson, M., Zhang, C., y Kjellström, H. (2019). A hierarchical grocery store image dataset with
visual and semantic labels. En *2019 IEEE Winter Conference on Applications of Computer Vision* (pp.
491–500). IEEE. https://doi.org/10.1109/WACV.2019.00058

Musgrave, K., Belongie, S., y Lim, S.-N. (2020). PyTorch metric learning. arXiv.
https://doi.org/10.48550/arXiv.2008.09164

Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without
supervision. *Transactions on Machine Learning Research*.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning
Research, 12*, 2825–2830.

Schroff, F., Kalenichenko, D., y Philbin, J. (2015). FaceNet: A unified embedding for face recognition
and clustering. En *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp.
815–823). IEEE. https://doi.org/10.1109/CVPR.2015.7298682

Singh, A. (2023, 14 de julio). *Comprehensive guide to ensemble learning (with Python codes)*.
Analytics Vidhya. https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/

Solovyev, R., Wang, W., y Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different
object detection models. *Image and Vision Computing, 107*, 104117.
https://doi.org/10.1016/j.imavis.2021.104117

Swain, M. J., y Ballard, D. H. (1991). Color indexing. *International Journal of Computer Vision,
7*(1), 11–32. https://doi.org/10.1007/BF00130487

VanderPlas, J. (2022). *Python data science handbook* (2a ed.). O'Reilly Media.

Visengeriyeva, L., Kammer, A., Bär, I., Kniesz, A., y Plöd, M. (2023). *CRISP-ML(Q). The ML lifecycle
process*. MLOps. INNOQ. https://ml-ops.org/content/crisp-ml

Wang, X., Han, X., Huang, W., Dong, D., y Scott, M. R. (2019). Multi-similarity loss with general pair
weighting for deep metric learning. En *Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition* (pp. 5022–5030). IEEE.

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241–259.
https://doi.org/10.1016/S0893-6080(05)80023-1
""")

interpretacion('declaracion_ia')


def a_notebook():
    cells = []
    for i, (tipo, texto) in enumerate(celdas):
        c = {"cell_type": tipo, "id": f"celda-{i:03d}", "metadata": {},
             "source": texto.splitlines(keepends=True)}
        if tipo == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    DESTINO.write_text(json.dumps(a_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{DESTINO.relative_to(RAIZ)}: {len(celdas)} celdas")
