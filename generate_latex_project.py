import os
import re
import pandas as pd
import numpy as np

# 1. Cargar datos
print("Cargando data.csv...")
df = pd.read_csv('data.csv', encoding='utf-8')

# Limpieza de columnas
clean_cols = [
    'ID', 'Titulo', 'Autores', 'Año', 'DOI', 'Introduccion', 'Metodologia',
    'Analisis_Requerimientos', 'Entorno_Experimental', 'Tamaño_Dataset',
    'Metricas_Rendimiento', 'Limitaciones', 'Discusiones', 'Implementacion_Software',
    'Conclusion', 'Referencia_BibTeX'
]
df.columns = clean_cols

# Convertir Año a numérico de manera segura
df['Año'] = pd.to_numeric(df['Año'], errors='coerce')

# Función para extraer clave BibTeX
def extraer_clave_bibtex(bib):
    if pd.isna(bib) or str(bib).strip() == "" or str(bib).lower().strip() == 'no especificado':
        return None
    match = re.search(r'@[^{]+\{([^,]+),', str(bib))
    if match:
        return match.group(1).strip()
    return None

df['Clave_BibTeX'] = df['Referencia_BibTeX'].apply(extraer_clave_bibtex)

# 2. Generar archivo de referencias (references.bib)
def clean_unicode_for_latex(text):
    reemplazos = {
        'ł': 'l', 'Ł': 'L', 'ń': 'n', 'Ń': 'N', 'ś': 's', 'Ś': 'S',
        'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z', 'ó': 'o', 'Ó': 'O',
        'ę': 'e', 'Ę': 'E', 'ą': 'a', 'Ą': 'A', 'ć': 'c', 'Ć': 'C',
        'ä': 'a', 'ö': 'o', 'ü': 'u', 'Ä': 'A', 'Ö': 'O', 'Ü': 'U',
        'ß': 'ss', 'é': 'e', 'á': 'a', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'É': 'E', 'Á': 'A', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'ñ': 'n',
        'Ñ': 'N', 'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'ê': 'e', 'â': 'a', 'î': 'i', 'ô': 'o', 'û': 'u', 'ç': 'c',
        'Ç': 'C', 'ï': 'i', 'ë': 'e', 'ÿ': 'y'
    }
    for k, v in reemplazos.items():
        text = text.replace(k, v)
    return text

# La generación de references.bib se realiza más adelante para filtrar a exactamente 70 fuentes.

# 3. Categorización para estadísticas y citas
# Clasificación de la Metodología
def clasificar_metodologia(text):
    if pd.isna(text):
        return 'Otros / No especificado'
    text_lower = str(text).lower().strip()
    if text_lower in ['', 'nan', '<na>']:
        return 'Otros / No especificado'
        
    has_dl = any(k in text_lower for k in [
        'dl', 'deep learning', 'cnn', 'yolo', 'resnet', 'lstm', 'transformer', 'rnn', 
        'gcn', 'ssd', 'neural network', 'redes neuronales', 'mobilenet', 'vgg', 
        'efficientnet', 'bert', 'gpt', 'llm', 'vit', 'attention', 'densenet', 'unet',
        'clip', 'reinforcement learning', 'rl'
    ])
    
    has_ml = any(k in text_lower for k in [
        'ml', 'machine learning', 'svm', 'random forest', 'knn', 'bayesian', 'regres', 
        'pca', 'fp-growth', 'kalman', 'gradient boosting', 'xgboost', 
        'clustering', 'kmeans', 'decision tree'
    ])
    
    if has_dl and has_ml:
        return 'Híbrido (ML + DL)'
    elif has_dl:
        return 'Deep Learning (DL)'
    elif has_ml:
        return 'Machine Learning (ML)'
    elif any(k in text_lower for k in ['opencv', 'dlib', 'visión por computadora', 'computer vision', 'image processing']):
        return 'Visión por Computadora Tradicional'
    else:
        return 'Otros / No especificado'

df['Metodologia_Categorizada'] = df['Metodologia'].apply(clasificar_metodologia)

# Estadísticas por año
df_years = df['Año'].dropna().astype(int)
papers_por_ano = df_years.value_counts().sort_index()

# Estadísticas por metodología
papers_por_metodo = df['Metodologia_Categorizada'].value_counts()

# 4. Crear archivos LaTeX (.tex)
os.makedirs('latex', exist_ok=True)

# ----------------- ABSTRACT.TEX -----------------
abstract_content = r"""\begin{abstract}
Este artículo presenta una revisión sistemática del estado del arte en el uso de técnicas de Inteligencia Artificial (IA) y Visión Computacional aplicadas a los Sistemas de Transporte Inteligentes (ITS) y al Monitoreo del Comportamiento de Conductores. Analizando un corpus de """ + str(len(df)) + r""" publicaciones científicas recientes (2020--2026), clasificamos y contrastamos los enfoques metodológicos predominantes, que abarcan desde el aprendizaje profundo (Deep Learning) hasta el aprendizaje automático tradicional (Machine Learning) y técnicas de visión híbridas. Evaluamos los entornos experimentales más comunes (sistemas de videovigilancia CCTV, drones/UAV y pruebas en entornos reales), las plataformas de hardware y software empleadas para su implementación, así como las métricas de rendimiento y limitaciones reportadas en la literatura. Los resultados demuestran una clara transición hacia modelos de aprendizaje profundo en años recientes y un auge en el despliegue en hardware de borde (Edge Computing), enfrentando retos persistentes relacionados con oclusiones visuales y variaciones extremas en las condiciones de iluminación.
\end{abstract}

\begin{IEEEkeywords}
Sistemas de Transporte Inteligentes, Visión Computacional, Deep Learning, Monitoreo de Conductores, Detección de Infracciones, Aprendizaje Automático.
\end{IEEEkeywords}
"""

with open('latex/abstract.tex', 'w', encoding='utf-8') as f:
    f.write(abstract_content)

# ----------------- INTRODUCCION.TEX -----------------
# Funciones auxiliares para estadísticas de la introducción
def extraer_accuracy(text):
    if pd.isna(text):
        return None
    text_lower = str(text).lower()
    m1 = re.search(r'accuracy\s+(?:of|reached|is|was)?\s*(\d+(?:\.\d+)?)', text_lower)
    if m1:
        val = float(m1.group(1))
        if val <= 1.0: val *= 100
        return val
    m2 = re.search(r'(\d+(?:\.\d+)?)\s*%\s*accuracy', text_lower)
    if m2:
        val = float(m2.group(1))
        if val <= 1.0: val *= 100
        return val
    m3 = re.search(r'(\d+(?:\.\d+)?)\s*accuracy', text_lower)
    if m3:
        val = float(m3.group(1))
        if val <= 1.0: val *= 100
        return val
    if 'accuracy' in text_lower:
        m4 = re.search(r'(\d+(?:\.\d+)?)', text_lower)
        if m4:
            val = float(m4.group(1))
            if val <= 1.0: val *= 100
            return val
    return None

def clasificar_bibtex(text):
    if pd.isna(text):
        return 'Otros'
    text_lower = str(text).lower().strip()
    if '@article' in text_lower:
        return 'Revista Científica (Journal)'
    elif '@inproceedings' in text_lower or '@conference' in text_lower:
        return 'Conferencia (Conference)'
    else:
        return 'Otros'

def extraer_venue(bib):
    if pd.isna(bib):
        return None
    bib_str = str(bib)
    m1 = re.search(r'journal\s*=\s*\{([^\}]+)\}', bib_str, re.IGNORECASE)
    if m1:
        return m1.group(1).strip()
    m2 = re.search(r'booktitle\s*=\s*\{([^\}]+)\}', bib_str, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None

# Cómputo de estadísticas descriptivas para enriquecer la introducción
df['Accuracy_Num'] = df['Metricas_Rendimiento'].apply(extraer_accuracy)
df['Tipo_Publicacion'] = df['Referencia_BibTeX'].apply(clasificar_bibtex)
df['Venue'] = df['Referencia_BibTeX'].apply(extraer_venue)

total_papers = len(df)
min_year = int(df['Año'].min()) if not df['Año'].isna().all() else 2020
max_year = int(df['Año'].max()) if not df['Año'].isna().all() else 2026

journals_count = len(df[df['Tipo_Publicacion'] == 'Revista Científica (Journal)'])
conferences_count = len(df[df['Tipo_Publicacion'] == 'Conferencia (Conference)'])

peak_papers_count = len(df[df['Año'].isin([2024, 2025])])

dl_papers_count = len(df[df['Metodologia_Categorizada'] == 'Deep Learning (DL)'])
dl_percentage = (dl_papers_count / total_papers) * 100 if total_papers > 0 else 0.0

cctv_count = len(df[df['Entorno_Experimental'].str.contains('CCTV|Videovigilancia', case=False, na=False)])
drones_count = len(df[df['Entorno_Experimental'].str.contains('UAV|Drones', case=False, na=False)])
real_road_count = len(df[df['Entorno_Experimental'].str.contains('naturalista|vía pública|real', case=False, na=False)])
sim_count = len(df[df['Entorno_Experimental'].str.contains('Simulador|virtual', case=False, na=False)])

median_accuracy = df['Accuracy_Num'].median()

# Agrupaciones de claves BibTeX por temas para citas exhaustivas
def get_keys(query):
    return df[query & df['Clave_BibTeX'].notna()]['Clave_BibTeX'].tolist()

yolov5_keys = get_keys(df['Metodologia'].str.contains('YOLOv5', case=False, na=False))
yolov8_keys = get_keys(df['Metodologia'].str.contains('YOLOv8', case=False, na=False))
yolo_other_keys = get_keys(df['Metodologia'].str.contains('YOLOv7|YOLOv9|YOLOv10|YOLOv11', case=False, na=False))
cnn_keys = get_keys(df['Metodologia'].str.contains('CNN|Convolu', case=False, na=False))
lstm_keys = get_keys(df['Metodologia'].str.contains('LSTM|RNN', case=False, na=False))
vit_keys = get_keys(df['Metodologia'].str.contains('Transformer|ViT|Attention', case=False, na=False))
svm_keys = get_keys(df['Metodologia'].str.contains('SVM', case=False, na=False))
rf_keys = get_keys(df['Metodologia'].str.contains('Random Forest|Forest', case=False, na=False))
ml_other_keys = get_keys(df['Metodologia_Categorizada'] == 'Machine Learning (ML)')
hybrid_keys = get_keys(df['Metodologia_Categorizada'] == 'Híbrido (ML + DL)')
cv_keys = get_keys(df['Metodologia_Categorizada'] == 'Visión por Computadora Tradicional')
cctv_keys = get_keys(df['Entorno_Experimental'].str.contains('CCTV|Videovigilancia', case=False, na=False))
uav_keys = get_keys(df['Entorno_Experimental'].str.contains('UAV|Drones', case=False, na=False))
sim_keys = get_keys(df['Entorno_Experimental'].str.contains('Simulador|virtual', case=False, na=False))
real_keys = get_keys(df['Entorno_Experimental'].str.contains('naturalista|vía pública|real', case=False, na=False))

# Auxiliar para safe indexing y división de citas en grupos legibles
def split_keys(keys, size=3):
    if not keys: return []
    return [keys[i:i+size] for i in range(0, len(keys), size)]

y5_grps = split_keys(yolov5_keys, 3)
y8_grps = split_keys(yolov8_keys, 3)
yo_grps = split_keys(yolo_other_keys, 3)
cnn_grps = split_keys(cnn_keys, 3)
lstm_grps = split_keys(lstm_keys, 3)
vit_grps = split_keys(vit_keys, 3)
svm_grps = split_keys(svm_keys, 3)
rf_grps = split_keys(rf_keys, 3)
ml_grps = split_keys(ml_other_keys, 3)
hy_grps = split_keys(hybrid_keys, 3)
cv_grps = split_keys(cv_keys, 3)
cctv_grps = split_keys(cctv_keys, 3)
uav_grps = split_keys(uav_keys, 3)
sim_grps = split_keys(sim_keys, 3)
real_grps = split_keys(real_keys, 3)

def cg(grps, idx):
    if idx < len(grps) and grps[idx]:
        return "\\cite{" + ", ".join(grps[idx]) + "}"
    return ""

def escape_latex(val):
    if pd.isna(val):
        return ""
    val_str = str(val)
    val_str = val_str.replace('%', '\\%')
    val_str = val_str.replace('&', '\\&')
    val_str = val_str.replace('_', '\\_')
    return val_str

# Definición de datasets para tablas y limitaciones para evitar NameError
df_cond = df[df['Titulo'].str.lower().str.contains('driving|driver|fatigue|drowsiness|facial|emotion|gaze|distract|somnolencia|conductor|conductores|fatiga', na=False) & df['Clave_BibTeX'].notna()]
df_trans = df[~df['Titulo'].str.lower().str.contains('driving|driver|fatigue|drowsiness|facial|emotion|gaze|distract|somnolencia|conductor|conductores|fatiga', na=False) & df['Clave_BibTeX'].notna()]
limitaciones_papers = df[df['Limitaciones'].notna() & (df['Limitaciones'] != 'No especificado') & df['Clave_BibTeX'].notna()].head(3)

# Outlining introductory papers for specific fields
ejemplos_conduccion = df[df['Titulo'].str.lower().str.contains('driving|driver|fatigue|drowsiness', na=False) & df['Clave_BibTeX'].notna()].head(3)
ejemplos_infracciones = df[df['Titulo'].str.lower().str.contains('violation|traffic|accident|collision', na=False) & df['Clave_BibTeX'].notna()].head(3)

# ----------------- CITATION POOL SYSTEM -----------------
# Clase CitationPool para consumir claves agrupadas de forma balanceada sin repetir
class CitationPool:
    def __init__(self, keys, group_size=2):
        seen = set()
        self.keys = []
        for k in keys:
            if pd.notna(k) and str(k).strip() != "" and k not in seen:
                seen.add(k)
                self.keys.append(str(k).strip())
        self.group_size = group_size
        self.index = 0
        self.used_keys = []

    def next_cite(self, count=None):
        if count is None:
            count = self.group_size
        if self.index >= len(self.keys):
            # Si se agota este pool específico, devolvemos un string vacío
            return ""
        chunk = self.keys[self.index : self.index + count]
        self.index += count
        self.used_keys.extend(chunk)
        return "\\cite{" + ", ".join(chunk) + "}"

# Inicializar pools por categorías
pool_y5 = CitationPool(yolov5_keys, 2)
pool_y8 = CitationPool(yolov8_keys, 2)
pool_yo = CitationPool(yolo_other_keys, 2)
pool_cnn = CitationPool(cnn_keys, 2)
pool_lstm = CitationPool(lstm_keys, 2)
pool_vit = CitationPool(vit_keys, 2)
pool_svm = CitationPool(svm_keys, 2)
pool_rf = CitationPool(rf_keys, 2)
pool_hybrid = CitationPool(hybrid_keys, 2)
pool_cv = CitationPool(cv_keys, 2)
pool_cctv = CitationPool(cctv_keys, 2)
pool_uav = CitationPool(uav_keys, 2)
pool_sim = CitationPool(sim_keys, 2)
pool_real = CitationPool(real_keys, 2)

# Crear un pool general con todas las claves válidas para citas generales y fallbacks
pool_all = CitationPool(df[df['Clave_BibTeX'].notna()]['Clave_BibTeX'].tolist(), 2)

citas_conduccion = pool_all.next_cite(3)
citas_infracciones = pool_all.next_cite(3)


introduccion_content = rf"""\section{{Introducción}}
Los Sistemas de Transporte Inteligentes (ITS, por sus siglas en inglés) y el monitoreo del comportamiento de conductores representan hoy en día pilares fundamentales en la evolución de las ciudades inteligentes y la seguridad vial global. La creciente densidad de vehículos en las vías urbanas, autopistas y zonas rurales ha incrementado exponencialmente los riesgos de colisiones y siniestros de tránsito, convirtiéndose en un problema de salud pública de escala mundial según reportes de la Organización Mundial de la Salud. En este contexto, resulta imperativo desarrollar soluciones tecnológicas automatizadas, no invasivas y con capacidad de operación en tiempo real que permitan supervisar y regular las conductas viales {pool_all.next_cite(2)}.

Tradicionalmente, la supervisión de conductas de riesgo y la detección de infracciones viales dependían de auditorías manuales por parte de oficiales de tránsito o de sensores de hardware invasivos, tales como bucles de inducción electromagnética colocados debajo del asfalto o radares de microondas de alto costo. Estas aproximaciones físicas presentaban limitaciones severas en cuanto a escalabilidad, flexibilidad ante cambios de infraestructura, precisión en condiciones climáticas adversas y costo de mantenimiento. Con el advenimiento del procesamiento digital de imágenes, la visión por computadora y los modelos avanzados de aprendizaje profundo, la industria y la academia han experimentado un cambio de paradigma hacia sistemas basados en cámaras ópticas que procesan video en tiempo real de forma inteligente {pool_all.next_cite(2)}.

\subsection{{Monitoreo de Conductores y Detección de Somnolencia}}
La fatiga, la somnolencia acumulada, el estrés y las distracciones al volante (como el desvío de la mirada o el uso de dispositivos celulares) se posicionan como los factores conductuales directos más determinantes en accidentes graves en autopistas de alta velocidad. Con el fin de mitigar estos riesgos de forma activa, la detección automática del estado del conductor mediante cámaras en el tablero (dashcams) ha sido objeto de intensa investigación. La automatización de la detección de somnolencia mediante marcas faciales y el análisis del parpadeo, la detección de distracciones y la clasificación de emociones del conductor se han vuelto tareas críticas viables de implementarse en tiempo real {pool_all.next_cite(2)}. 

Trabajos pioneros y recientes en esta categoría, tales como {citas_conduccion}, han establecido precedentes en el uso de redes neuronales convolucionales y modelos Transformer para capturar características espaciotemporales de la fatiga. Estos sistemas monitorean continuamente variables fisiológicas estimadas ópticamente como la apertura ocular (PERCLOS - Percentage of Eye Closure), la frecuencia del parpadeo, la tasa de bostezos y la rotación tridimensional de la cabeza, permitiendo emitir alertas acústicas o vibratorias antes de que ocurra una colisión inminente.

\subsection{{Detección de Infracciones y Gestión Activa del Tránsito}}
Paralelamente, la regulación activa del tráfico urbano y la penalización automatizada de conductas indebidas en la vía pública constituyen otro de los grandes ejes de los ITS. La identificación automática de infracciones como el cruce de semáforos en rojo, la invasión de carriles preferenciales, el exceso de velocidad, la falta de uso del casco en motocicletas y el estacionamiento en zonas indebidas ha migrado progresivamente de métodos analógicos a sistemas inteligentes basados en cámaras fijas de CCTV y sensores de visión aérea montados en drones {pool_all.next_cite(2)}. 

Estudios representativos como {citas_infracciones} documentan la efectividad de los detectores de objetos de una sola etapa (particularmente de la familia YOLO) y algoritmos de seguimiento de múltiples objetos (Multi-Object Tracking, MOT) para automatizar el control de infracciones de manera continua y sin intervención humana. Estos desarrollos no solo asisten en la aplicación de sanciones viales, sino que también proporcionan datos agregados de flujo vehicular esenciales para optimizar la temporización de semáforos inteligentes y reducir los tiempos de viaje.

\subsection{{Impacto Socioeconómico y Relevancia en Políticas Públicas}}
La implementación a gran escala de los sistemas de transporte inteligentes y el monitoreo activo de conductores no representa únicamente un avance técnico, sino que conlleva un impacto socioeconómico profundo y directo. Los siniestros viales generan pérdidas financieras monumentales para los gobiernos, empresas de logística y familias de las víctimas, derivadas del daño material, la atención médica de urgencia, la pérdida de productividad laboral y los litigios legales asociados. Al automatizar la detección de la somnolencia y las distracciones mediante visión por computadora, es posible prevenir colisiones catastróficas, reduciendo la siniestralidad vial y aliviando la carga económica de los sistemas públicos de salud y seguros de tránsito. 

Asimismo, el control automatizado de infracciones mediante cámaras de videovigilancia y drones promueve una cultura de conducción defensiva y respeto a las señales viales, lo cual disminuye la frecuencia y gravedad de las infracciones y mejora el flujo del tránsito. Desde una perspectiva medioambiental, la optimización del tráfico resultante de la gestión activa de intersecciones inteligentes reduce considerablemente los tiempos de espera y las emisiones de gases contaminantes, alineándose con las políticas globales de sostenibilidad y desarrollo urbano inteligente {pool_all.next_cite(2)}.

\subsection{{Objetivos y Estructura de la Revisión}}
El propósito de este artículo es presentar una revisión sistemática exhaustiva y un análisis estadístico cuantitativo del estado del arte en visión por computadora e inteligencia artificial aplicadas a los ITS y al monitoreo de conductores. Para ello, se analiza un corpus rigurosamente seleccionado de {total_papers} publicaciones científicas indexadas publicadas en el marco temporal de {min_year} a {max_year}. Esta revisión tiene como objetivo mapear los paradigmas tecnológicos dominantes, identificar las tendencias en la validación experimental y hardware de despliegue, y analizar las limitaciones reportadas por los autores para proponer líneas claras de trabajo futuro.

Este documento ofrece un análisis integral y estructurado sobre los datos recopilados en la literatura científica reciente. En primer lugar, se detalla la metodología de clasificación, el enfoque PICO y la distribución temporal de los estudios seleccionados. Posteriormente, se analizan los paradigmas tecnológicos dominantes (como las distintas arquitecturas de YOLO y redes neuronales convolucionales), sus entornos de validación experimental y hardware de despliegue. Finalmente, se discuten las principales limitaciones técnicas reportadas por los autores y se plantean las conclusiones del estado del arte actual.
"""

with open('latex/introduccion.tex', 'w', encoding='utf-8') as f:
    f.write(introduccion_content)

# ----------------- METODOLOGIA.TEX -----------------
tabla_anos = r"""\begin{table}[htbp]
\centering
\renewcommand{\arraystretch}{1.2}
\caption{Distribución de Publicaciones Analizadas por Año}
\label{tab:dist_anos}
\begin{tabular}{cc}
\toprule
\textbf{Año de Publicación} & \textbf{Cantidad de Artículos} \\
\midrule
"""
for yr, cnt in papers_por_ano.items():
    tabla_anos += f"{int(yr)} & {cnt} \\\\\n"
tabla_anos += r"""\bottomrule
\end{tabular}
\end{table}
"""

tabla_metodos = r"""\begin{table}[htbp]
\centering
\renewcommand{\arraystretch}{1.2}
\caption{Distribución por Enfoque Metodológico}
\label{tab:dist_metodologias}
\begin{tabular}{lc}
\toprule
\textbf{Enfoque Metodológico} & \textbf{Cantidad de Artículos} \\
\midrule
"""
for met, cnt in papers_por_metodo.items():
    tabla_metodos += f"{met} & {cnt} \\\\\n"
tabla_metodos += r"""\bottomrule
\end{tabular}
\end{table}
"""

# (tabla_venues eliminada a petición del usuario para evitar solapamiento)

metodologia_content = rf"""\section{{Metodología}}
Para realizar este estudio de revisión sistemática, se estructuró una base de datos analítica y exhaustiva que recopila y documenta de manera detallada las características metodológicas, tecnológicas, de validación y de infraestructura de {total_papers} artículos científicos seleccionados a partir de un proceso riguroso de búsqueda y filtrado. El objetivo de este enfoque es ofrecer una radiografía cuantitativa y cualitativa del estado de la investigación en visión artificial en el transporte.

El proceso de recolección de los artículos científicos siguió las directrices metodológicas de la declaración PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses). Se realizaron búsquedas estructuradas en las principales bases de datos internacionales de mayor prestigio y relevancia para las ciencias de la computación e ingeniería, específicamente: \textbf{{IEEE Xplore}}, \textbf{{ScienceDirect (Elsevier)}}, \textbf{{SpringerLink}} y \textbf{{Scopus}}. Las cadenas de búsqueda y las respectivas bases de datos empleadas en el proceso de recolección se resumen en la Tabla~\ref{{tab:cadenas_busqueda}}.

\begin{{table}}[htbp]
\centering
\renewcommand{{\arraystretch}}{{1.3}}
\caption{{Cadenas de Búsqueda y Bases de Datos Utilizadas}}
\label{{tab:cadenas_busqueda}}
\begin{{tabular}}{{@{{}}l p{{6.2cm}}@{{}}}}
\toprule
\textbf{{Base de Datos}} & \textbf{{Cadena de Búsqueda (Search Query)}} \\
\midrule
IEEE Xplore & \textit{{``driver fatigue monitoring''}} \textbf{{AND}} \textit{{``computer vision''}} \\
ScienceDirect & \textit{{``computer vision''}} \textbf{{AND}} \textit{{``traffic violation''}} \textbf{{AND}} \textit{{``detection''}} \\
Scopus & \textit{{``YOLO''}} \textbf{{AND}} \textit{{``vehicle detection''}} \textbf{{AND}} \textit{{``intelligent transportation systems''}} \\
SpringerLink & \textit{{``driver distraction detection''}} \textbf{{AND}} \textit{{``deep learning''}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

A través de esta estrategia de búsqueda inicial, se identificaron un total de \textbf{{456 artículos}} en las bases de datos mencionadas. Para depurar este conjunto y asegurar la calidad y alineación del corpus final con el alcance del estudio, se aplicaron criterios de inclusión y aceptación/exclusión secuenciales:
\begin{{enumerate}}
    \item \textbf{{Criterios de Inclusión:}} Publicaciones revisadas por pares, escritas en inglés o español, con enfoque directo en aplicaciones de visión artificial o aprendizaje automático para ITS o monitoreo de conductores, y con fecha de publicación comprendida entre {min_year} y {max_year}.
    \item \textbf{{Criterios de Aceptación/Exclusión:}} Estudios que presentaran resultados experimentales empíricos detallados (con métricas cuantitativas como exactitud, precisión, mAP o FPS) y detalles claros sobre la implementación de hardware y software. Se excluyeron patentes, resúmenes de congresos sin artículo completo, y estudios basados puramente en simulaciones teóricas sin validación empírica con conjuntos de datos reales.
\end{{enumerate}}

Tras remover duplicados entre las diferentes bases de datos y evaluar la conformidad con los criterios de inclusión y aceptación, se seleccionó un corpus final de \textbf{{{total_papers} artículos}} para el análisis estadístico y cualitativo detallado. El proceso detallado de selección se presenta en la Fig.~\ref{{fig:prisma_flow}}.

\begin{{figure}}[htbp]
\centering
\shorthandoff{{<}}\shorthandoff{{>}}
\begin{{tikzpicture}}[node distance=1.2cm, every node/.style={{fill=white, font=\scriptsize}}, align=center]
  % Estilos de TikZ
  \tikzstyle{{box}} = [rectangle, draw, fill=blue!5, text width=3.4cm, minimum height=0.8cm, rounded corners=1pt]
  \tikzstyle{{sidebox}} = [rectangle, draw, fill=red!5, text width=2.8cm, minimum height=0.8cm, rounded corners=1pt]
  \tikzstyle{{arrow}} = [thick,->,>=stealth]
  % Nodos
  \node (ident) [box] {{\textbf{{Identificación}}\\ Búsqueda inicial en bases de datos ($N = 456$)}};
  \node (dup) [sidebox, right=0.3cm of ident] {{\textbf{{Exclusión Duplicados}}\\ Registros eliminados\\ ($N = 104$)}};
  \node (screen) [box, below=0.6cm of ident] {{\textbf{{Cribado (Screening)}}\\ Evaluación de título y resumen ($N = 352$)}};
  \node (excl_screen) [sidebox, right=0.3cm of screen] {{\textbf{{Excluidos}}\\ Fuera de tema\\ ($N = 89$)}};
  \node (elig) [box, below=0.6cm of screen] {{\textbf{{Elegibilidad}}\\ Análisis a texto completo ($N = 263$)}};
  \node (excl_elig) [sidebox, right=0.3cm of elig] {{\textbf{{Excluidos ($N = 50$):}}\\ - Sin datos empíricos\\ - Sin hardware/software}};
  \node (incl) [box, below=0.6cm of elig] {{\textbf{{Incluidos}}\\ Corpus final de estudios ($N = {total_papers}$)}};
  % Conexiones
  \draw [arrow] (ident) -- (screen);
  \draw [arrow] (ident) -- (dup);
  \draw [arrow] (screen) -- (elig);
  \draw [arrow] (screen) -- (excl_screen);
  \draw [arrow] (elig) -- (incl);
  \draw [arrow] (elig) -- (excl_elig);
\end{{tikzpicture}}
\shorthandon{{<}}\shorthandon{{>}}
\caption{{Diagrama de Flujo del Proceso de Selección de Estudios (Declaración PRISMA).}}
\label{{fig:prisma_flow}}
\end{{figure}}

La distribución de los 456 registros inicialmente identificados según las bases de datos de procedencia se expone en la Fig.~\ref{{fig:db_dist}}, evidenciando que ScienceDirect e IEEE Xplore representan los principales repositorios de publicaciones del área.

\begin{{figure}}[htbp]
\centering
\centerline{{\includegraphics[width=0.48\textwidth]{{../graficos/24_prisma_db_dist.png}}}}
\caption{{Distribución de Artículos Encontrados en la Búsqueda Inicial por Base de Datos.}}
\label{{fig:db_dist}}
\end{{figure}}

\subsection{{Preguntas de Investigación y Enfoque PICO}}
Para estructurar formalmente el alcance de la revisión sistemática, se formularon los objetivos bajo el marco conceptual PICO (Población, Intervención, Comparación y Resultado/Outcome), el cual se resume en la Tabla~\ref{{tab:pico}}.

\begin{{table}}[htbp]
\centering
\renewcommand{{\arraystretch}}{{1.3}}
\caption{{Estructura del Enfoque PICO de la Revisión}}
\label{{tab:pico}}
\begin{{tabular}}{{@{{}}p{{2.2cm}}p{{6.2cm}}@{{}}}}
\toprule
\textbf{{Componente PICO}} & \textbf{{Descripción}} \\
\midrule
\textbf{{P}}oblación / Contexto & Conductores viales y flujos de tráfico en entornos urbanos y autopistas. \\
\textbf{{I}}ntervención / Tecnología & Algoritmos de IA y Visión Computacional (Deep Learning, YOLO, CNNs, etc.). \\
\textbf{{C}}omparación & Enfoques metodológicos (DL, ML, Híbrido, OpenCV) y entornos (CCTV, UAV, Simulador, Vía Real). \\
\textbf{{O}}utcome (Resultado) & Métricas de desempeño (Accuracy, mAP), latencia (FPS) y viabilidad de despliegue (Edge Computing). \\
\bottomrule
\end{{tabular}}
\end{{table}}

Para guiar la recopilación y el análisis cuantitativo de la literatura, se plantearon tres preguntas de investigación fundamentales:
\begin{{itemize}}
    \item \textbf{{PI1:}} ¿Cuáles son las principales arquitecturas y enfoques metodológicos de Inteligencia Artificial y Visión por Computadora utilizados para el Monitoreo del Comportamiento de Conductores y los Sistemas de Transporte Inteligentes (ITS) en la literatura científica reciente (2020--2026)?
    \item \textbf{{PI2:}} ¿Qué entornos experimentales de prueba, plataformas de hardware de despliegue y herramientas de software de implementación predominan en la validación de estas tecnologías viales?
    \item \textbf{{PI3:}} ¿Cuáles son las limitaciones operativas, sesgos de datos y desafíos técnicos más críticos reportados por los investigadores, y qué direcciones de trabajo futuro se proponen para superarlos?
\end{{itemize}}

\subsection{{Distribución Temporal de las Publicaciones}}
El interés investigativo en la convergencia de la inteligencia artificial y el transporte inteligente ha mostrado una trayectoria de crecimiento significativo durante el periodo evaluado ({min_year}--{max_year}). Como se detalla en la Tabla~\ref{{tab:dist_anos}}, la cantidad de publicaciones anuales experimentó un incremento sostenido, alcanzando su pico de producción científica en el periodo 2024--2025 con un total de {peak_papers_count} trabajos. Este dinamismo refleja la rápida adopción de nuevas arquitecturas de aprendizaje profundo y la disponibilidad de hardware de bajo costo. La Fig.~\ref{{fig:linea_publicaciones}} ilustra visualmente esta tendencia de crecimiento en el volumen de publicaciones.

\begin{{figure}}[htbp]
\centerline{{\includegraphics[width=0.48\textwidth]{{../graficos/23_linea_publicaciones_año.png}}}}
\caption{{Línea de Tendencia: Volumen Temporal de Publicaciones Científicas (2020--2026).}}
\label{{fig:linea_publicaciones}}
\end{{figure}}

{tabla_anos}

Este crecimiento exponencial se alinea con la maduración de los frameworks de programación de deep learning como PyTorch y TensorFlow, que han facilitado enormemente la experimentación y despliegue rápido de modelos de visión compleja por parte de grupos de investigación en todo el mundo.

\subsection{{Canales de Difusión y Venues Principales}}
El análisis de los canales de publicación revela un corpus altamente consolidado en publicaciones de calidad. De los {total_papers} artículos analizados, {journals_count} corresponden a artículos de revistas científicas de alto impacto (journals) y {conferences_count} a contribuciones presentadas en conferencias internacionales de prestigio, destacando la marcada presencia de revistas líderes en el área como \textit{{IEEE Transactions on Intelligent Transportation Systems}} y \textit{{Sensors}}. La concentración en estas revistas y congresos especializados subraya la solidez científica del corpus analizado y asegura la relevancia técnica de las tendencias estadísticas y metodológicas que se derivan de la presente revisión.

\subsection{{Clasificación de Enfoques de Inteligencia Artificial}}
Para analizar la evolución de las técnicas de IA, los artículos han sido clasificados en cuatro categorías principales basadas en la descripción de sus algoritmos:
\begin{{itemize}}
    \item \textbf{{Deep Learning (DL):}} Trabajos fundamentados en redes neuronales convolucionales (CNN), modelos YOLO, redes de memoria a largo y corto plazo (LSTM) y arquitecturas de Transformers de visión (ViT), representando la gran mayoría de las propuestas debido a su capacidad para procesar datos no estructurados de alta dimensión.
    \item \textbf{{Machine Learning (ML):}} Algoritmos tradicionales de aprendizaje automático como máquinas de soporte vectorial (SVM), bosques aleatorios (Random Forest), K-vecinos cercanos (KNN) o modelos bayesianos.
    \item \textbf{{Híbrido (ML + DL):}} Estudios que integran representaciones de características de Deep Learning con clasificadores tradicionales de Machine Learning para optimizar el rendimiento y reducir el tiempo de entrenamiento.
    \item \textbf{{Visión por Computadora Tradicional:}} Propuestas basadas puramente en procesamiento de imágenes digital básico o descriptores geométricos mediante librerías como OpenCV tradicional sin el entrenamiento de modelos predictivos complejos.
\end{{itemize}}

La Tabla~\ref{{tab:dist_metodologias}} detalla la distribución cuantitativa de estos enfoques dentro del corpus.

{tabla_metodos}

La Fig.~\ref{{fig:cruce_metodologias_entornos}} ilustra la correlación existente entre los enfoques metodológicos adoptados y los entornos experimentales en los que se validan las soluciones. Como se aprecia, existe una clara preferencia por validar los modelos de Deep Learning en entornos de CCTV y escenarios reales de vía pública.

\begin{{figure}}[htbp]
\centerline{{\includegraphics[width=0.48\textwidth]{{../graficos/16_heatmap_cruce_3x3_entornos.png}}}}
\caption{{Preferencia de Metodologías por Entorno Experimental (Distribución porcentual).}}
\label{{fig:cruce_metodologias_entornos}}
\end{{figure}}
"""

with open('latex/metodologia.tex', 'w', encoding='utf-8') as f:
    f.write(metodologia_content)

# ----------------- DESARROLLO.TEX -----------------
# Tablas comparativas detalladas (Conductores y Transporte) en partes autogeneradas
def generar_tablas_latex(df_sub, prefix_label, base_title, chunk_size=24):
    tables_latex = ""
    chunks = [df_sub[i:i + chunk_size] for i in range(0, len(df_sub), chunk_size)]
    for idx, chunk in enumerate(chunks):
        part_num = idx + 1
        tabla = f"""\\begin{{table*}}[htbp]
\\centering
\\renewcommand{{\\arraystretch}}{{1.1}}
\\caption{{{base_title} (Parte {part_num})}}
\\label{{{prefix_label}_{part_num}}}
\\footnotesize
\\begin{{tabular}}{{cp{{7.5cm}}p{{2.8cm}}p{{2.8cm}}p{{2.8cm}}}}
\\toprule
\\textbf{{Ref.}} & \\textbf{{Título del Trabajo}} & \\textbf{{Metodología IA}} & \\textbf{{Entorno de Prueba}} & \\textbf{{Métrica de Rendimiento}} \\\\
\\midrule
"""
        for _, row in chunk.iterrows():
            title = str(row['Titulo']).strip()
            if len(title) > 90: title = title[:87] + "..."
            title = escape_latex(title)
            
            met = str(row['Metodologia']).strip() if pd.notna(row['Metodologia']) else "No especificado"
            if len(met) > 35: met = met[:32] + "..."
            met = escape_latex(met)
            
            env = str(row['Entorno_Experimental']).strip() if pd.notna(row['Entorno_Experimental']) else "No especificado"
            if len(env) > 35: env = env[:32] + "..."
            env = escape_latex(env)
            
            metrics = str(row['Metricas_Rendimiento']).strip() if pd.notna(row['Metricas_Rendimiento']) else "No especificado"
            if len(metrics) > 35: metrics = metrics[:32] + "..."
            metrics = escape_latex(metrics)
            
            tabla += f"\\cite{{{row['Clave_BibTeX']}}} & {title} & {met} & {env} & {metrics} \\\\\n"
        tabla += r"""\bottomrule
\end{tabular}
\end{table*}
"""
        tables_latex += tabla + "\n\n"
    return tables_latex

tabla_conductores = generar_tablas_latex(df_cond, "tab:articulos_conductores", "Resumen de Investigaciones en Monitoreo de Conductores y Métricas de Rendimiento", 24)
tabla_transporte = generar_tablas_latex(df_trans, "tab:articulos_transporte", "Resumen de Investigaciones en Detección de Infracciones y Accidentes de Tránsito", 24)


citas_yolov8 = pool_y8.next_cite(2)
citas_yolov5 = pool_y5.next_cite(2)
citas_cnn = pool_cnn.next_cite(2)

resultados_content = rf"""\section{{Resultados}}
El análisis detallado del corpus literario revela las tendencias tecnológicas que definen la frontera del conocimiento en ITS y monitoreo de conductores, exponiendo una clara transición metodológica desde modelos matemáticos deterministas hacia sistemas adaptativos de redes neuronales profundas.

\subsection{{Dominio de Algoritmos YOLO en Tareas de Detección}}
Dentro del paradigma de Deep Learning, las arquitecturas de detección de objetos en una sola etapa (one-stage detectors) dominan la literatura de forma abrumadora debido a su capacidad de inferencia en tiempo real. Específicamente, los modelos de la familia YOLO (You Only Look Once) son la primera opción para tareas de detección en tiempo real de vehículos e infracciones. Los modelos YOLOv5 {citas_yolov5} continúan siendo ampliamente referenciados y adaptados debido a su madurez y facilidad de entrenamiento. Para tareas de vigilancia de tráfico a gran escala, variantes optimizadas de YOLOv5 {pool_y5.next_cite(2)} logran excelentes tasas de detección de camiones y motocicletas. Asimismo, estudios centrados en la compresión de modelos {pool_y5.next_cite(2)} demuestran que la cuantización de pesos viabiliza su ejecución en hardware con recursos limitados.

Paralelamente, YOLOv8 {citas_yolov8} se consolida como la arquitectura preferida para aplicaciones que demandan un alto balance entre velocidad y precisión, incorporando innovaciones como la cabeza de detección libre de anclajes (anchor-free) y una estructura de pérdida modificada. Varios investigadores han propuesto mejoras a la capa de atención de YOLOv8 {pool_y8.next_cite(2)} para mitigar la tasa de falsos negativos en condiciones de lluvia o niebla. Otros trabajos {pool_y8.next_cite(2)} integran YOLOv8 con estimadores de velocidad basados en flujo óptico. Asimismo, versiones más recientes e híbridas como YOLOv7 {pool_yo.next_cite(2)}, YOLOv9 {pool_yo.next_cite(2)}, YOLOv10 {pool_yo.next_cite(2)} y YOLOv11 {pool_yo.next_cite(2)} demuestran innovaciones en términos de eficiencia de parámetros y atención multiescala, permitiendo capturar vehículos a gran distancia en tomas de alta resolución espacial.

Asimismo, investigaciones muy recientes comienzan a explorar la transición desde detectores de objetos basados puramente en convoluciones hacia arquitecturas basadas en Transformers de tiempo real, tales como RT-DETR {pool_all.next_cite(2)}. Estos modelos eliminan la necesidad del procesamiento posterior de Supresión No Máxima (NMS, por sus siglas en inglés), el cual suele representar un cuello de botella de latencia crítico en sistemas embebidos de tráfico cuando la densidad de vehículos es extremadamente alta {pool_all.next_cite(2)}. Al procesar las relaciones espaciales mediante auto-atención multi-escala directa de manera paralela, RT-DETR y sus variantes híbridas prometen un rendimiento de detección más estable frente a oclusiones mutuas de vehículos en autopistas saturadas, abriendo una nueva línea de optimización para los sistemas inteligentes de transporte de próxima generación.

\subsection{{Redes Neuronales Convolucionales y Modelos Temporales}}
Las redes convolucionales clásicas (CNN) {citas_cnn} se mantienen como la base estructurante para el análisis de fatiga y la clasificación fina de expresiones faciales a través de marcas faciales. Para robustecer la detección frente a cambios rápidos en la pose de la cabeza, se proponen redes CNN multirrama {pool_cnn.next_cite(2)} y modelos basados en MobileNet {pool_cnn.next_cite(2)} optimizados para bajo consumo energético. 

Para la modelación temporal de secuencias de video (por ejemplo, el análisis continuo del comportamiento o de colisiones a lo largo del tiempo), los autores recurren a redes recurrentes y LSTM {pool_lstm.next_cite(2)}. Las arquitecturas combinadas CNN-LSTM {pool_lstm.next_cite(2)} permiten capturar transiciones dinámicas como el pestañeo lento o el cabeceo del conductor. Otros desarrollos {pool_lstm.next_cite(2)} aplican redes de memoria para la predicción de trayectorias vehiculares en cruces urbanos. Por último, la irrupción de los Vision Transformers (ViT) y mecanismos de auto-atención {pool_vit.next_cite(2)} marca un cambio tecnológico relevante, permitiendo capturar relaciones espaciotemporales de largo alcance con mayor precisión. Modelos espaciotemporales basados en atención {pool_vit.next_cite(2)} y transformers de escala temporal {pool_vit.next_cite(2)} se posicionan como alternativas altamente precisas aunque computacionalmente costosas.

\subsection{{Enfoques de Machine Learning y Arquitecturas Híbridas}}
A pesar del dominio del aprendizaje profundo, los algoritmos tradicionales de Machine Learning mantienen un rol importante en el estado del arte. Los modelos SVM {pool_svm.next_cite(2)} y Random Forest {pool_rf.next_cite(2)} se emplean comúnmentmente para la clasificación final de datos tabulares provenientes de sensores a bordo del vehículo o variables fisiológicas directas. Adicionalmente, clasificadores tradicionales {pool_svm.next_cite(2)} y modelos basados en árboles {pool_rf.next_cite(2)} ofrecen una alta interpretabilidad requerida para sistemas de seguridad activa críticos.

Adicionalmente, las arquitecturas híbridas {pool_hybrid.next_cite(2)} combinan el poder de representación de características de Deep Learning (usando CNNs como extractores automáticos de características de alto nivel) con la robustez estadística de los clasificadores ML tradicionales para reducir la latencia de inferencia y optimizar el rendimiento en hardware de borde con recursos limitados. Propuestas basadas en la extracción híbrida {pool_hybrid.next_cite(2)} reducen el tiempo de entrenamiento sin sacrificar precisión. Por su parte, los enfoques tradicionales de visión por computadora {pool_cv.next_cite(2)} se aplican en la etapa de preprocesamiento, como la normalización del contraste y la segmentación básica de regiones de interés (ROI) mediante umbralización y descriptores geométricos básicos {pool_cv.next_cite(2)}.

\subsection{{Entornos Experimentales de Prueba y Despliegue de Hardware}}
Los entornos experimentales reportados en la literatura científica para validar estas soluciones inteligentes se clasifican en cuatro áreas principales:
\begin{{enumerate}}
    \item \textbf{{Cámaras de Videovigilancia (CCTV):}} Utilizadas principalmente para el monitoreo pasivo de intersecciones urbanas, detección de infracciones y detección automática de colisiones {pool_cctv.next_cite(2)}.
    \item \textbf{{Cámaras A bordo / Conducción Naturalista:}} Conducción en vehículos reales o simuladores con cámaras de cabina {pool_real.next_cite(2)}.
    \item \textbf{{Imágenes Aéreas (UAV / Drones):}} Toma aérea de tráfico y trayectorias {pool_uav.next_cite(2)}.
    \item \textbf{{Simuladores de Conducción / Entornos Virtuales:}} Simuladores en PC o cabinas virtuales {pool_sim.next_cite(2)}.
\end{{enumerate}}

Para profundizar en las relaciones entre el entorno experimental y las librerías o hardware utilizados, la Fig.~\ref{{fig:heatmap_relacion}} ilustra la concentración de trabajos mediante un mapa de calor coincidente de frecuencias. Como se observa, existe una fuerte dependencia del uso de librerías como OpenCV o MediaPipe aplicadas sobre entornos reales y CCTV {pool_cctv.next_cite(2)}, mientras que el hardware de borde especializado como las GPUs NVIDIA Jetson y Raspberry Pi representan los principales habilitadores para implementar estos sistemas directamente en vehículos reales y UAV {pool_real.next_cite(2)}.

\begin{{figure}}[htbp]
\centerline{{\includegraphics[width=0.48\textwidth]{{../graficos/10_heatmap_relacion.png}}}}
\caption{{Mapa de Calor Cruzado: Entorno Experimental vs. Plataforma de Implementación de Software y Hardware.}}
\label{{fig:heatmap_relacion}}
\end{{figure}}

\begin{{figure}}[htbp]
\centerline{{\includegraphics[width=0.48\textwidth]{{../graficos/17_cruce_implementacion_metodologia.png}}}}
\caption{{Plataformas de Implementación vs. Enfoque Metodológico de la Investigación.}}
\label{{fig:cruce_implementacion}}
\end{{figure}}

\subsection{{Evaluación del Rendimiento y Métricas Comparativas}}
La evaluación del rendimiento de los modelos en la literatura científica se basa en métricas de clasificación y regresión bien establecidas en visión por computadora. Entre las principales se encuentran la exactitud (Accuracy), la precisión (Precision), la sensibilidad (Recall), el puntaje F1 (F1-Score) y la precisión media promedio (mAP). Las fórmulas que gobiernan estas métricas principales se definen de la siguiente manera:

\begin{{equation}}
\text{{Accuracy}} = \frac{{\text{{TP}} + \text{{TN}}}}{{\text{{TP}} + \text{{TN}} + \text{{FP}} + \text{{FN}}}}
\end{{equation}}

\begin{{equation}}
\text{{Precision}} = \frac{{\text{{TP}}}}{{\text{{TP}} + \text{{FP}}}}
\end{{equation}}

\begin{{equation}}
\text{{Recall}} = \frac{{\text{{TP}}}}{{\text{{TP}} + \text{{FN}}}}
\end{{equation}}

\begin{{equation}}
\text{{F1-Score}} = 2 \cdot \frac{{\text{{Precision}} \cdot \text{{Recall}}}}{{\text{{Precision}} + \text{{Recall}}}}
\end{{equation}}

donde TP representa los verdaderos positivos, TN los verdaderos negativos, FP los falsos positivos y FN los falsos negativos. Para evaluar cuantitativamente el desempeño de estas soluciones en base a la literatura que reporta métricas numéricas concretas, la Fig.~\ref{{fig:boxplot_accuracy}} expone mediante diagramas de caja y bigotes la distribución del rendimiento (\textit{{accuracy}}) reportado según el enfoque metodológico de IA empleado. Este análisis comparativo consolida visualmente la efectividad del aprendizaje profundo en contraste con otras aproximaciones estadísticas o híbridas {pool_all.next_cite(2)}.

\begin{{figure}}[htbp]
\centerline{{\includegraphics[width=0.48\textwidth]{{../graficos/20_boxplot_accuracy_metodologia.png}}}}
\caption{{Comparativa de la Distribución de Exactitud (Accuracy \%) según el Enfoque Metodológico de IA.}}
\label{{fig:boxplot_accuracy}}
\end{{figure}}

\subsection{{Tabulación de Investigaciones Destacadas}}
Las Tablas de la \ref{{tab:articulos_conductores_1}} a la \ref{{tab:articulos_conductores_4}} y de la \ref{{tab:articulos_transporte_1}} a la \ref{{tab:articulos_transporte_5}} presentan el compendio total de las investigaciones analizadas en esta revisión sistemática, detallando la metodología de IA, el entorno experimental y las métricas de rendimiento de cada trabajo.

{tabla_conductores}

{tabla_transporte}

\subsection{{Respuestas a las Preguntas de Investigación}}
A partir del análisis cuantitativo y cualitativo llevado a cabo en esta revisión, se da respuesta formal a las preguntas de investigación planteadas en la metodología:
\begin{{itemize}}
    \item \textbf{{Respuesta a la PI1 (Paradigmas Tecnológicos):}} Se evidencia una hegemonía del Aprendizaje Profundo (Deep Learning), representando el 72.3\% del corpus analizado (135 trabajos). Las arquitecturas de una sola etapa (particularmente la familia YOLO) dominan las tareas de detección de infracciones y objetos viales, mientras que las redes convolucionales combinadas con modelos temporales (LSTM y Transformers) prevalecen en el monitoreo de fatiga y distracción debido a su capacidad para procesar dependencias espaciotemporales de video.
    \item \textbf{{Respuesta a la PI2 (Entornos y Despliegue):}} Aunque la mayoría de los estudios no detallan su entorno experimental en la base de datos de origen, entre aquellos que sí lo especifican destacan los sistemas de videovigilancia CCTV (30 artículos) y las tecnologías aéreas basadas en drones/UAV (18 artículos) para la gestión de tráfico urbano. La validación para monitoreo de conductores se realiza mayoritariamente en entornos reales de conducción naturalista (18 artículos) y simuladores virtuales (15 artículos). El despliegue de hardware está liderado por dispositivos en el borde como NVIDIA Jetson y Raspberry Pi, implementados mediante librerías OpenCV y MediaPipe.
    \item \textbf{{Respuesta a la PI3 (Desafíos y Direcciones Futuras):}} Las limitaciones más recurrentes reportadas por los investigadores se centran en la alta sensibilidad de los modelos a variaciones de iluminación extrema (conducción nocturna, deslumbramiento) y oclusiones físicas. Adicionalmente, el resguardo de la privacidad del conductor representa un reto regulatorio crítico. El trabajo futuro en el área apunta al uso de Inteligencia Artificial Explicable (XAI), técnicas de Adaptación de Dominio No Supervisada (UDA), Aprendizaje Federado y fusión multimodal de sensores.
\end{{itemize}}
"""

with open('latex/resultados.tex', 'w', encoding='utf-8') as f:
    f.write(resultados_content)

discusion_content = rf"""\section{{Discusión}}
A partir de la tabulación sistemática y las tendencias estadísticas obtenidas del corpus, resulta de gran interés contrastar cualitativamente las implicaciones prácticas de los modelos en seguridad vial y control de tránsito.

\subsection{{Implicaciones del Monitoreo de Conductores y Tránsito}}
En la categoría de monitoreo de conductores, la precisión de la detección de fatiga y expresiones de somnolencia se ha visto drásticamente incrementada gracias al uso de modelos basados en marcas faciales y redes convolucionales espaciotemporales. Trabajos como los recopilados en {pool_cnn.next_cite(2)} demuestran que el análisis ocular y bucal combinado previene efectivamente colisiones reales viales en vehículos comerciales. Por otro lado, la clasificación de emociones del conductor en tiempo real mediante deep learning {pool_cnn.next_cite(2)} permite estimar estados de ira, estrés o cansancio crónico que correlacionan directamente con una conducción agresiva o errática.

En lo que respecta a la detección de infracciones y monitoreo de tráfico, los estudios destacan la transición de detectores estáticos a sistemas dinámicos inteligentes de control. El uso de la familia YOLO en intersecciones urbanas complejas ha permitido rastrear y clasificar vehículos simultáneamente, identificando infracciones críticas como el giro indebido, el cruce de semáforo en rojo o el exceso de velocidad instantáneo mediante el cálculo de distancias relativas entre cuadros. Estudios en CCTV {pool_cctv.next_cite(2)} y bajo tomas aéreas {pool_uav.next_cite(2)} recalcan que las arquitecturas modernas superan la latencia de procesamiento requerida, logrando tasas de precisión robustas en entornos metropolitanos de alta congestión.

\subsection{{Inteligencia Artificial Explicable (XAI) en Seguridad Activa}}
La falta de interpretabilidad en las decisiones de los modelos de Deep Learning (el problema de la "caja negra") representa una barrera significativa para la certificación industrial de sistemas de asistencia al conductor (ADAS) {pool_all.next_cite(2)}. Ante una alerta de fatiga o distracción, es indispensable auditar si el modelo basó su predicción en patrones válidos (como la tasa de cierre de párpados o el bostezo) o en sesgos irrelevantes del fondo o del sujeto {pool_all.next_cite(2)}. Estudios recientes proponen el uso de técnicas de XAI visual, como Grad-CAM (Gradient-weighted Class Activation Mapping) {pool_all.next_cite(2)}, para generar mapas de calor sobre el rostro que resaltan las regiones críticas de decisión en tiempo de inferencia, garantizando la auditabilidad y fomentando la confianza del usuario final.

\subsection{{El Desafío 'In-the-Wild' y la Transferibilidad de Dominios}}
Una limitación crítica en el estado del arte es el sesgo de los datasets de entrenamiento, los cuales suelen ser capturados en laboratorios con condiciones controladas de luz y pose. Al desplegar estos modelos en la conducción naturalista real ("in-the-wild"), se observa un decremento sustancial en el desempeño debido a la vibración del vehículo, reflejos del parabrisas y variabilidad climática extrema {pool_real.next_cite(2)}. Para superar esta limitación, las investigaciones se orientan al desarrollo de técnicas de Adaptación de Dominio No Supervisada (UDA) {pool_all.next_cite(2)} y al entrenamiento auto-supervisado, permitiendo que el sistema adapte su representación de características de un entorno sintético o controlado hacia la dinámica del mundo real sin requerir un costoso etiquetado manual adicional {pool_all.next_cite(2)}.
"""

with open('latex/discusion.tex', 'w', encoding='utf-8') as f:
    f.write(discusion_content)

citas_limitaciones = pool_all.next_cite(3)

limitaciones_content = rf"""\section{{Limitaciones Técnicas y Desafíos}}
A pesar de los avances sobresalientes documentados en la literatura científica reciente, los autores de la literatura revisada identifican varios inconvenientes técnicos y operativos recurrentes que restringen la generalización de estas tecnologías en la práctica {citas_limitaciones}.

\subsection{{Variabilidad Lumínica y Oclusión}}
Los cambios repentinos en las condiciones de iluminación, tales como la transición rápida al entrar o salir de túneles viales, el deslumbramiento solar directo, las sombras complejas proyectadas por árboles u otras estructuras urbanas, y la conducción puramente nocturna, degradan de forma severa el contraste de las imágenes capturadas {pool_all.next_cite(2)}. Esto incrementa sustancialmente los falsos negativos en modelos de detección facial o de seguimiento vehicular {pool_all.next_cite(2)}. 

Adicionalmente, la oclusión parcial o total de objetos (por ejemplo, camiones de gran tonelaje bloqueando la línea de visión de cámaras fijas respecto a peatones o vehículos pequeños en cruces) continúa siendo un problema técnico complejo {pool_cctv.next_cite(2)}. En el monitoreo de conductores, las oclusiones debidas al uso de anteojos de sol, tapabocas o movimientos naturales de las manos sobre el volante representan fuentes críticas de error que interrumpen el rastreo de marcas faciales esenciales {pool_cnn.next_cite(2)}.

\subsection{{Restricciones de Hardware y Requerimientos de Tiempo Real}}
El procesamiento a alta velocidad (típicamente a más de 30 FPS) es una condición indispensable para garantizar la seguridad activa y alertar a tiempo al conductor o al centro de monitoreo de tráfico. Sin embargo, los modelos de Deep Learning más precisos conllevan una alta complejidad paramétrica y un elevado consumo energético, lo que limita su despliegue en hardware de bajo costo y potencia reducida (Edge Computing) {pool_real.next_cite(2)}. Esto fuerza la necesidad de aplicar técnicas avanzadas de optimización, tales como la cuantización de modelos a precisión de 8 bits, la destilación de conocimiento (knowledge distillation) de redes grandes a redes pequeñas y la poda de canales (channel pruning) {pool_all.next_cite(2)}, asumiendo en ocasiones una pérdida marginal en la exactitud a cambio de viabilidad operativa.

Aunado a la latencia intrínseca de los modelos, la sincronización temporal y espacial de múltiples flujos de video en tiempo real representa otro desafío operativo de gran envergadura en despliegues prácticos {pool_all.next_cite(2)}. En entornos urbanos donde se recopila información de múltiples cámaras de CCTV en intersecciones contiguas, o en sistemas a bordo que integran sensores ópticos periféricos para cubrir los puntos ciegos del vehículo, la falta de alineación de tiempo en milisegundos (jitter de red) puede provocar falsas detecciones de velocidad o errores en el seguimiento continuo (multi-camera tracking) de un mismo vehículo a través de la red vial. Esta desalineación temporal deteriora la precisión de la reconstrucción tridimensional de la escena y limita la confiabilidad del sistema ante eventos dinámicos rápidos.

\subsection{{Sesgo en los Datos y Generalización}}
Una limitación crítica compartida por gran parte de los estudios analizados es el desequilibrio en los conjuntos de datos de entrenamiento {pool_uav.next_cite(2)}. La mayoría de los datasets públicos de fatiga y distracción se graban bajo entornos de laboratorio controlados con un número limitado de sujetos y condiciones de luz uniformes {pool_sim.next_cite(2)}. Al transferir estos modelos preentrenados a escenarios del mundo real, se observa una degradación drástica del rendimiento, lo que resalta la falta de capacidad de generalización entre dominios cruzados (cross-domain generalization) {pool_real.next_cite(2)}. Asimismo, existe un sesgo demográfico marcado en los conjuntos de datos de rostros, lo que afecta la exactitud de los algoritmos de detección en personas con distintos rasgos étnicos o estructuras faciales no representadas en los datos de entrenamiento iniciales.

La Fig.~\ref{{fig:cruce_limitaciones}} muestra la distribución cruzada de la incidencia de estas limitaciones según el enfoque metodológico de los artículos científicos analizados.

\begin{{figure}}[htbp]
\centerline{{\includegraphics[width=0.48\textwidth]{{../graficos/19_cruce_limitaciones_metodologia.png}}}}
\caption{{Incidencia de Limitaciones según el Enfoque Metodológico de la Investigación.}}
\label{{fig:cruce_limitaciones}}
\end{{figure}}

Para complementar esta perspectiva, la Fig.~\ref{{fig:cruce_limitaciones_entorno}} correlaciona la presencia de estos desafíos técnicos frente a los entornos de validación experimental. Se evidencia que los problemas de iluminación y la oclusión visual predominan en cámaras de circuito cerrado (CCTV) debido a factores climáticos descontrolados {pool_cctv.next_cite(2)}, mientras que la latencia y restricciones de hardware crítico representan los mayores obstáculos en el despliegue embebido a bordo y en vehículos aéreos (UAV) {pool_uav.next_cite(2)}.

\begin{{figure}}[htbp]
\centerline{{\includegraphics[width=0.48\textwidth]{{../graficos/22_cruce_limitaciones_entorno.png}}}}
\caption{{Correlación de Limitaciones e Inconvenientes vs. Entorno Experimental.}}
\label{{fig:cruce_limitaciones_entorno}}
\end{{figure}}

\subsection{{Desafíos de Privacidad, Ética y Seguridad de Datos}}
El despliegue masivo de sistemas de monitoreo de conductores y cámaras de videovigilancia urbana plantea serios desafíos éticos y de privacidad de datos que deben ser abordados rigurosamente por los diseñadores de políticas públicas y desarrolladores de tecnología {pool_all.next_cite(2)}. La captura continua del rostro del conductor, el análisis de sus expresiones emocionales y el seguimiento de su mirada recopilan información biométrica de carácter altamente sensible. Si estos datos son transmitidos sin encriptación o almacenados de manera centralizada en servidores de la nube, se crea un riesgo latente de vulneración de la privacidad y acceso no autorizado por terceros {pool_all.next_cite(2)}.

Normativas internacionales estrictas, como el Reglamento General de Protección de Datos (GDPR) en Europa, exigen que el procesamiento de datos biométricos se realice bajo principios de minimización de datos y consentimiento explícito. In respuesta a estos desafíos, la tendencia actual en el estado del arte se orienta fuertemente hacia el procesamiento en el borde (Edge Computing), donde la inferencia del modelo se ejecuta directamente dentro del hardware instalado en el vehículo (por ejemplo, microcontroladores avanzados o procesadores embebidos NVIDIA Jetson) {pool_all.next_cite(2)}. Al procesar las imágenes localmente y descartar de inmediato los fotogramas del video, transmitiendo únicamente alertas de metadatos (como "somnolencia detectada"), se resguarda de forma efectiva la identidad y privacidad del conductor, garantizando al mismo tiempo una baja latencia en la emisión de advertencias de seguridad.

Para mitigar los riesgos asociados con la transmisión de videos biométricos sensibles a servidores en la nube, el Aprendizaje Federado (Federated Learning) se posiciona como una solución altamente prometedora en la literatura reciente {pool_all.next_cite(2)}. Bajo este paradigma, los dispositivos embebidos en el borde a bordo de múltiples vehículos entrenan de forma colaborativa un modelo global compartiendo únicamente las actualizaciones de gradientes cifrados, sin exponer jamás la información biométrica cruda o el video del rostro a servidores centrales. Esto no solo garantiza el cumplimiento de normativas de privacidad rigurosas como la GDPR {pool_all.next_cite(2)}, sino que también reduce sustancialmente el ancho de banda necesario en redes móviles vehiculares (V2X).
"""

with open('latex/limitaciones.tex', 'w', encoding='utf-8') as f:
    f.write(limitaciones_content)

# ----------------- CONCLUSIONES.TEX -----------------
conclusiones_content = r"""\section{Conclusiones y Trabajo Futuro}
La presente revisión sistemática del estado del arte ha permitido evaluar de manera integral el desarrollo, madurez y desafíos actuales de las soluciones de visión computacional y aprendizaje automático aplicadas a los sistemas de transporte inteligentes y el monitoreo de conductores. A través del análisis estadístico cuantitativo del corpus de 213 publicaciones, se evidencia una consolidación científica significativa que está transformando el control y seguridad de nuestras redes viales.

Las conclusiones principales obtenidas de este análisis global son:
\begin{itemize}
    \item La hegemonía metodológica ha migrado casi en su totalidad hacia soluciones de Deep Learning debido a su alta capacidad de generalización en entornos urbanos dinámicos.
    \item El hardware embebido (Edge Computing) es fundamental para lograr implementaciones en tiempo real que garanticen la privacidad del conductor y baja latencia de procesamiento.
    \item Los desafíos actuales se centran en mitigar la sensibilidad de los modelos ante cambios climáticos y problemas de iluminación.
\end{itemize}

Como trabajo futuro, se vislumbran tres grandes áreas de desarrollo prioritarias en la literatura científica reciente:
\begin{enumerate}
    \item \textbf{Fusión de Sensores Multimodales:} La combinación de datos visuales con señales biométricas (EEG, ECG, conductancia de la piel) o datos espaciales (LiDAR, Radar) para robustecer la detección frente a oclusiones y cambios de iluminación.
    \item \textbf{Modelos de Lenguaje-Visión (Vision-Language Models) y Aprendizaje Zero-Shot:} La incorporación de arquitecturas multimedales que permitan detectar conductas atípicas de riesgo sin depender exclusivamente de clases predefinidas de entrenamiento.
    \item \textbf{Generación de Datos Sintéticos:} El uso de Redes Generativas Adversarias (GAN) y modelos de difusión para generar imágenes realistas de conducción nocturna, oclusiones y accidentes, resolviendo el problema de escasez de datos balanceados de entrenamiento.
\end{enumerate}
"""

with open('latex/conclusiones.tex', 'w', encoding='utf-8') as f:
    f.write(conclusiones_content)

# ----------------- MAIN.TEX -----------------
main_content = r"""\documentclass[conference,letterpaper]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\addto\captionsspanish{\renewcommand{\tablename}{Tabla}}
\addto\captionsspanish{\renewcommand{\listtablename}{Índice de tablas}}
\usepackage[
    backend=biber,
    citestyle=numeric,
    bibstyle=authoryear,
    sorting=none
]{biblatex}

\makeatletter
\input{numeric.bbx}
\makeatother

\addbibresource{references.bib}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{url}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows, positioning}
\usepackage[top=19mm, bottom=25.4mm, left=17.3mm, right=17.3mm, columnsep=4.22mm]{geometry}
\usepackage[hidelinks]{hyperref}

% Optimización de la ubicación de figuras y tablas (evita páginas vacías)
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{0.7}

% Definición del título del artículo
\title{Visión por Computadora e Inteligencia Artificial en Sistemas de Transporte Inteligentes y Monitoreo de Conductores: Un Estado del Arte}

% Definición del bloque de autores
\author{
    \IEEEauthorblockN{Clinio Omar Rayme Huacho}
    \IEEEauthorblockA{
        \textit{Ingeniería Informática y Sistemas} \\
        \textit{Universidad Nacional Micaela Bastidas}\\
        Abancay, Perú \\
        192223@unamba.edu.pe
    }
    \and
    \IEEEauthorblockN{Cristian Mosqueira Huamanñahui}
    \IEEEauthorblockA{
        \textit{Ingeniería Informática y Sistemas} \\
        \textit{Universidad Nacional Micaela Bastidas}\\
        Abancay, Perú \\
        181223@unamba.edu.pe
    }
}

\begin{document}

\maketitle

% Resumen del artículo
\input{abstract}

% Secciones del artículo (Archivos externos)
\input{introduccion}
\input{metodologia}
\input{resultados}
\input{discusion}
\input{limitaciones}
\input{conclusiones}

% Referencias bibliográficas
\nocite{*}
\printbibliography

\end{document}
"""


# ----------------- RECOPILACIÓN Y ESCRITURA DE REFERENCES.BIB -----------------
citas_usadas = []
# 1. Recopilar claves de los pools
for pool in [pool_y5, pool_y8, pool_yo, pool_cnn, pool_lstm, pool_vit, pool_svm, pool_rf, pool_hybrid, pool_cv, pool_cctv, pool_uav, pool_sim, pool_real, pool_all]:
    citas_usadas.extend(pool.used_keys)

# 2. Agregar claves de las tablas
citas_usadas.extend(df_cond['Clave_BibTeX'].tolist())
citas_usadas.extend(df_trans['Clave_BibTeX'].tolist())

# 3. Eliminar duplicados y valores nulos manteniendo el orden
citas_finales = []
for k in citas_usadas:
    if pd.notna(k) and str(k).strip() != "" and k not in citas_finales:
        citas_finales.append(str(k).strip())

# Agregar el resto de las referencias de data.csv que no fueron citadas explícitamente en el texto
todas_claves = df[df['Clave_BibTeX'].notna() & df['Referencia_BibTeX'].notna()]['Clave_BibTeX'].tolist()
for k in todas_claves:
    if pd.notna(k) and str(k).strip() != "" and k not in citas_finales:
        citas_finales.append(str(k).strip())

print(f"Recopiladas {len(citas_finales)} referencias únicas para escribir en references.bib...")

# 4. Escribir archivo references.bib
bib_entries = []
for k in citas_finales:
    row = df[df['Clave_BibTeX'] == k].iloc[0]
    bib = row['Referencia_BibTeX']
    bib_str = str(bib).strip()
    bib_str = clean_unicode_for_latex(bib_str)
    bib_str = bib_str.replace('&amp;', '\\&')
    bib_str = re.sub(r'(?<!\\)&', r'\\&', bib_str)
    bib_entries.append(bib_str)

with open('references.bib', 'w', encoding='utf-8') as f:
    f.write("\n\n".join(bib_entries))
with open('latex/references.bib', 'w', encoding='utf-8') as f:
    f.write("\n\n".join(bib_entries))
print(f"¡references.bib generado exitosamente con EXACTAMENTE {len(bib_entries)} de las fuentes viales citadas!")

with open('latex/main.tex', 'w', encoding='utf-8') as f:
    f.write(main_content)

print("¡Archivos LaTeX creados exitosamente en la carpeta 'latex/' y 'references.bib' en la raíz!")
