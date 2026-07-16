import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Configuración de Estilos para Publicaciones Científicas
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16,
    'legend.fontsize': 11,
    'figure.dpi': 300
})

# Paleta de colores personalizada (escala de azules/grises con detalles en verde azulado y coral suave)
colors_palette = ['#1F4E79', '#2E75B6', '#5B9BD5', '#7F8C8D', '#BDC3C7', '#34495E', '#D35400', '#16A085']
sns.set_palette(sns.color_palette(colors_palette))

# Crear carpeta para guardar los gráficos si no existe
os.makedirs('graficos', exist_ok=True)

# 2. Cargar y Renombrar Columnas
print("Cargando data.csv...")
df = pd.read_csv('data.csv', encoding='utf-8')

clean_cols = [
    'ID', 'Titulo', 'Autores', 'Año', 'DOI', 'Introduccion', 'Metodologia',
    'Analisis_Requerimientos', 'Entorno_Experimental', 'Tamaño_Dataset',
    'Metricas_Rendimiento', 'Limitaciones', 'Discusiones', 'Implementacion_Software',
    'Conclusion', 'Referencia_BibTeX'
]
df.columns = clean_cols

# Convertir Año a numérico de manera segura
df['Año'] = pd.to_numeric(df['Año'], errors='coerce')

# 3. Extracción de Métricas de Rendimiento (Accuracy numérica)
# Extrae porcentajes de exactitud como float (ej. "accuracy of 95.1" -> 95.1)
def extraer_accuracy(text):
    if pd.isna(text):
        return None
    text_lower = str(text).lower()
    
    # Patrón 1: accuracy of/reached/is XX.XX
    m1 = re.search(r'accuracy\s+(?:of|reached|is|was)?\s*(\d+(?:\.\d+)?)', text_lower)
    if m1:
        val = float(m1.group(1))
        if val <= 1.0: val *= 100
        return val
        
    # Patrón 2: XX.XX% accuracy
    m2 = re.search(r'(\d+(?:\.\d+)?)\s*%\s*accuracy', text_lower)
    if m2:
        val = float(m2.group(1))
        if val <= 1.0: val *= 100
        return val
        
    # Patrón 3: XX.XX accuracy
    m3 = re.search(r'(\d+(?:\.\d+)?)\s*accuracy', text_lower)
    if m3:
        val = float(m3.group(1))
        if val <= 1.0: val *= 100
        return val
        
    # Patrón 4: fallback con palabra accuracy y algún flotante
    if 'accuracy' in text_lower:
        m4 = re.search(r'(\d+(?:\.\d+)?)', text_lower)
        if m4:
            val = float(m4.group(1))
            if val <= 1.0: val *= 100
            return val
            
    return None

df['Accuracy_Num'] = df['Metricas_Rendimiento'].apply(extraer_accuracy)

# 4. Clasificación del Tipo de Publicación y Nombre del Venue (BibTeX)
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

df['Tipo_Publicacion'] = df['Referencia_BibTeX'].apply(clasificar_bibtex)

# Extraer journal o booktitle del BibTeX
def extraer_venue(bib):
    if pd.isna(bib):
        return None
    bib_str = str(bib)
    # Buscar journal={...}
    m1 = re.search(r'journal\s*=\s*\{([^\}]+)\}', bib_str, re.IGNORECASE)
    if m1:
        return m1.group(1).strip()
    # Buscar booktitle={...}
    m2 = re.search(r'booktitle\s*=\s*\{([^\}]+)\}', bib_str, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None

df['Venue'] = df['Referencia_BibTeX'].apply(extraer_venue)

# 5. Limpieza de Datos Faltantes (Para otras columnas estadísticas)
print("Limpiando celdas vacías e inicializando tipos...")
for col in df.columns:
    if col not in ['Año', 'Accuracy_Num']:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'No especificado': pd.NA, 'nan': pd.NA, '<na>': pd.NA, '...': pd.NA})
        df[col] = df[col].astype('string')

# 6. Funciones de Clasificación Temática

# Clasificación de la Metodología
def clasificar_metodologia(text):
    if pd.isna(text):
        return 'Sin especificar'
    text_lower = str(text).lower().strip()
    if text_lower in ['', 'nan', '<na>']:
        return 'Sin especificar'
        
    if any(k in text_lower for k in ['revisión', 'revision', 'review', 'survey', 'literatura', 'estado del arte']):
        return 'Revisión / Estado del Arte'
        
    has_dl = any(k in text_lower for k in [
        'dl', 'deep learning', 'cnn', 'yolo', 'resnet', 'lstm', 'transformer', 'rnn', 
        'gcn', 'ssd', 'neural network', 'redes neuronales', 'mobilenet', 'vgg', 
        'efficientnet', 'bert', 'gpt', 'llm', 'vit', 'attention', 'densenet', 'unet',
        'clip', 'reinforcement learning', 'rl', 'actor-critic', 'deep rl'
    ])
    
    has_ml = any(k in text_lower for k in [
        'ml', 'machine learning', 'svm', 'random forest', 'knn', 'bayesian', 'regres', 
        'pca', 'fp-growth', 'kalman', 'gradient boosting', 'xgboost', 
        'clustering', 'kmeans', 'decision tree', 'árboles de decisión', 'naive bayes',
        'svr', 'gmm', 'plsr'
    ])
    
    if has_dl and has_ml:
        return 'Híbrido (ML + DL)'
    elif has_dl:
        return 'Deep Learning (DL)'
    elif has_ml:
        return 'Machine Learning (ML)'
    elif any(k in text_lower for k in ['opencv', 'dlib', 'visión por computadora', 'computacion edge', 'computer vision', 'image processing', 'procesamiento de imágenes']):
        return 'Visión por Computadora Tradicional'
    else:
        return 'Otros / No especificado'

df['Metodologia_Categorizada'] = df['Metodologia'].apply(clasificar_metodologia)

# Clasificación por Tipo de Dataset
def clasificar_tipo_dataset(text):
    if pd.isna(text):
        return 'No especificado'
    text_lower = str(text).lower().strip()
    if text_lower in ['', 'nan', '<na>', 'no especificado']:
        return 'No especificado'
    
    is_img = any(k in text_lower for k in ['image', 'images', 'imágenes', 'imagenes', 'fotos', 'photo'])
    is_vid = any(k in text_lower for k in ['video', 'videos', 'hours', 'horas', 'frames', 'frames;', 'sec', 'seconds', 'minutos', 'minutes'])
    is_sub = any(k in text_lower for k in ['subject', 'subjects', 'participant', 'participants', 'sujetos', 'participantes', 'personas'])
    
    types = []
    if is_img: types.append('Imágenes')
    if is_vid: types.append('Video / Frames')
    if is_sub: types.append('Sujetos Humanos')
    
    if not types:
        return 'Otros / Especificado'
    return ' & '.join(types)

df['Dataset_Tipo'] = df['Tamaño_Dataset'].apply(clasificar_tipo_dataset)

# --- NUEVOS AGRUPAMIENTOS PARA TABLAS DE CONTINGENCIA (Cruces de Datos 2x2 y 3x3) ---
def simplificar_metodologia_2x2(met):
    if pd.isna(met) or met in ['Sin especificar', 'Otros / No especificado']:
        return 'Otras Metodologías'
    if 'Deep Learning' in str(met):
        return 'Deep Learning (DL)'
    return 'Otras Metodologías'

def simplificar_metodologia_3x3(met):
    if pd.isna(met) or met in ['Sin especificar', 'Otros / No especificado']:
        return 'Otras / Híbrido / Tradicional'
    if 'Deep Learning' in str(met):
        return 'Deep Learning (DL)'
    if 'Machine Learning' in str(met):
        return 'Machine Learning (ML)'
    return 'Otras / Híbrido / Tradicional'

def agrupar_ano_3x3(yr):
    if pd.isna(yr):
        return 'No especificado'
    yr = int(yr)
    if yr <= 2022:
        return '2020 - 2022'
    elif yr <= 2024:
        return '2023 - 2024'
    else:
        return '2025 - 2026'

def agrupar_entorno_3x3(env):
    if pd.isna(env) or env == 'No especificado':
        return 'No especificado'
    env_lower = str(env).lower()
    if 'cctv' in env_lower or 'videovigilancia' in env_lower:
        return 'Cámaras (CCTV)'
    elif 'drone' in env_lower or 'uav' in env_lower or 'aére' in env_lower or 'aere' in env_lower:
        return 'Drones / UAV'
    elif 'vía pública' in env_lower or 'real' in env_lower or 'calle' in env_lower:
        return 'Vía pública / Real'
    return 'Otros'

def agrupar_implementacion_cruce(impl):
    if pd.isna(impl) or impl == 'No especificado':
        return 'No especificado'
    impl_lower = str(impl).lower()
    if 'interfaz' in impl_lower or 'aplicación' in impl_lower or 'aplicacion' in impl_lower or 'software' in impl_lower:
        return 'Aplicación / Interfaz'
    elif 'edge' in impl_lower or 'jetson' in impl_lower or 'raspberry' in impl_lower:
        return 'Edge Computing'
    elif 'drone' in impl_lower or 'uav' in impl_lower:
        return 'Drones / UAV'
    elif 'cctv' in impl_lower or 'cámara' in impl_lower or 'camara' in impl_lower:
        return 'Cámaras / CCTV'
    return 'Otros'

def agrupar_limitaciones_cruce(lim):
    if pd.isna(lim) or lim == 'No especificado':
        return 'No especificado'
    lim_lower = str(lim).lower()
    if 'iluminación' in lim_lower or 'iluminacion' in lim_lower or 'luz' in lim_lower or 'nocturna' in lim_lower:
        return 'Condiciones de iluminación'
    elif 'costo' in lim_lower or 'recurso' in lim_lower or 'computacional' in lim_lower:
        return 'Costo computacional / Recursos'
    elif 'privacidad' in lim_lower:
        return 'Restricciones de privacidad'
    elif 'desequilibrio' in lim_lower or 'balance' in lim_lower or 'sesgo' in lim_lower:
        return 'Desequilibrio de datasets'
    elif 'oclusión' in lim_lower or 'oclusion' in lim_lower or 'superposición' in lim_lower or 'superposicion' in lim_lower:
        return 'Oclusiones / Superposición'
    return 'Otros'

df['Metodologia_2x2'] = df['Metodologia_Categorizada'].apply(simplificar_metodologia_2x2)
df['Metodologia_3x3'] = df['Metodologia_Categorizada'].apply(simplificar_metodologia_3x3)
df['Periodo_Año'] = df['Año'].apply(agrupar_ano_3x3)
df['Entorno_3x3'] = df['Entorno_Experimental'].apply(agrupar_entorno_3x3)
df['Implementacion_Cruce'] = df['Implementacion_Software'].apply(agrupar_implementacion_cruce)
df['Limitacion_Cruce'] = df['Limitaciones'].apply(agrupar_limitaciones_cruce)

# Exportar datos limpios a datos_limpios.csv
print("Exportando datos limpios a datos_limpios.csv...")
df.to_csv('datos_limpios.csv', index=False, encoding='utf-8')
print("¡Base de datos exportada con éxito!")

# ==============================================================================
# DEFINICIÓN DE FUNCIONES DE TRAZADO (Soportan gráficos individuales y rejillas)
# ==============================================================================

# ---- GRÁFICO 1: Evolución Temporal de las Publicaciones ----
def plot_publicaciones_por_año(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        is_standalone = True
    else:
        is_standalone = False
        
    years_counts = df['Año'].dropna().astype(int).value_counts().sort_index()
    barplot1 = sns.barplot(x=years_counts.index, y=years_counts.values, color='#457B9D', edgecolor='black', linewidth=0.7, ax=ax)
    ax.set_title('Evolución Temporal de las Publicaciones por Año (2020 - 2026)', pad=15, fontweight='bold')
    ax.set_xlabel('Año de Publicación')
    ax.set_ylabel('Cantidad de Artículos')
    
    for p in barplot1.patches:
        height = p.get_height()
        if not pd.isna(height) and height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
                        
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/01_publicaciones_por_año.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 2: Distribución de Metodologías ----
def plot_distribucion_metodologias(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        is_standalone = True
    else:
        is_standalone = False
        
    method_counts = df['Metodologia_Categorizada'].value_counts()
    
    barplot2 = sns.barplot(x=method_counts.values, y=method_counts.index, hue=method_counts.index, palette='viridis', edgecolor='black', linewidth=0.7, legend=False, ax=ax)
    ax.set_title('Clasificación General de Metodologías Tecnológicas', pad=15, fontweight='bold')
    ax.set_xlabel('Cantidad de Artículos')
    ax.set_ylabel('Categoría Metodológica')
    
    total_specified_methods = method_counts.sum()
    for p in barplot2.patches:
        val = int(p.get_width())
        pct = (val / total_specified_methods) * 100 if total_specified_methods > 0 else 0
        ax.annotate(f' {val} ({pct:.1f}%)', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10, color='black', xytext=(5, 0),
                    textcoords='offset points')
    ax.set_xlim(0, max(method_counts.values) * 1.15 if len(method_counts) > 0 else 10)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/02_distribucion_metodologias.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 3: Entornos Experimentales de Evaluación ----
def plot_entornos_experimentales(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        is_standalone = True
    else:
        is_standalone = False
        
    exp_list = []
    for val in df['Entorno_Experimental'].dropna():
        parts = [p.strip() for p in val.split(',')]
        exp_list.extend(parts)
    exp_counts = pd.Series(exp_list).value_counts()
    
    barplot3 = sns.barplot(x=exp_counts.values, y=exp_counts.index, hue=exp_counts.index, palette='crest', edgecolor='black', linewidth=0.7, legend=False, ax=ax)
    ax.set_title('Distribución de Entornos Experimentales de Evaluación', pad=15, fontweight='bold')
    ax.set_xlabel('Número de Menciones')
    ax.set_ylabel('Entorno de Evaluación')
    
    for p in barplot3.patches:
        val = int(p.get_width())
        ax.annotate(f' {val}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10, color='black', xytext=(5, 0),
                    textcoords='offset points')
    ax.set_xlim(0, max(exp_counts.values) * 1.1 if len(exp_counts) > 0 else 10)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/03_entornos_experimentales.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 4: Plataformas de Implementación de Software ----
def plot_implementacion_software(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
        is_standalone = True
    else:
        is_standalone = False
        
    impl_list = []
    for val in df['Implementacion_Software'].dropna():
        parts = [p.strip() for p in val.split(',')]
        impl_list.extend(parts)
    impl_counts = pd.Series(impl_list).value_counts()
    impl_counts = impl_counts.head(12)
    
    barplot4 = sns.barplot(x=impl_counts.values, y=impl_counts.index, hue=impl_counts.index, palette='flare', edgecolor='black', linewidth=0.7, legend=False, ax=ax)
    ax.set_title('Plataformas de Implementación / Despliegue de Software (Top 12)', pad=15, fontweight='bold')
    ax.set_xlabel('Número de Menciones')
    ax.set_ylabel('Librería / Hardware / Plataforma')
    
    for p in barplot4.patches:
        val = int(p.get_width())
        ax.annotate(f' {val}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10, color='black', xytext=(5, 0),
                    textcoords='offset points')
    ax.set_xlim(0, max(impl_counts.values) * 1.1 if len(impl_counts) > 0 else 10)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/04_implementacion_software.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 5: Limitaciones y Falsos Positivos ----
def plot_limitaciones(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
        is_standalone = True
    else:
        is_standalone = False
        
    lim_list = []
    for val in df['Limitaciones'].dropna():
        parts = [p.strip() for p in val.split(',')]
        lim_list.extend(parts)
    lim_counts = pd.Series(lim_list).value_counts()
    lim_counts = lim_counts.head(10)
    
    barplot5 = sns.barplot(x=lim_counts.values, y=lim_counts.index, hue=lim_counts.index, palette='copper', edgecolor='black', linewidth=0.7, legend=False, ax=ax)
    ax.set_title('Principales Limitaciones y Fuentes de Falsos Positivos (Top 10)', pad=15, fontweight='bold')
    ax.set_xlabel('Número de Menciones')
    ax.set_ylabel('Tipo de Limitación')
    
    for p in barplot5.patches:
        val = int(p.get_width())
        ax.annotate(f' {val}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10, color='black', xytext=(5, 0),
                    textcoords='offset points')
    ax.set_xlim(0, max(lim_counts.values) * 1.1 if len(lim_counts) > 0 else 10)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/05_limitaciones.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 6: Métricas de Rendimiento Reportadas ----
def plot_metricas_rendimiento(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        is_standalone = True
    else:
        is_standalone = False
        
    metrics_dict = {
        'Accuracy / Exactitud': df['Metricas_Rendimiento'].str.lower().str.contains('accuracy|exactitud', na=False).sum(),
        'Precision / Precisión': df['Metricas_Rendimiento'].str.lower().str.contains('precision|precisión', na=False).sum(),
        'Recall / Sensibilidad': df['Metricas_Rendimiento'].str.lower().str.contains('recall|sensibilidad', na=False).sum(),
        'F1-Score': df['Metricas_Rendimiento'].str.lower().str.contains('f1|f-score|f score', na=False).sum(),
        'mAP (Mean Average Precision)': df['Metricas_Rendimiento'].str.lower().str.contains('map', na=False).sum(),
        'FPS (Frames Per Second)': df['Metricas_Rendimiento'].str.lower().str.contains('fps|frames per second|frames/s|cuadros', na=False).sum()
    }
    metrics_series = pd.Series(metrics_dict).sort_values(ascending=False)
    
    barplot6 = sns.barplot(x=metrics_series.values, y=metrics_series.index, hue=metrics_series.index, palette='mako', edgecolor='black', linewidth=0.7, legend=False, ax=ax)
    ax.set_title('Métricas de Rendimiento más Reportadas en la Literatura', pad=15, fontweight='bold')
    ax.set_xlabel('Número de Publicaciones que la Reportan')
    ax.set_ylabel('Métrica')
    
    total_papers = len(df)
    for p in barplot6.patches:
        val = int(p.get_width())
        pct = (val / total_papers) * 100 if total_papers > 0 else 0
        ax.annotate(f' {val} ({pct:.1f}%)', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10, color='black', xytext=(5, 0),
                    textcoords='offset points')
    ax.set_xlim(0, max(metrics_series.values) * 1.15 if len(metrics_series) > 0 else 10)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/06_metricas_rendimiento.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 7: Distribución por Tipo de Dataset ----
def plot_tipo_dataset(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        is_standalone = True
    else:
        is_standalone = False
        
    dataset_counts = df['Dataset_Tipo'].value_counts()
    
    ax.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%', startangle=140, 
           colors=['#1D3557', '#457B9D', '#2A9D8F', '#E76F51', '#8D99AE'][:len(dataset_counts)],
           wedgeprops={'edgecolor': 'black', 'linewidth': 0.8, 'antialiased': True})
    centre_circle = plt.Circle((0,0),0.70,fc='white',edgecolor='black',linewidth=0.5)
    ax.add_artist(centre_circle)
    ax.set_title('Clasificación por Tipo de Dataset Utilizado', pad=20, fontweight='bold')
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/07_tipo_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 8: Evolución Histórica de Metodologías ----
def plot_evolucion_metodologias(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        is_standalone = True
    else:
        is_standalone = False
        
    df_years = df[df['Año'].notna()]
    cross_tab = pd.crosstab(df_years['Año'].astype(int), df_years['Metodologia_Categorizada'])
    cols_order = [c for c in ['Deep Learning (DL)', 'Híbrido (ML + DL)', 'Machine Learning (ML)', 'Revisión / Estado del Arte', 'Visión por Computadora Tradicional', 'Sin especificar', 'Otros / No especificado'] if c in cross_tab.columns]
    cross_tab = cross_tab[cols_order]
    
    cross_tab.plot(kind='bar', stacked=True, color=colors_palette[:len(cols_order)], edgecolor='black', linewidth=0.7, ax=ax)
    ax.set_title('Evolución de Metodologías por Año (2020 - 2026)', pad=15, fontweight='bold')
    ax.set_xlabel('Año de Publicación')
    ax.set_ylabel('Cantidad de Artículos')
    ax.tick_params(axis='x', rotation=0)
    
    if is_standalone:
        ax.legend(title='Metodología', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('graficos/08_evolucion_metodologias.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        ax.legend(title='Metodología', loc='upper left', fontsize=8, title_fontsize=9)

# ---- GRÁFICO 9: Popularidad de las Versiones de YOLO ----
def plot_popularidad_yolo(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        is_standalone = True
    else:
        is_standalone = False
        
    yolo_versions = {
        'YOLOv8': df['Metodologia'].str.lower().str.contains('yolov8|yolo v8', na=False).sum(),
        'YOLOv5': df['Metodologia'].str.lower().str.contains('yolov5|yolo v5', na=False).sum(),
        'YOLOv3': df['Metodologia'].str.lower().str.contains('yolov3|yolo v3', na=False).sum(),
        'YOLOv4': df['Metodologia'].str.lower().str.contains('yolov4|yolo v4', na=False).sum(),
        'YOLOv10': df['Metodologia'].str.lower().str.contains('yolov10|yolo v10', na=False).sum(),
        'YOLOv11': df['Metodologia'].str.lower().str.contains('yolov11|yolo v11', na=False).sum(),
        'YOLOv6': df['Metodologia'].str.lower().str.contains('yolov6|yolo v6', na=False).sum(),
        'YOLOv7': df['Metodologia'].str.lower().str.contains('yolov7|yolo v7', na=False).sum(),
        'YOLO Genérico': (df['Metodologia'].str.lower().str.contains('yolo', na=False) & ~df['Metodologia'].str.lower().str.contains('v3|v4|v5|v6|v7|v8|v10|v11', na=False)).sum()
    }
    yolo_series = pd.Series(yolo_versions)
    yolo_series = yolo_series[yolo_series > 0].sort_values(ascending=False)
    
    barplot9 = sns.barplot(x=yolo_series.index, y=yolo_series.values, hue=yolo_series.index, palette='rocket', edgecolor='black', linewidth=0.7, legend=False, ax=ax)
    ax.set_title('Uso y Popularidad de Versiones YOLO en la Literatura', pad=15, fontweight='bold')
    ax.set_xlabel('Versión de YOLO')
    ax.set_ylabel('Cantidad de Publicaciones')
    
    for p in barplot9.patches:
        val = int(p.get_height())
        ax.annotate(f'{val}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
                    
    if not is_standalone:
        ax.tick_params(axis='x', rotation=15)
        
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/09_popularidad_yolo.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 10: Mapa de Calor (Heatmap) Entorno Experimental vs Implementación ----
def plot_heatmap_relacion(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        is_standalone = True
    else:
        is_standalone = False
        
    envs = {
        'Cámaras de videovigilancia (CCTV)': 'CCTV',
        'Imágenes aéreas (Drones / UAV)': 'Drones / UAV',
        'Entorno real / Vía pública': 'Vía pública',
        'Simulador de conducción': 'Simulador',
        'No especificado': 'No especificado'
    }
    impls = {
        'Aplicación de Software / Interfaz gráfica': 'App / Interfaz',
        'Edge Computing': 'Edge Computing',
        'Drones / UAV': 'Drones / UAV',
        'Sistemas de Cámaras / CCTV': 'Cámaras / CCTV',
        'Librería OpenCV': 'OpenCV',
        'NVIDIA Jetson': 'NVIDIA Jetson',
        'Raspberry Pi': 'Raspberry Pi',
        'No especificado': 'No especificado'
    }
    
    co_matrix = pd.DataFrame(0, index=envs.values(), columns=impls.values())
    
    for idx, row in df.iterrows():
        row_env = str(row['Entorno_Experimental'])
        row_impl = str(row['Implementacion_Software'])
        if pd.isna(row['Entorno_Experimental']) or pd.isna(row['Implementacion_Software']):
            continue
        for full_env, short_env in envs.items():
            if full_env in row_env:
                for full_impl, short_impl in impls.items():
                    if full_impl in row_impl:
                        co_matrix.loc[short_env, short_impl] += 1
                        
    sns.heatmap(co_matrix, annot=True, cmap="YlGnBu", fmt="d", cbar=True, linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Número de Publicaciones Co-incidentes'}, ax=ax)
    ax.set_title('Relación: Entorno Experimental vs. Plataforma de Software', pad=20, fontweight='bold')
    ax.set_xlabel('Plataforma / Hardware de Implementación')
    ax.set_ylabel('Entorno de Prueba Experimental')
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/10_heatmap_relacion.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 11: Evolución de Accuracy (%) por Año (Dispersión con Jitter) ----
def plot_dispersion_accuracy(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
        is_standalone = True
    else:
        is_standalone = False
        
    df_acc = df[df['Accuracy_Num'].notna()]
    
    np.random.seed(42)
    jitter = np.random.uniform(-0.15, 0.15, size=len(df_acc))
    df_acc_jittered = df_acc.copy()
    df_acc_jittered['Año_Jittered'] = df_acc_jittered['Año'] + jitter
    
    sns.scatterplot(
        data=df_acc_jittered,
        x='Año_Jittered',
        y='Accuracy_Num',
        hue='Metodologia_Categorizada',
        palette='Set2',
        s=90,
        edgecolor='black',
        alpha=0.9,
        ax=ax
    )
    ax.set_title('Evolución de la Exactitud (Accuracy %) de los Modelos por Año', pad=15, fontweight='bold')
    ax.set_xlabel('Año de Publicación')
    ax.set_ylabel('Exactitud (Accuracy %)')
    ax.set_ylim(50, 101.5)
    ax.set_xticks(sorted(df['Año'].dropna().unique().astype(int)))
    
    if is_standalone:
        ax.legend(title='Metodología', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('graficos/11_dispersion_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        ax.legend(title='Metodología', loc='upper left', fontsize=8, title_fontsize=9)

# ---- GRÁFICO 12: Tipo de Publicación (Revista vs Conferencia) ----
def plot_tipo_publicacion(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        is_standalone = True
    else:
        is_standalone = False
        
    pub_counts = df['Tipo_Publicacion'].value_counts()
    ax.pie(pub_counts.values, labels=pub_counts.index, autopct='%1.1f%%', startangle=90, 
           colors=['#457B9D', '#E76F51', '#2A9D8F'],
           wedgeprops={'edgecolor': 'black', 'linewidth': 0.8, 'antialiased': True})
    centre_circle = plt.Circle((0,0),0.70,fc='white',edgecolor='black',linewidth=0.5)
    ax.add_artist(centre_circle)
    ax.set_title('Distribución por Tipo de Publicación', pad=20, fontweight='bold')
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/12_tipo_publicacion.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 13: Top 10 Canales de Publicación (Venues) ----
def plot_top_venues(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
        is_standalone = True
    else:
        is_standalone = False
        
    df_venues = df[df['Venue'].notna()]
    venue_counts = df_venues['Venue'].value_counts().head(10)
    
    barplot13 = sns.barplot(x=venue_counts.values, y=venue_counts.index, hue=venue_counts.index, palette='plasma', edgecolor='black', linewidth=0.7, legend=False, ax=ax)
    ax.set_title('Top 10 Canales de Publicación (Journals y Conferencias)', pad=15, fontweight='bold')
    ax.set_xlabel('Cantidad de Artículos Publicados')
    ax.set_ylabel('Revista / Conferencia')
    
    for p in barplot13.patches:
        val = int(p.get_width())
        ax.annotate(f' {val}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', fontsize=10, color='black', xytext=(5, 0),
                    textcoords='offset points')
    ax.set_xlim(0, max(venue_counts.values) + 1 if len(venue_counts) > 0 else 10)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/13_top_venues.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 14: Barras Agrupadas (Cruce 2x2: Tipo de Publicación vs Enfoque Tecnológico) ----
def plot_heatmap_cruce_2x2(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        is_standalone = True
    else:
        is_standalone = False
        
    ct = pd.crosstab(df['Tipo_Publicacion'], df['Metodologia_2x2'])
    df_plot = ct.reset_index().melt(id_vars='Tipo_Publicacion', value_name='Cantidad', var_name='Metodología')
    
    barplot = sns.barplot(
        data=df_plot,
        x='Tipo_Publicacion',
        y='Cantidad',
        hue='Metodología',
        palette=['#457B9D', '#E76F51'],
        edgecolor='black',
        linewidth=0.7,
        ax=ax
    )
    
    ax.set_title('Distribución Científica: Tipo de Publicación vs. Enfoque Metodológico', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Tipo de Publicación', fontsize=11)
    ax.set_ylabel('Cantidad de Artículos', fontsize=11)
    
    for p in barplot.patches:
        height = p.get_height()
        if not pd.isna(height) and height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3),
                        textcoords='offset points')
                        
    ax.set_ylim(0, max(df_plot['Cantidad']) * 1.15 if len(df_plot) > 0 else 10)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/14_heatmap_cruce_2x2.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 15: Barras Apiladas al 100% (Cruce 3x3: Período vs Metodología) ----
def plot_heatmap_cruce_3x3_temporal(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        is_standalone = True
    else:
        is_standalone = False
        
    df_filtered = df
    ct = pd.crosstab(df_filtered['Periodo_Año'], df_filtered['Metodologia_3x3'])
    
    rows = [r for r in ['2020 - 2022', '2023 - 2024', '2025 - 2026', 'No especificado'] if r in ct.index]
    ct = ct.reindex(rows)
    
    cols = [c for c in ['Deep Learning (DL)', 'Machine Learning (ML)', 'Otras / Híbrido / Tradicional'] if c in ct.columns]
    ct = ct[cols]
    
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    
    ct_pct.plot(
        kind='bar',
        stacked=True,
        color=['#1D3557', '#E76F51', '#E9C46A'],
        edgecolor='black',
        linewidth=0.7,
        ax=ax
    )
    
    ax.set_title('Evolución Proporcional de Metodologías por Período de Tiempo', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Período de Publicación', fontsize=11)
    ax.set_ylabel('Porcentaje (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=0)
    
    for r_idx, row_name in enumerate(ct.index):
        cumm_pct = 0.0
        for col_name in ct.columns:
            val = ct.loc[row_name, col_name]
            pct = ct_pct.loc[row_name, col_name]
            if pct > 4:
                y_pos = cumm_pct + pct / 2.0
                ax.text(r_idx, y_pos, f'{pct:.1f}%\n({val})', ha='center', va='center', 
                        color='white' if col_name != 'Otras / Híbrido / Tradicional' else 'black', 
                        fontweight='bold', fontsize=9)
            cumm_pct += pct
            
    ax.legend(title='Metodología', loc='upper left', bbox_to_anchor=(1.02, 1) if is_standalone else (1.0, 1.0))
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/15_heatmap_cruce_3x3_temporal.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 16: Mapa de Calor (Cruce: Entorno de Prueba vs Metodología) ----
def plot_heatmap_cruce_3x3_entornos(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        is_standalone = True
    else:
        is_standalone = False
        
    df_filtered = df
    ct = pd.crosstab(df_filtered['Metodologia_3x3'], df_filtered['Entorno_3x3'])
    
    rows = [r for r in ['Deep Learning (DL)', 'Machine Learning (ML)', 'Otras / Híbrido / Tradicional'] if r in ct.index]
    cols = [c for c in ['Cámaras (CCTV)', 'Drones / UAV', 'Vía pública / Real', 'Otros', 'No especificado'] if c in ct.columns]
    ct = ct.reindex(index=rows, columns=cols)
    
    sns.heatmap(ct, annot=True, cmap="Blues", fmt="d", cbar=True, 
                linewidths=0.8, linecolor='white',
                cbar_kws={'label': 'Cantidad de Artículos'}, ax=ax)
    
    ax.set_title('Preferencia de Metodologías por Entorno Experimental', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Entorno Experimental', fontsize=11, labelpad=10)
    ax.set_ylabel('Enfoque Metodológico', fontsize=11, labelpad=10)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/16_heatmap_cruce_3x3_entornos.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 17: Barras Agrupadas (Cruce: Plataformas de Implementación vs Metodología) ----
def plot_heatmap_cruce_implementacion_metodologia(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
        is_standalone = True
    else:
        is_standalone = False
        
    df_filtered = df
    ct = pd.crosstab(df_filtered['Implementacion_Cruce'], df_filtered['Metodologia_3x3'])
    
    impl_order = ct.sum(axis=1).sort_values(ascending=False).index.tolist()
    ct = ct.reindex(impl_order)
    
    cols = [c for c in ['Deep Learning (DL)', 'Machine Learning (ML)', 'Otras / Híbrido / Tradicional'] if c in ct.columns]
    ct = ct[cols]
    
    df_plot = ct.reset_index().melt(id_vars='Implementacion_Cruce', value_name='Cantidad', var_name='Metodología')
    
    barplot = sns.barplot(
        data=df_plot,
        x='Implementacion_Cruce',
        y='Cantidad',
        hue='Metodología',
        palette=['#1D3557', '#E76F51', '#E9C46A'],
        edgecolor='black',
        linewidth=0.7,
        ax=ax
    )
    
    ax.set_title('Plataformas de Implementación vs. Enfoque Metodológico', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Plataforma / Hardware de Implementación', fontsize=11)
    ax.set_ylabel('Cantidad de Artículos', fontsize=11)
    
    for p in barplot.patches:
        height = p.get_height()
        if not pd.isna(height) and height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2),
                        textcoords='offset points')
                        
    ax.set_ylim(0, max(df_plot['Cantidad']) * 1.15 if len(df_plot) > 0 else 10)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/17_cruce_implementacion_metodologia.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 18: Barras Apiladas al 100% (Cruce: Tipo de Dataset vs Período Temporal) ----
def plot_heatmap_cruce_tipo_dataset_temporal(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        is_standalone = True
    else:
        is_standalone = False
        
    df_filtered = df
    ct = pd.crosstab(df_filtered['Periodo_Año'], df_filtered['Dataset_Tipo'])
    
    rows = [r for r in ['2020 - 2022', '2023 - 2024', '2025 - 2026', 'No especificado'] if r in ct.index]
    ct = ct.reindex(rows)
    
    cols = [c for c in ['Imágenes', 'Video / Frames', 'Sujetos Humanos', 'Otros / Especificado', 'No especificado'] if c in ct.columns]
    for c in ct.columns:
        if c not in cols:
            cols.append(c)
    ct = ct[cols]
    
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    
    colors = ['#2A9D8F', '#E76F51', '#457B9D', '#E9C46A', '#8D99AE', '#BDBDBD']
    ct_pct.plot(
        kind='bar',
        stacked=True,
        color=colors[:len(cols)],
        edgecolor='black',
        linewidth=0.7,
        ax=ax
    )
    
    ax.set_title('Evolución Proporcional del Tipo de Dataset por Período', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Período de Publicación', fontsize=11)
    ax.set_ylabel('Porcentaje (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=0)
    
    for r_idx, row_name in enumerate(ct.index):
        cumm_pct = 0.0
        for col_name in ct.columns:
            val = ct.loc[row_name, col_name]
            pct = ct_pct.loc[row_name, col_name]
            if pct > 4:
                y_pos = cumm_pct + pct / 2.0
                ax.text(r_idx, y_pos, f'{pct:.1f}%\n({val})', ha='center', va='center', 
                        color='white' if col_name not in ['Otros / Especificado', 'No especificado'] else 'black', 
                        fontweight='bold', fontsize=9)
            cumm_pct += pct
            
    ax.legend(title='Tipo de Dataset', loc='upper left', bbox_to_anchor=(1.02, 1) if is_standalone else (1.0, 1.0))
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/18_cruce_tipo_dataset_temporal.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 19: Mapa de Calor (Cruce: Limitaciones vs Metodología) ----
def plot_heatmap_cruce_limitaciones_metodologia(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        is_standalone = True
    else:
        is_standalone = False
        
    df_filtered = df
    ct = pd.crosstab(df_filtered['Metodologia_3x3'], df_filtered['Limitacion_Cruce'])
    
    rows = [r for r in ['Deep Learning (DL)', 'Machine Learning (ML)', 'Otras / Híbrido / Tradicional'] if r in ct.index]
    lim_cols = ct.sum(axis=0).sort_values(ascending=False).index.tolist()
    ct = ct.reindex(index=rows, columns=lim_cols)
    
    sns.heatmap(ct, annot=True, cmap="Blues", fmt="d", cbar=True, 
                linewidths=0.8, linecolor='white',
                cbar_kws={'label': 'Cantidad de Artículos'}, ax=ax)
    
    ax.set_title('Incidencia de Limitaciones según Enfoque Metodológico', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Principales Limitaciones Reportadas', fontsize=11, labelpad=10)
    ax.set_ylabel('Enfoque Metodológico', fontsize=11, labelpad=10)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/19_cruce_limitaciones_metodologia.png', dpi=300, bbox_inches='tight')
        plt.close()


# ==============================================================================
# EJECUCIÓN: GENERACIÓN DE GRÁFICOS
# ==============================================================================
print("Generando gráficos individuales (1-13)...")
plot_publicaciones_por_año(df)
print("- Gráfico 1 generado (Evolución Temporal)")
plot_distribucion_metodologias(df)
print("- Gráfico 2 generado (Distribución de Metodologías)")
plot_entornos_experimentales(df)
print("- Gráfico 3 generado (Entornos Experimentales)")
plot_implementacion_software(df)
print("- Gráfico 4 generado (Implementación de Software)")
plot_limitaciones(df)
print("- Gráfico 5 generado (Limitaciones)")
plot_metricas_rendimiento(df)
print("- Gráfico 6 generado (Métricas de Rendimiento)")
plot_tipo_dataset(df)
print("- Gráfico 7 generado (Tipo de Dataset)")
plot_evolucion_metodologias(df)
print("- Gráfico 8 generado (Evolución de Metodologías)")
plot_popularidad_yolo(df)
print("- Gráfico 9 generado (Popularidad YOLO)")
plot_heatmap_relacion(df)
print("- Gráfico 10 generado (Mapa de Calor)")
plot_dispersion_accuracy(df)
print("- Gráfico 11 generado (Dispersión de Accuracy)")
plot_tipo_publicacion(df)
print("- Gráfico 12 generado (Tipo de Publicación)")
plot_top_venues(df)
print("- Gráfico 13 generado (Top Venues)")

print("\nGenerando gráficos de combinación/contingencia (14-16)...")
plot_heatmap_cruce_2x2(df)
print("- Gráfico 14 generado (Barras Agrupadas 2x2)")
plot_heatmap_cruce_3x3_temporal(df)
print("- Gráfico 15 generado (Barras Apiladas 100% 3x3)")
plot_heatmap_cruce_3x3_entornos(df)
print("- Gráfico 16 generado (Matriz de Burbujas 3x3)")

# ---- GRÁFICO 20: Boxplot (Comparativa de Exactitud por Enfoque Metodológico) ----
def plot_boxplot_accuracy_metodologia(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        is_standalone = True
    else:
        is_standalone = False
        
    df_plot = df[df['Accuracy_Num'].notna()]
    order = [m for m in ['Deep Learning (DL)', 'Machine Learning (ML)', 'Otras / Híbrido / Tradicional'] if m in df_plot['Metodologia_3x3'].unique()]
    
    sns.boxplot(
        data=df_plot,
        x='Metodologia_3x3',
        y='Accuracy_Num',
        order=order,
        palette=['#1D3557', '#E76F51', '#E9C46A'],
        width=0.5,
        ax=ax
    )
    sns.stripplot(
        data=df_plot,
        x='Metodologia_3x3',
        y='Accuracy_Num',
        order=order,
        color='black',
        alpha=0.5,
        size=5,
        jitter=0.15,
        ax=ax
    )
    
    ax.set_title('Comparativa de Exactitud (Accuracy %) por Enfoque Metodológico', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Enfoque Metodológico', fontsize=11)
    ax.set_ylabel('Exactitud (Accuracy %)', fontsize=11)
    ax.set_ylim(45, 102)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/20_boxplot_accuracy_metodologia.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 21: Boxplot (Comparativa de Exactitud por Tipo de Publicación) ----
def plot_boxplot_accuracy_tipo_publicacion(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        is_standalone = True
    else:
        is_standalone = False
        
    df_plot = df[df['Accuracy_Num'].notna()]
    
    sns.boxplot(
        data=df_plot,
        x='Tipo_Publicacion',
        y='Accuracy_Num',
        palette=['#457B9D', '#E76F51', '#2A9D8F'],
        width=0.4,
        ax=ax
    )
    sns.stripplot(
        data=df_plot,
        x='Tipo_Publicacion',
        y='Accuracy_Num',
        color='black',
        alpha=0.5,
        size=5,
        jitter=0.12,
        ax=ax
    )
    
    ax.set_title('Comparativa de Exactitud (Accuracy %) por Canal de Publicación', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Tipo de Publicación', fontsize=11)
    ax.set_ylabel('Exactitud (Accuracy %)', fontsize=11)
    ax.set_ylim(45, 102)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/21_boxplot_accuracy_tipo_publicacion.png', dpi=300, bbox_inches='tight')
        plt.close()

# ---- GRÁFICO 22: Mapa de Calor (Limitaciones vs Entorno Experimental) ----
def plot_heatmap_cruce_limitaciones_entorno(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        is_standalone = True
    else:
        is_standalone = False
        
    ct = pd.crosstab(df['Entorno_3x3'], df['Limitacion_Cruce'])
    
    env_rows = ['Cámaras (CCTV)', 'Drones / UAV', 'Vía pública / Real', 'Otros', 'No especificado']
    env_rows = [r for r in env_rows if r in ct.index]
    lim_cols = ct.sum(axis=0).sort_values(ascending=False).index.tolist()
    
    ct = ct.reindex(index=env_rows, columns=lim_cols)
    
    sns.heatmap(ct, annot=True, cmap="Blues", fmt="d", cbar=True, 
                linewidths=0.8, linecolor='white',
                cbar_kws={'label': 'Cantidad de Artículos'}, ax=ax)
    
    ax.set_title('Limitaciones e Inconvenientes vs. Entorno Experimental', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Principales Limitaciones Reportadas', fontsize=11, labelpad=10)
    ax.set_ylabel('Entorno Experimental', fontsize=11, labelpad=10)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/22_cruce_limitaciones_entorno.png', dpi=300, bbox_inches='tight')
        plt.close()


# ---- GRÁFICO 23: Tendencia Temporal del Volumen de Publicaciones (Gráfico de Líneas) ----
def plot_linea_publicaciones_año(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        is_standalone = True
    else:
        is_standalone = False
        
    years_counts = df['Año'].dropna().astype(int).value_counts().sort_index()
    
    ax.plot(
        years_counts.index, 
        years_counts.values, 
        marker='o', 
        markersize=6, 
        linewidth=2.0, 
        color='#1F4E79', 
        label='Publicaciones'
    )
    
    ax.fill_between(
        years_counts.index, 
        years_counts.values, 
        color='#1F4E79', 
        alpha=0.08
    )
    
    ax.set_title('Tendencia Temporal del Volumen de Publicaciones (2020 - 2026)', pad=20, fontweight='bold', fontsize=12)
    ax.set_xlabel('Año de Publicación', fontsize=11)
    ax.set_ylabel('Cantidad de Artículos', fontsize=11)
    ax.set_xticks(sorted(years_counts.index))
    ax.grid(True, linestyle='--', alpha=0.4, color='lightgray')
    
    for x, y in zip(years_counts.index, years_counts.values):
        ax.annotate(
            f'{int(y)}', 
            (x, y), 
            textcoords="offset points", 
            xytext=(0, 8), 
            ha='center', 
            fontsize=10, 
            fontweight='bold', 
            color='black'
        )
        
    ax.set_ylim(0, max(years_counts.values) * 1.15)
    
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/23_linea_publicaciones_año.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_database_distribution(df, is_standalone=True):
    db_names = ['IEEE Xplore', 'ScienceDirect\n(Elsevier)', 'Scopus', 'SpringerLink']
    db_counts = [120, 150, 110, 76]  # Total: 456
    
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = ['#1F4E79', '#2E75B6', '#5B9BD5', '#7F8C8D']
    
    bars = ax.barh(db_names, db_counts, color=colors, edgecolor='none', height=0.55)
    
    ax.set_title('Distribución de la Búsqueda Inicial ($N = 456$)', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Cantidad de Artículos', fontsize=10)
    ax.set_xlim(0, 180)
    
    sns.despine(left=True, bottom=False)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f'{int(width)}',
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha='left', va='center',
            fontsize=9.5, fontweight='bold', color='black'
        )
        
    if is_standalone:
        plt.tight_layout()
        plt.savefig('graficos/24_prisma_db_dist.png', dpi=300, bbox_inches='tight')
        plt.close()


print("\nGenerando gráficos cruzados adicionales (17-19)...")
plot_heatmap_cruce_implementacion_metodologia(df)
print("- Gráfico 17 generado (Implementación vs Metodología)")
plot_heatmap_cruce_tipo_dataset_temporal(df)
print("- Gráfico 18 generado (Dataset vs Período)")
plot_heatmap_cruce_limitaciones_metodologia(df)
print("- Gráfico 19 generado (Limitaciones vs Metodología)")

print("\nGenerando gráficos estadísticos y correlacionales adicionales (20-22)...")
plot_boxplot_accuracy_metodologia(df)
print("- Gráfico 20 generado (Boxplot Accuracy vs Metodología)")
plot_boxplot_accuracy_tipo_publicacion(df)
print("- Gráfico 21 generado (Boxplot Accuracy vs Tipo Publicación)")
plot_heatmap_cruce_limitaciones_entorno(df)
print("- Gráfico 22 generado (Limitaciones vs Entorno)")

print("\nGenerando gráficos lineales adicionales (23)...")
plot_linea_publicaciones_año(df)
print("- Gráfico 23 generado (Línea de Tendencia por Año)")

print("\nGenerando gráficos metodológicos adicionales (24)...")
plot_database_distribution(df)
print("- Gráfico 24 generado (Búsqueda por Base de Datos)")

print("\n¡Los 14 gráficos y los 10 cruces de datos se han generado satisfactoriamente en la carpeta 'graficos/'!")
