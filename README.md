# 🌟 Dashboard de Comportamiento — Santiago Calix

Dashboard interactivo para seguimiento del comportamiento escolar de Santiago,
con modelos de Machine Learning (Regresión Lineal + Random Forest).

## ¿Cómo publicar este dashboard GRATIS en la nube?

### Paso 1 — Crear cuenta en GitHub (gratis)
1. Ve a **https://github.com** y crea una cuenta
2. Haz clic en **"New repository"**
3. Nómbralo: `santiago-dashboard`
4. Déjalo en **Public**
5. Haz clic en **"Create repository"**

### Paso 2 — Subir los archivos
Sube estos dos archivos a tu repositorio:
- `app.py`  ← el dashboard principal
- `requirements.txt`  ← las librerías necesarias

Para subirlos:
1. En tu repositorio, haz clic en **"uploading an existing file"**
2. Arrastra ambos archivos
3. Haz clic en **"Commit changes"**

### Paso 3 — Publicar en Streamlit Cloud (gratis)
1. Ve a **https://streamlit.io/cloud** y crea una cuenta con tu Gmail
2. Haz clic en **"New app"**
3. Conecta tu cuenta de GitHub
4. Selecciona tu repositorio `santiago-dashboard`
5. En "Main file path" escribe: `app.py`
6. Haz clic en **"Deploy!"**

¡En 2 minutos tendrás un link público tipo:
`https://santiago-dashboard-xxxx.streamlit.app`

### Paso 4 — Hacer tu Google Sheet público (para que el dashboard lo lea)
1. Abre tu Google Sheet
2. Haz clic en **Compartir** (esquina superior derecha)
3. En "Acceso general", cambia a **"Cualquier persona con el enlace"**
4. Asegúrate de que sea solo "Lector" (no editor)
5. Guarda

### Paso 5 — Actualizar datos diariamente
¡No necesitas hacer nada! Cada vez que la escuela llene el Google Sheet,
el dashboard se actualizará automáticamente en 5 minutos.

## Estructura de columnas del Google Sheet
```
Date | Student Name | Activity | Time | follow_num | task_transition_num |
safe_body_num | bonus_point | Total | behavior_level | Notes | My Goal | Trigger
```

## Modelos incluidos
- **Regresión Lineal**: predice el puntaje total (0-8)
- **Random Forest**: predice el nivel (good / medium / low) con probabilidades
- **Visualizaciones**: tendencias, heatmaps, desglose por actividad y día de la semana

---
Desarrollado con ❤️ para el seguimiento de Santiago
