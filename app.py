# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Intento importar matplotlib; si no está, uso plotly
try:
    import matplotlib.pyplot as plt
    USE_MPL = True
except Exception:
    import plotly.express as px  # fallback si no hay Matplotlib
    USE_MPL = False

st.set_page_config(page_title="California Housing Explorer", layout="wide")

@st.cache_data
def load_data() -> pd.DataFrame:
    ds = fetch_california_housing(as_frame=False)
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="MedHouseVal")
    df = pd.concat([X, y], axis=1)
    return df

df_california = load_data()

st.title("California Housing – Explorador Interactivo")

# ====== Chequeos simples de faltantes ======
st.subheader("Valores faltantes / vacíos (simple)")
total_nulos = int(df_california.isna().sum().sum())
st.markdown(f'Nulos por columna (isna): "{total_nulos}"')

# Strings vacíos (solo si existen columnas de texto)
if df_california.select_dtypes(include="object").shape[1] > 0:
    vacios_str = (
        df_california.select_dtypes(include="object")
        .apply(lambda s: s.fillna("").str.strip().eq("").sum())
    )
else:
    vacios_str = pd.Series(dtype=int)

total_vacios = int(vacios_str.sum()) if not vacios_str.empty else 0
st.markdown(f'Strings vacíos (solo texto): "{total_vacios}"')

# Vista rápida
st.subheader("Vista rápida del DataFrame")
st.dataframe(df_california.head())
st.write("Tipos de datos por columna:")
st.write(df_california.dtypes.to_frame("dtype"))

# ========= Fase 2: Controles =========
# ---- Sidebar: índice + toggles de secciones ----
st.sidebar.markdown("## Índice")
show_faltantes = st.sidebar.checkbox("Valores faltantes / vacíos", True)
show_vista     = st.sidebar.checkbox("Vista rápida + dtypes", True)
show_controles = st.sidebar.checkbox("Controles y filtros", True)
show_resumen   = st.sidebar.checkbox("Resumen descriptivo", True)
show_vis       = st.sidebar.checkbox("Visualizaciones (hist + scatter)", True)
show_mapa      = st.sidebar.checkbox("Mapa geográfico (opcional)", False)
st.sidebar.markdown("---")
st.sidebar.markdown("## Controles")
st.sidebar.markdown(
    "Usa los filtros para acotar por **HouseAge** y **Latitud mínima (vecindario)**."
)

# Slider HouseAge
houseage_min = float(df_california["HouseAge"].min())
houseage_max = float(df_california["HouseAge"].max())
age_range = st.sidebar.slider(
    "Rango de HouseAge",
    min_value=houseage_min, max_value=houseage_max,
    value=(houseage_min, houseage_max), step=1.0
)

df_f = df_california.loc[
    (df_california["HouseAge"] >= age_range[0]) &
    (df_california["HouseAge"] <= age_range[1])
].copy()


# Checkbox + number_input para latitud mínima (usando el DF ya filtrado por HouseAge)
use_lat_filter = st.sidebar.checkbox("Filtrar por vecindario (Latitud mínima)", value=False)
if use_lat_filter and not df_f.empty:
    lat_min_total = float(df_f["Latitude"].min())
    lat_max_total = float(df_f["Latitude"].max())
    lat_min = st.sidebar.number_input(
        "Latitud mínima",
        min_value=lat_min_total, max_value=lat_max_total,
        value=lat_min_total, step=0.5, format="%.2f"
    )
    df_f = df_f.loc[df_f["Latitude"] >= lat_min].copy()

# Resumen post-filtros
st.subheader("Resumen descriptivo")
if df_f.empty:
    st.warning("No hay filas tras aplicar los filtros.")
else:
    mediana = df_f["MedHouseVal"].median()
    rango = df_f["MedHouseVal"].max() - df_f["MedHouseVal"].min()
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas vigentes", f"{len(df_f):,}")
    c2.metric("Mediana MedHouseVal", f"{mediana:,.3f}")
    c3.metric("Rango (max - min)", f"{rango:,.3f}")

# ========= Fase 3: Visualizaciones =========
st.header("Visualizaciones")

# --- Histograma del Target (fondo negro, sin outliers) ---
st.subheader("Distribución de MedHouseVal")
if not df_f.empty:
    if USE_MPL:
        fig1, ax1 = plt.subplots(figsize=(6, 4), facecolor="black")
        ax1.set_facecolor("black")
        ax1.hist(df_f["MedHouseVal"], bins=30, color="#CCCCCC", edgecolor="#CCCCCC", alpha=0.6)

        # Estética: ejes/labels/ticks blancos
        for spine in ax1.spines.values():
            spine.set_color("white")
        ax1.tick_params(colors="white")
        ax1.set_xlabel("MedHouseVal", color="white")
        ax1.set_ylabel("Frecuencia", color="white")
        ax1.set_title("Histograma de MedHouseVal", color="white")

        st.pyplot(fig1)
    else:
        fig1 = px.histogram(df_f, x="MedHouseVal", nbins=30, title="Histograma de MedHouseVal")
        fig1.update_layout(template="plotly_dark", paper_bgcolor="black", plot_bgcolor="black", font_color="white")
        st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Sin datos para graficar.")

# --- Scatter MedInc vs MedHouseVal (fondo negro, sin outliers) ---
st.subheader("Relación: MedInc (X) vs MedHouseVal (Y)")
if not df_f.empty:
    if USE_MPL:
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="black")
        ax2.set_facecolor("black")

        ax2.scatter(
            df_f["MedInc"], df_f["MedHouseVal"],
            s=10, alpha=0.7, color="#CCCCCC", label="Datos"
        )

        # Estética: ejes/labels/ticks blancos
        for spine in ax2.spines.values():
            spine.set_color("white")
        ax2.tick_params(colors="white")
        ax2.set_xlabel("MedInc (Mediana de Ingresos)", color="white")
        ax2.set_ylabel("MedHouseVal (Valor mediano vivienda)", color="white")
        ax2.set_title("Scatter: MedInc vs MedHouseVal", color="white")

        st.pyplot(fig2)
    else:
        fig2 = px.scatter(
            df_f, x="MedInc", y="MedHouseVal",
            opacity=0.8, title="Scatter: MedInc vs MedHouseVal",
            labels={"MedInc":"MedInc (Mediana de Ingresos)", "MedHouseVal":"Valor mediano vivienda"}
        )
        fig2.update_layout(template="plotly_dark", paper_bgcolor="black", plot_bgcolor="black", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Sin datos para graficar.")

# --- Mapa opcional ---
with st.expander("📍Mapa geográfico (Lat/Long)"):
    if not df_f.empty:
        df_map = df_f.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        try:
            st.map(df_map[["lat", "lon"]].sample(min(5000, len(df_map)), random_state=42))
        except Exception:
            st.map(df_map[["lat", "lon"]])
    else:
        st.info("Sin datos para mapear.")
