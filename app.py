# app.py
#%pip install streamlit pandas numpy matplotlib scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="California Housing Explorer", layout="wide")

# =========================
# Fase 1: Carga y Preparación
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    ds = fetch_california_housing(as_frame=False)
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="MedHouseVal")
    df = pd.concat([X, y], axis=1)
    return df
df_california = load_data()
st.title("California Housing – Explorador Interactivo")
st.caption("Fase 1: Carga y preparación de datos con Pandas")


# Check valores faltantes
st.subheader("Valores faltantes / vacíos (simple)")

# 1) NULOS (NaN o None en Pandas)
nulos_por_col = df_california.isna().sum()
st.write("Nulos por columna (isna):")
st.write(nulos_por_col)

# 2) None (explícito)
none_por_col = df_california.map(lambda x: x is None).sum()
st.write("Valores None por columna:")
st.write(none_por_col)

# 3) NaN (numérico)
nan_por_col = df_california.map(lambda x: isinstance(x, float) and np.isnan(x)).sum()
st.write("Valores NaN por columna:")
st.write(nan_por_col)

# 4) Strings vacíos ("", o solo espacios) – solo en columnas de texto
if df_california.select_dtypes(include=["object"]).shape[1] > 0:
    vacios_str = (
        df_california
        .select_dtypes(include=["object"])
        .apply(lambda s: s.fillna("").str.strip().eq("").sum())
    )
else:
    vacios_str = pd.Series(dtype=int)

st.write("Strings vacíos por columna (solo texto):")
st.write(vacios_str)

# Resumen rápido
st.write("¿Hay nulos (NaN/None)?", bool(df_california.isna().any().any()))
st.write("¿Hay strings vacíos?", bool(vacios_str.sum() > 0))

# Mostrar primeras filas + tipos
st.subheader("Vista rápida del DataFrame")
st.dataframe(df_california.head())
st.write("Tipos de datos por columna:")
st.write(df_california.dtypes.to_frame("dtype"))

# =========================
# Fase 2: Análisis Descriptivo Interactivo
# =========================
st.sidebar.markdown("## Controles")
st.sidebar.markdown(
    "Usá los filtros para acotar el conjunto de datos por "
    "**edad mediana de la casa** y **latitud mínima**."
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

# Checkbox + number_input para latitud mínima
use_lat_filter = st.sidebar.checkbox("Filtrar por Latitud mínima", value=False)
if use_lat_filter:
    lat_min_total = float(df_california["Latitude"].min())
    lat_max_total = float(df_california["Latitude"].max())
    lat_min = st.sidebar.number_input(
        "Latitude mínima",
        min_value=lat_min_total, max_value=lat_max_total,
        value=lat_min_total, step=0.5, format="%.2f"
    )
    df_f = df_f.loc[df_f["Latitude"] >= lat_min].copy()

st.subheader("Resumen descriptivo (tras filtros)")
if df_f.empty:
    st.warning("No hay filas tras aplicar los filtros.")
else:
    mediana = df_f["MedHouseVal"].median()
    rango = df_f["MedHouseVal"].max() - df_f["MedHouseVal"].min()
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas vigentes", f"{len(df_f):,}")
    c2.metric("Mediana MedHouseVal", f"{mediana:,.3f}")
    c3.metric("Rango (max - min)", f"{rango:,.3f}")

# =========================
# Fase 3: Visualización Dinámica
# =========================
st.header("Visualizaciones")

# Histograma del Target (MedHouseVal)
st.subheader("Distribución de MedHouseVal (tras filtros)")
if not df_f.empty:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(df_f["MedHouseVal"], bins=30)
    ax1.set_xlabel("MedHouseVal")
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Histograma de MedHouseVal")
    st.pyplot(fig1)
else:
    st.info("Sin datos para graficar.")

# Scatter MedInc vs MedHouseVal
st.subheader("Relación: MedInc (X) vs MedHouseVal (Y)")
if not df_f.empty:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(df_f["MedInc"], df_f["MedHouseVal"], s=8, alpha=0.6)
    ax2.set_xlabel("MedInc (Mediana de Ingresos)")
    ax2.set_ylabel("MedHouseVal (Valor mediano vivienda)")
    ax2.set_title("Scatter: MedInc vs MedHouseVal")
    st.pyplot(fig2)
else:
    st.info("Sin datos para graficar.")

# Opcional: mapa (lat/lon)
with st.expander("📍 Opcional: Mapa geográfico (Lat/Long)"):
    if not df_f.empty:
        df_map = df_f.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        try:
            st.map(df_map[["lat", "lon"]].sample(min(5000, len(df_map)), random_state=42))
        except Exception:
            st.map(df_map[["lat", "lon"]])
    else:
        st.info("Sin datos para mapear.")


