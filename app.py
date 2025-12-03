# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import streamlit.components.v1 as components  # <-- para el scroll

# Intento importar Matplotlib; si falla, uso Plotly
try:
    import matplotlib.pyplot as plt
    USE_MPL = True
except Exception:
    import plotly.express as px  # fallback
    USE_MPL = False

st.set_page_config(page_title="California Housing Explorer", layout="wide")

# --- CSS: separador visual y margen de scroll para encabezados ---
st.markdown("""
<style>
h2, h3 { scroll-margin-top: 80px; }  /* evita que el header quede tapado */
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data() -> pd.DataFrame:
    ds = fetch_california_housing(as_frame=False)
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="MedHouseVal")
    df = pd.concat([X, y], axis=1)
    return df

df_california = load_data()

st.title("California Housing – Explorador Interactivo")

# ========== SIDEBAR: Índice con scroll programático ==========
st.sidebar.markdown("### Índice")
if st.sidebar.button("Valores faltantes / vacíos", use_container_width=True):
    st.session_state["_scroll_to"] = "faltantes"
if st.sidebar.button("Vista rápida + dtypes", use_container_width=True):
    st.session_state["_scroll_to"] = "vista"
if st.sidebar.button("Controles y filtros", use_container_width=True):
    st.session_state["_scroll_to"] = "controles"
if st.sidebar.button("Resumen descriptivo", use_container_width=True):
    st.session_state["_scroll_to"] = "resumen"
if st.sidebar.button("Visualizaciones", use_container_width=True):
    st.session_state["_scroll_to"] = "vis"
if st.sidebar.button("Mapa geográfico", use_container_width=True):
    st.session_state["_scroll_to"] = "mapa"

def do_scroll():
    anchor = st.session_state.pop("_scroll_to", None)
    if not anchor:
        return
    components.html(f"""
    <script>
    const anchorId = "{anchor}";
    // Busca por id exacto
    let el = window.parent.document.getElementById(anchorId);
    // fallback: busca <h2>/<h3> con data-anchor (lo ponemos abajo)
    if (!el) {{
        el = window.parent.document.querySelector(`[data-anchor="${{anchorId}}"]`);
    }}
    if (el && el.scrollIntoView) {{
        el.scrollIntoView({{ behavior: "smooth", block: "start" }});
    }}
    </script>
    """, height=0)

# ========= Sección: Faltantes =========
st.markdown('<h2 id="faltantes" data-anchor="faltantes">Valores faltantes / vacíos (simple)</h2>', unsafe_allow_html=True)
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

# ========= Sección: Vista + dtypes =========
st.markdown('<h2 id="vista" data-anchor="vista">Vista rápida del DataFrame</h2>', unsafe_allow_html=True)
st.dataframe(df_california.head())
st.markdown('<h3>Tipos de datos por columna</h3>', unsafe_allow_html=True)
st.write(df_california.dtypes.to_frame("dtype"))

# ========= Sección: Controles (sidebar) =========
st.markdown('<h2 id="controles" data-anchor="controles">Controles y filtros</h2>', unsafe_allow_html=True)
st.sidebar.markdown("## Controles")
st.sidebar.markdown(
    "Usá los filtros para acotar por **HouseAge** y **Latitud mínima (vecindario)**."
)

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

# ========= Sección: Resumen =========
st.markdown('<h2 id="resumen" data-anchor="resumen">Resumen descriptivo (tras filtros)</h2>', unsafe_allow_html=True)
if df_f.empty:
    st.warning("No hay filas tras aplicar los filtros.")
else:
    mediana = df_f["MedHouseVal"].median()
    rango = df_f["MedHouseVal"].max() - df_f["MedHouseVal"].min()
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas vigentes", f"{len(df_f):,}")
    c2.metric("Mediana MedHouseVal", f"{mediana:,.3f}")
    c3.metric("Rango (max - min)", f"{rango:,.3f}")

# ========= Sección: Visualizaciones =========
st.markdown('<h2 id="vis" data-anchor="vis">Visualizaciones</h2>', unsafe_allow_html=True)

# Histograma (tema oscuro)
st.subheader("Distribución de MedHouseVal (tras filtros)")
if not df_f.empty:
    if USE_MPL:
        fig1, ax1 = plt.subplots(figsize=(6, 4), facecolor="black")
        ax1.set_facecolor("black")
        ax1.hist(df_f["MedHouseVal"], bins=30, color="#CCCCCC", edgecolor="#CCCCCC", alpha=0.75)
        for sp in ax1.spines.values():
            sp.set_color("white")
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

# Scatter (tema oscuro)
st.subheader("Relación: MedInc (X) vs MedHouseVal (Y)")
if not df_f.empty:
    if USE_MPL:
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="black")
        ax2.set_facecolor("black")
        ax2.scatter(df_f["MedInc"], df_f["MedHouseVal"], s=10, alpha=0.8, color="#CCCCCC")
        for sp in ax2.spines.values():
            sp.set_color("white")
        ax2.tick_params(colors="white")
        ax2.set_xlabel("MedInc (Mediana de Ingresos)", color="white")
        ax2.set_ylabel("MedHouseVal (Valor mediano vivienda)", color="white")
        ax2.set_title("Scatter: MedInc vs MedHouseVal", color="white")
        st.pyplot(fig2)
    else:
        fig2 = px.scatter(
            df_f, x="MedInc", y="MedHouseVal",
            opacity=0.85, title="Scatter: MedInc vs MedHouseVal",
            labels={"MedInc":"MedInc (Mediana de Ingresos)", "MedHouseVal":"Valor mediano vivienda"}
        )
        fig2.update_layout(template="plotly_dark", paper_bgcolor="black", plot_bgcolor="black", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Sin datos para graficar.")

# ========= Sección: Mapa =========
st.markdown('<h2 id="mapa" data-anchor="mapa">Mapa geográfico</h2>', unsafe_allow_html=True)
with st.expander("📍Mapa geográfico (Lat/Long)"):
    if not df_f.empty:
        df_map = df_f.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        try:
            st.map(df_map[["lat", "lon"]].sample(min(5000, len(df_map)), random_state=42))
        except Exception:
            st.map(df_map[["lat", "lon"]])
    else:
        st.info("Sin datos para mapear.")

# ========== Ejecuta el scroll si alguien tocó un botón del índice ==========
do_scroll()
