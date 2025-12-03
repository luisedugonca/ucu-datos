# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Intento importar matplotlib; si no está, uso plotly, me daba error
try:
    import matplotlib.pyplot as plt
    USE_MPL = True
except Exception:
    import plotly.express as px
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
# Valores faltantes (versión simple)
valor = int(df_california.isna().sum().sum())
st.subheader("Valores faltantes / vacíos (simple)")
st.markdown(f'Nulos por columna (isna): "{valor}"')

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

st.subheader("Vista rápida del DataFrame")
st.dataframe(df_california.head())
st.write("Tipos de datos por columna:")
st.write(df_california.dtypes.to_frame("dtype"))

# ========= Fase 2 =========
st.sidebar.markdown("## Controles")
st.sidebar.markdown(
    "Usa los filtros para acotar por **HouseAge** y **latitud mínima**."
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
use_lat_filter = st.sidebar.checkbox("Filtrar por vecindario", value=False)
if use_lat_filter:
    lat_min_total = float(df_california["Latitude"].min())
    lat_max_total = float(df_california["Latitude"].max())
    lat_min = st.sidebar.number_input(
        "Latitud mínima",
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

# ========= Fase 3 =========
st.header("Visualizaciones")

#helpers
def iqr_outliers(s: pd.Series):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    return mask, lower, upper

# Histograma del Target (fondo negro + outliers)
st.subheader("Distribución de MedHouseVal")
if not df_f.empty:
    mask_y, lower, upper = iqr_outliers(df_f["MedHouseVal"])

    if USE_MPL:
        fig1, ax1 = plt.subplots(figsize=(6, 4), facecolor="black")
        ax1.set_facecolor("black")
        ax1.hist(df_f["MedHouseVal"], bins=30, color="#CCCCCC", edgecolor="#CCCCCC", alpha=0.35)

        # Líneas de umbral IQR
        ax1.axvline(lower, color="white", linestyle="--", linewidth=1)
        ax1.axvline(upper, color="white", linestyle="--", linewidth=1)

        # Rug de outliers sobre el eje X
        if mask_y.any():
            ax1.plot(
                df_f.loc[mask_y, "MedHouseVal"],
                [0] * mask_y.sum(),
                "|", color="#FF4136", markersize=12, label="Outliers"
            )

        # Estética: ejes/labels/ticks blancos
        for spine in ax1.spines.values():
            spine.set_color("white")
        ax1.tick_params(colors="white")
        ax1.set_xlabel("MedHouseVal", color="white")
        ax1.set_ylabel("Frecuencia", color="white")
        ax1.set_title("Histograma de MedHouseVal (IQR outliers)", color="white")
        if mask_y.any():
            leg = ax1.legend()
            for text in leg.get_texts():
                text.set_color("white")
        st.pyplot(fig1)

    else:
        import plotly.express as px
        fig1 = px.histogram(df_f, x="MedHouseVal", nbins=30, title="Histograma de MedHouseVal (IQR outliers)")
        # Fondo negro + textos blancos
        fig1.update_layout(template="plotly_dark", paper_bgcolor="black", plot_bgcolor="black", font_color="white")
        # Líneas IQR
        fig1.add_vline(x=lower, line_dash="dash", line_color="white")
        fig1.add_vline(x=upper, line_dash="dash", line_color="white")
        # Rug/markers para outliers en y=0
        if mask_y.any():
            fig1.add_scatter(
                x=df_f.loc[mask_y, "MedHouseVal"], y=[0]*mask_y.sum(),
                mode="markers", marker=dict(color="#FF4136", size=8),
                name="Outliers"
            )
        st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Sin datos para graficar.")

# Scatter MedInc vs MedHouseVal (fondo negro + outliers)
st.subheader("Relación: MedInc (X) vs MedHouseVal (Y)")
if not df_f.empty:
    mask_y, lower, upper = iqr_outliers(df_f["MedHouseVal"])

    if USE_MPL:
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="black")
        ax2.set_facecolor("black")

        # Puntos normales (gris claro) y outliers (rojo)
        ax2.scatter(
            df_f.loc[~mask_y, "MedInc"], df_f.loc[~mask_y, "MedHouseVal"],
            s=8, alpha=0.6, color="#CCCCCC", label="Datos"
        )
        if mask_y.any():
            ax2.scatter(
                df_f.loc[mask_y, "MedInc"], df_f.loc[mask_y, "MedHouseVal"],
                s=20, alpha=0.9, color="#FF4136", label="Outliers"
            )

        # Estética: ejes/labels/ticks blancos
        for spine in ax2.spines.values():
            spine.set_color("white")
        ax2.tick_params(colors="white")
        ax2.set_xlabel("MedInc (Mediana de Ingresos)", color="white")
        ax2.set_ylabel("MedHouseVal (Valor mediano vivienda)", color="white")
        ax2.set_title("Scatter: MedInc vs MedHouseVal", color="white")
        leg = ax2.legend()
        for text in leg.get_texts():
            text.set_color("white")
        st.pyplot(fig2)

    else:
        import plotly.express as px
        df_plot = df_f.copy()
        df_plot["outlier"] = np.where(mask_y, "Outlier", "Dato")

        fig2 = px.scatter(
            df_plot, x="MedInc", y="MedHouseVal",
            color="outlier", opacity=0.8,
            title="Scatter: MedInc vs MedHouseVal",
            labels={"MedInc":"MedInc (Mediana de Ingresos)", "MedHouseVal":"Valor mediano vivienda"},
            color_discrete_map={"Dato":"#CCCCCC", "Outlier":"#FF4136"}
        )
        fig2.update_layout(template="plotly_dark", paper_bgcolor="black", plot_bgcolor="black", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Sin datos para graficar.")


# Opcional: mapa
with st.expander("📍Mapa geográfico (Lat/Long)"):
    if not df_f.empty:
        df_map = df_f.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        try:
            st.map(df_map[["lat", "lon"]].sample(min(5000, len(df_map)), random_state=42))
        except Exception:
            st.map(df_map[["lat", "lon"]])
    else:
        st.info("Sin datos para mapear.")


