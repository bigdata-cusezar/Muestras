import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Control de Resistencia",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLES ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background-color: #0f1117; color: #e8eaf0; }

section[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a3045;
}

.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7db3;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #e8eaf0;
    font-family: 'DM Mono', monospace;
    line-height: 1.1;
}
.metric-sub {
    font-size: 12px;
    color: #4a90d9;
    margin-top: 4px;
    font-weight: 500;
}
.cumple { color: #2ecc71 !important; }
.nocumple { color: #e74c3c !important; }
.excelente { color: #2ecc71 !important; }
.muybueno { color: #27ae60 !important; }
.bueno { color: #f1c40f !important; }
.aceptable { color: #e67e22 !important; }
.pobre { color: #e74c3c !important; }

.section-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a90d9;
    margin: 24px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a3a5c;
}

.upload-area {
    background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
    border: 2px dashed #2a3a5c;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
}

div[data-testid="stPlotlyChart"] {
    background: #161b27;
    border-radius: 12px;
    border: 1px solid #2a3045;
    padding: 8px;
}

.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ────────────────────────────────────────────────────────────────
COLORS = {
    14:  "#4a90d9",
    28:  "#f39c12",
    56:  "#9b59b6",
}
CURVE_COLORS = {
    "Distribución Real": "#1a73e8",
    "Aceptable (Ds=50)":  "#e74c3c",
    "Bueno (Ds=45)":      "#e67e22",
    "Muy Bueno (Ds=37.5)":"#f1c40f",
    "Excelente (Ds=30)":  "#2ecc71",
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b27",
    plot_bgcolor="#161b27",
    font=dict(family="DM Sans", color="#e8eaf0", size=12),
    margin=dict(t=50, b=40, l=50, r=20),
    legend=dict(
        bgcolor="rgba(26,32,53,0.9)",
        bordercolor="#2a3a5c",
        borderwidth=1,
        font=dict(size=11)
    ),
    xaxis=dict(gridcolor="#1e2640", zerolinecolor="#2a3a5c"),
    yaxis=dict(gridcolor="#1e2640", zerolinecolor="#2a3a5c"),
)

def estandarizar_edad(edad):
    if pd.isna(edad): return None
    if edad < 14:     return None
    if edad < 28:     return 14
    if edad < 56:     return 28
    return 56

def calidad_cv(cv):
    if cv <= 0.03: return "Excelente",   "excelente"
    if cv <= 0.04: return "Muy Bueno",   "muybueno"
    if cv <= 0.05: return "Bueno",       "bueno"
    if cv <= 0.06: return "Aceptable",   "aceptable"
    return "Pobre", "pobre"

def calidad_ds(ds):
    if ds <= 25: return "Excelente",   "excelente"
    if ds <= 35: return "Muy Bueno",   "muybueno"
    if ds <= 40: return "Bueno",       "bueno"
    if ds <= 50: return "Aceptable",   "aceptable"
    return "Pobre", "pobre"

def extraer_nombre_proyecto(proyecto):
    if pd.isna(proyecto): return proyecto
    partes = str(proyecto).split(" - ")
    if len(partes) >= 3:
        return " - ".join(partes[1:-1])
    if len(partes) == 2:
        return partes[1].strip()
    return proyecto

def cargar_datos(archivo):
    ext = archivo.name.split(".")[-1].lower()
    if ext in ["xlsx", "xls"]:
        xls = pd.ExcelFile(archivo)
        hoja = xls.sheet_names[0]
        df = pd.read_excel(archivo, sheet_name=hoja, header=None)
        # Encontrar fila de encabezado
        for i, row in df.iterrows():
            if "Proyecto" in str(row.values):
                df.columns = df.iloc[i]
                df = df.iloc[i+1:].reset_index(drop=True)
                break
    else:
        df = pd.read_csv(archivo)

    # Limpiar columnas
    df.columns = [str(c).strip() for c in df.columns]
    col_res = [c for c in df.columns if "kg/cm" in c]
    col_edad = [c for c in df.columns if "Edad" in c and "días" in c]
    col_nominal = [c for c in df.columns if "nominal" in c.lower()]

    for col in col_res + col_edad + col_nominal:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fecha_cols = [c for c in df.columns if c in ["Toma", "Recepción", "Rotura"]]
    for col in fecha_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    edad_col = col_edad[0] if col_edad else "Edad (días)"
    df["Edad Estandar"] = df[edad_col].apply(estandarizar_edad)
    df["Nombre Proyecto"] = df["Proyecto"].apply(extraer_nombre_proyecto)
    df["Clave Muestra"] = df.apply(
        lambda r: f"{r['Tipo de mezcla']}-{r['Cilindro N°']}"
        if pd.notna(r.get("Cilindro N°")) else None, axis=1
    )
    return df

def get_res_col(df):
    cols = [c for c in df.columns if "kg/cm" in c]
    return cols[0] if cols else "Resistencia (kg/cm²)"

def get_nominal_col(df):
    cols = [c for c in df.columns if "nominal" in c.lower()]
    return cols[0] if cols else "Resistencia nominal (MPa)"

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Control de Resistencia")
    st.markdown("---")

    archivo = st.file_uploader(
        "Cargar archivo Excel",
        type=["xlsx", "xls", "csv"],
        help="El archivo debe tener una hoja con los datos de ensayos"
    )

    if archivo:
        df_raw = cargar_datos(archivo)
        proyectos = sorted(df_raw["Nombre Proyecto"].dropna().unique())
        tipos = sorted(df_raw["Tipo de mezcla"].dropna().unique())

        st.markdown('<div class="section-title">Filtros</div>', unsafe_allow_html=True)
        proyecto_sel = st.selectbox("Proyecto", proyectos)
        tipo_sel = st.selectbox("Tipo de mezcla", tipos)

        df = df_raw[
            (df_raw["Nombre Proyecto"] == proyecto_sel) &
            (df_raw["Tipo de mezcla"] == tipo_sel) &
            (df_raw["Edad Estandar"].notna())
        ].copy()

        st.markdown("---")
        st.caption(f"**{len(df)}** registros cargados")

# ─── MAIN ───────────────────────────────────────────────────────────────────
if not archivo:
    st.markdown("""
    <div class="upload-area">
        <h2 style="color:#4a90d9; margin-bottom:8px;">🏗️ Control de Resistencia</h2>
        <p style="color:#6b7db3; font-size:15px;">Carga tu archivo Excel con los datos de ensayos para comenzar</p>
        <p style="color:#4a5568; font-size:13px; margin-top:16px;">
            Columnas requeridas: Proyecto · OT · Cilindro N° · Tipo de mezcla · Localización<br>
            Toma · Edad (días) · Resistencia (kg/cm²) · Resistencia nominal (MPa)
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

res_col    = get_res_col(df)
nom_col    = get_nominal_col(df)
fc_nominal = df[nom_col].dropna().iloc[0] * 10.197 if not df[nom_col].dropna().empty else 125

df28 = df[df["Edad Estandar"] == 28][res_col].dropna()
prom28 = df28.mean()
ds    = df28.std(ddof=1)
n     = df["Clave Muestra"].nunique()
cv    = ds / prom28 if prom28 else 0
fcr1  = prom28 + 1.34 * ds
fcr2  = prom28 + 2.33 * ds - 35
fcr   = max(fcr1, fcr2) if fc_nominal <= 350 else max(fcr1, 0.9 * fc_nominal + 2.33 * ds)
cumple = prom28 >= fcr
cal_cv, cls_cv = calidad_cv(cv)
cal_ds, cls_ds = calidad_ds(ds)

# ─── TARJETAS ───────────────────────────────────────────────────────────────
st.markdown(f"### {proyecto_sel} · {tipo_sel}")
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)

def card(col, label, value, sub="", cls=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
        {"<div class='metric-sub'>"+sub+"</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)

card(c1, "f'c Nominal", f"{fc_nominal:.0f}", "kg/cm²")
card(c2, "Promedio 28d", f"{prom28:.1f}", "kg/cm²")
card(c3, "Desv. Estándar", f"{ds:.1f}", "kg/cm²", cls_ds)
card(c4, "Coef. Variación", f"{cv*100:.1f}%", cal_cv, cls_cv)
card(c5, "N° Muestras", str(n))
card(c6, "f'cr Diseño", f"{fcr:.1f}", "kg/cm²")
card(c7, "NSR-10", "Cumple" if cumple else "No Cumple", sub="", cls="cumple" if cumple else "nocumple")

st.markdown("<br>", unsafe_allow_html=True)

# ─── GRÁFICA 1: CONTROL DE RESISTENCIA ─────────────────────────────────────
st.markdown('<div class="section-title">Control de Resistencia</div>', unsafe_allow_html=True)

fig1 = go.Figure()
for edad in [14, 28, 56]:
    df_edad = df[df["Edad Estandar"] == edad].copy()
    if df_edad.empty: continue
    prom_por_cilindro = df_edad.groupby("Cilindro N°")[res_col].mean().reset_index()
    prom_por_cilindro = prom_por_cilindro.sort_values("Cilindro N°")
    fig1.add_trace(go.Scatter(
        x=prom_por_cilindro["Cilindro N°"],
        y=prom_por_cilindro[res_col],
        mode="lines+markers",
        name=f"{edad} días",
        line=dict(color=COLORS[edad], width=2),
        marker=dict(size=6),
    ))

prom_general = df[df["Edad Estandar"] == 28].groupby("Cilindro N°")[res_col].mean().mean()
fig1.add_hline(y=fc_nominal, line_dash="dot", line_color="#4a90d9",
               annotation_text=f"Resistencia Nominal: {fc_nominal:.0f}", annotation_font_color="#4a90d9")
fig1.add_hline(y=prom_general, line_dash="dot", line_color="#e74c3c",
               annotation_text=f"Promedio General: {prom_general:.2f}", annotation_font_color="#e74c3c")
fig1.update_layout(**PLOTLY_LAYOUT,
    xaxis_title="Cilindro N°", yaxis_title="Promedio Resistencia (kg/cm²)",
    height=380)
st.plotly_chart(fig1, use_container_width=True)

# ─── GRÁFICAS 2 Y 3 ─────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

# DISTRIBUCIÓN NORMAL
with col_a:
    st.markdown('<div class="section-title">Distribución Normal</div>', unsafe_allow_html=True)

    mu = df28.mean()
    x_rel = np.linspace(-250, 250, 500)
    x_abs = x_rel + mu
    freq_counts, freq_bins = np.histogram(df28 - mu, bins=np.arange(-250, 260, 10))
    bin_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(
        x=bin_centers, y=freq_counts,
        name="Frecuencia", marker_color="rgba(74,144,217,0.4)",
        marker_line_color="rgba(74,144,217,0.8)", marker_line_width=1,
    ), secondary_y=False)

    curvas = {
        "Distribución Real": ds,
        "Aceptable (Ds=50)":  50,
        "Bueno (Ds=45)":      45,
        "Muy Bueno (Ds=37.5)":37.5,
        "Excelente (Ds=30)":  30,
    }
    for nombre, sigma in curvas.items():
        pdf = stats.norm.pdf(x_rel, 0, sigma)
        fig2.add_trace(go.Scatter(
            x=x_rel, y=pdf, mode="lines", name=nombre,
            line=dict(color=CURVE_COLORS[nombre], width=2,
                      dash="solid" if nombre == "Distribución Real" else "dot"),
        ), secondary_y=True)

    fig2.update_layout(
        paper_bgcolor="#161b27",
        plot_bgcolor="#161b27",
        font=dict(family="DM Sans", color="#e8eaf0", size=12),
        margin=dict(t=50, b=80, l=50, r=20),
        height=380,
        xaxis_title="x relativo (kg/cm²)",
        legend=dict(
            orientation="h", y=-0.35, font=dict(size=10),
            bgcolor="rgba(26,32,53,0.9)",
            bordercolor="#2a3a5c", borderwidth=1
        )
    )
    fig2.update_yaxes(
        title_text="Frecuencia", gridcolor="#1e2640",
        zerolinecolor="#2a3a5c", secondary_y=False
    )
    fig2.update_yaxes(
        title_text="Densidad", gridcolor="#1e2640",
        zerolinecolor="#2a3a5c", secondary_y=True
    )
    fig2.update_xaxes(gridcolor="#1e2640", zerolinecolor="#2a3a5c")
    st.plotly_chart(fig2, use_container_width=True)

# REGRESIÓN LOGARÍTMICA
with col_b:
    st.markdown('<div class="section-title">Resistencia vs Tiempo</div>', unsafe_allow_html=True)

    promedios_edad = []
    for edad in [14, 28, 56]:
        vals = df[df["Edad Estandar"] == edad][res_col].dropna()
        if not vals.empty:
            promedios_edad.append({"Edad": edad, "Promedio": vals.mean()})
    df_evol = pd.DataFrame(promedios_edad)

    fig3 = go.Figure()
    if len(df_evol) >= 2:
        try:
            def log_func(x, a, b): return a * np.log(x) + b
            popt, _ = curve_fit(log_func, df_evol["Edad"], df_evol["Promedio"])
            x_curve = np.linspace(1, 70, 300)
            y_curve = log_func(x_curve, *popt)
            fig3.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode="lines",
                name=f"Regresión log: {popt[0]:.2f}·ln(t) + {popt[1]:.2f}",
                line=dict(color="#4a90d9", width=2, dash="dash"),
            ))
        except Exception:
            pass

    fig3.add_trace(go.Scatter(
        x=df_evol["Edad"], y=df_evol["Promedio"],
        mode="markers+text",
        name="Promedio por edad",
        marker=dict(size=12, color=[COLORS.get(e, "#fff") for e in df_evol["Edad"]],
                    line=dict(width=2, color="#fff")),
        text=[f"{v:.1f}" for v in df_evol["Promedio"]],
        textposition="top center",
        textfont=dict(size=11),
    ))
    fig3.add_hline(y=fc_nominal, line_dash="dot", line_color="#4a90d9",
                   annotation_text=f"f'c = {fc_nominal:.0f}", annotation_font_color="#4a90d9")
    fig3.update_layout(**PLOTLY_LAYOUT, height=380,
        xaxis=dict(tickvals=[14, 28, 56], title="Edad (días)", gridcolor="#1e2640"),
        yaxis=dict(title="Promedio Resistencia (kg/cm²)", gridcolor="#1e2640"),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─── TABLA DETALLE ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Detalle por Muestra</div>', unsafe_allow_html=True)

tabla_rows = []
for cil in sorted(df["Cilindro N°"].dropna().unique()):
    df_cil = df[df["Cilindro N°"] == cil]
    row = {"N°": int(cil)}
    loc = df_cil["Localización"].dropna().iloc[0] if not df_cil["Localización"].dropna().empty else ""
    row["Localización"] = loc
    fecha = df_cil["Toma"].dropna().iloc[0] if "Toma" in df_cil.columns and not df_cil["Toma"].dropna().empty else None
    row["Fecha Toma"] = fecha.strftime("%Y-%m-%d") if fecha and not pd.isna(fecha) else ""
    for edad in [14, 28, 56]:
        vals = df_cil[df_cil["Edad Estandar"] == edad][res_col].dropna()
        row[f"Prom {edad}d (kg/cm²)"] = round(vals.mean(), 1) if not vals.empty else None
    prom28_cil = row.get("Prom 28d (kg/cm²)")
    row["% f'c"] = f"{prom28_cil / fc_nominal * 100:.1f}%" if prom28_cil else None
    row["NSR-10"] = "✅ Cumple" if prom28_cil and prom28_cil >= fcr else ("❌ No Cumple" if prom28_cil else "—")
    tabla_rows.append(row)

df_tabla = pd.DataFrame(tabla_rows)
st.dataframe(df_tabla, use_container_width=True, hide_index=True,
             column_config={"NSR-10": st.column_config.TextColumn(width="medium")})

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Control Estadístico de Resistencia · NSR-10 / ACI 318")
