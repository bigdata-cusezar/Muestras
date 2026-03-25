import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Control de Resistencia",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── THEME DETECTION ────────────────────────────────────────────────────────
is_dark = st.get_option("theme.base") == "dark"

if is_dark:
    BG        = "#0f1117"
    BG2       = "#161b27"
    BORDER    = "#2a3045"
    TEXT      = "#e8eaf0"
    TEXT2     = "#6b7db3"
    ACCENT    = "#4a90d9"
    PLOT_BG   = "#161b27"
    PAPER_BG  = "#0f1117"
    GRID      = "#1e2640"
    ZERO_LINE = "#2a3a5c"
    LEG_BG    = "rgba(26,32,53,0.95)"
    HLINE_C   = "#4a90d9"
    HLINE2_C  = "#e74c3c"
else:
    BG        = "#f7f9fc"
    BG2       = "#ffffff"
    BORDER    = "#e0e6f0"
    TEXT      = "#1a2035"
    TEXT2     = "#8a9ab5"
    ACCENT    = "#2563eb"
    PLOT_BG   = "#f7f9fc"
    PAPER_BG  = "#ffffff"
    GRID      = "#e8edf5"
    ZERO_LINE = "#c7d2e8"
    LEG_BG    = "rgba(255,255,255,0.97)"
    HLINE_C   = "#2563eb"
    HLINE2_C  = "#dc2626"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background-color: {BG}; color: {TEXT}; }}
.metric-card {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 16px 20px; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    height: 130px; display: flex; flex-direction: column; justify-content: center;
}}
.metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.12); }}
.metric-label {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: {TEXT2}; margin-bottom: 6px;
}}
.metric-value {{
    font-size: 24px; font-weight: 700; color: {TEXT};
    font-family: 'DM Mono', monospace; line-height: 1.1;
}}
.metric-sub {{ font-size: 11px; color: {ACCENT}; margin-top: 4px; font-weight: 500; }}
.metric-reason {{
    font-size: 10px; color: {TEXT2}; margin-top: 6px;
    line-height: 1.4; font-style: italic;
}}
.cumple    {{ color: #16a34a !important; }}
.nocumple  {{ color: #dc2626 !important; }}
.excelente {{ color: #16a34a !important; }}
.muybueno  {{ color: #15803d !important; }}
.bueno     {{ color: #ca8a04 !important; }}
.aceptable {{ color: #ea580c !important; }}
.pobre     {{ color: #dc2626 !important; }}
.section-title {{
    font-size: 12px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: {ACCENT};
    margin: 20px 0 10px 0; padding-bottom: 6px;
    border-bottom: 2px solid {BORDER};
}}
div[data-testid="stPlotlyChart"] {{
    background: {BG2}; border-radius: 12px; border: 1px solid {BORDER};
    padding: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}}
.stDataFrame {{ border-radius: 8px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ────────────────────────────────────────────────────────────────
COLORS = {14: "#4a90d9", 28: "#f39c12", 56: "#9b59b6"}
CURVE_COLORS = {
    "Distribucion Real": "#1a73e8",
    "Aceptable":         "#e74c3c",
    "Bueno":             "#e67e22",
    "Muy Bueno":         "#f1c40f",
    "Excelente":         "#2ecc71",
}

def plotly_base():
    return dict(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=TEXT, size=12),
        margin=dict(t=50, b=40, l=50, r=20),
        legend=dict(bgcolor=LEG_BG, bordercolor=BORDER, borderwidth=1, font=dict(size=11, color=TEXT)),
    )

def estandarizar_edad(edad):
    if pd.isna(edad): return None
    if edad < 14:  return None
    if edad < 28:  return 14
    if edad < 56:  return 28
    return 56

def calidad_cv(cv):
    if cv <= 0.03: return "Excelente",  "excelente"
    if cv <= 0.04: return "Muy Bueno",  "muybueno"
    if cv <= 0.05: return "Bueno",      "bueno"
    if cv <= 0.06: return "Aceptable",  "aceptable"
    return "Pobre", "pobre"

def calidad_ds(ds):
    if ds <= 25: return "Excelente",  "excelente"
    if ds <= 35: return "Muy Bueno",  "muybueno"
    if ds <= 40: return "Bueno",      "bueno"
    if ds <= 50: return "Aceptable",  "aceptable"
    return "Pobre", "pobre"

def extraer_nombre_proyecto(proyecto):
    if pd.isna(proyecto): return proyecto
    partes = str(proyecto).split(" - ")
    if len(partes) >= 3: return " - ".join(partes[1:-1])
    if len(partes) == 2: return partes[1].strip()
    return proyecto

def get_cil_col(df):
    for c in df.columns:
        if "Cilindro" in c:
            return c
    return None

def get_toma_col(df):
    for c in df.columns:
        if "Toma" in c:
            return c
    return None

def get_loc_col(df):
    for c in df.columns:
        if "Localiz" in c:
            return c
    return None

def cargar_datos(archivo):
    ext = archivo.name.split(".")[-1].lower()
    if ext in ["xlsx", "xls"]:
        xls  = pd.ExcelFile(archivo)
        hoja = xls.sheet_names[0]
        df   = pd.read_excel(archivo, sheet_name=hoja, header=None)
        for i, row in df.iterrows():
            if "Proyecto" in str(row.values):
                df.columns = df.iloc[i]
                df = df.iloc[i+1:].reset_index(drop=True)
                break
    else:
        df = pd.read_csv(archivo)

    df.columns  = [str(c).strip() for c in df.columns]
    col_res     = [c for c in df.columns if "kg/cm" in c]
    col_edad    = [c for c in df.columns if "Edad" in c and "dias" in c.lower().replace("í", "i")]
    col_nominal = [c for c in df.columns if "nominal" in c.lower()]

    for col in col_res + col_edad + col_nominal:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [c for c in df.columns if c in ["Toma", "Recepcion", "Rotura", "Recepción"]]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    edad_col = col_edad[0] if col_edad else None
    if edad_col:
        df["Edad Estandar"] = df[edad_col].apply(estandarizar_edad)

    df["Nombre Proyecto"] = df["Proyecto"].apply(extraer_nombre_proyecto)

    cil_col = get_cil_col(df)
    if cil_col:
        df["Clave Muestra"] = df.apply(
            lambda r: f"{r['Tipo de mezcla']}-{r[cil_col]}"
            if pd.notna(r.get(cil_col)) else None, axis=1
        )
    return df

def get_res_col(df):
    cols = [c for c in df.columns if "kg/cm" in c]
    return cols[0] if cols else None

def get_nominal_col(df):
    cols = [c for c in df.columns if "nominal" in c.lower()]
    return cols[0] if cols else None

# ─── PANTALLA DE CARGA ────────────────────────────────────────────────────────
if "archivo_data" not in st.session_state:
    st.session_state.archivo_data = None

if st.session_state.archivo_data is None:
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{BG2}; border:2px dashed {BORDER}; border-radius:20px;
                    padding:48px 40px; text-align:center;
                    box-shadow:0 8px 32px rgba(0,0,0,0.08);">
            <div style="font-size:52px; margin-bottom:16px;">🏗️</div>
            <h2 style="color:{ACCENT}; margin:0 0 10px 0; font-size:26px; font-weight:700;">
                Control de Resistencia
            </h2>
            <p style="color:{TEXT2}; font-size:14px; margin:0 0 8px 0;">
                Carga tu archivo Excel o CSV con los datos de ensayos
            </p>
            <p style="color:{TEXT2}; font-size:11px; opacity:0.7; margin:0;">
                Proyecto · OT · Cilindro N° · Tipo de mezcla · Localizacion<br>
                Toma · Edad · Resistencia (kg/cm²) · Resistencia nominal (MPa)
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "📂 Seleccionar archivo",
            type=["xlsx", "xls", "csv"],
            label_visibility="visible",
            help="Excel o CSV con los datos de ensayos"
        )
        if uploaded:
            st.session_state.archivo_data = uploaded
            st.rerun()
    st.stop()

archivo = st.session_state.archivo_data

# ─── CARGA Y FILTROS ─────────────────────────────────────────────────────────
df_raw    = cargar_datos(archivo)
cil_col   = get_cil_col(df_raw)
toma_col  = get_toma_col(df_raw)
loc_col   = get_loc_col(df_raw)
res_col   = get_res_col(df_raw)
nom_col   = get_nominal_col(df_raw)
proyectos = sorted(df_raw["Nombre Proyecto"].dropna().unique())
tipos     = sorted(df_raw["Tipo de mezcla"].dropna().unique())

# Clave de agrupación para ensayos:
# Si existe columna Toma (fecha de muestreo), cada toma agrupa los 2-3 cilindros del ensayo.
# Si no, se usa el número de cilindro como clave individual (fallback).
toma_key = toma_col if toma_col else cil_col

st.markdown('<div class="section-title">Filtros</div>', unsafe_allow_html=True)
fc1, fc2, fc3, fc4 = st.columns([3, 2, 1, 1])
with fc1:
    proyecto_sel = st.selectbox("Proyecto", proyectos, label_visibility="visible")
with fc2:
    tipo_sel = st.selectbox("Tipo de mezcla", tipos, label_visibility="visible")
with fc3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"**{len(df_raw)}** registros totales")
with fc4:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Cambiar archivo"):
        st.session_state.archivo_data = None
        st.rerun()

df = df_raw[
    (df_raw["Nombre Proyecto"] == proyecto_sel) &
    (df_raw["Tipo de mezcla"]  == tipo_sel) &
    (df_raw["Edad Estandar"].notna())
].copy()

# ─── ESTADÍSTICOS ────────────────────────────────────────────────────────────
fc_nominal = df[nom_col].dropna().iloc[0] * 10 if nom_col and not df[nom_col].dropna().empty else 125

# Promedio a 28d: promedio de los promedios por ensayo (toma), no de cilindros individuales.
# Cada toma puede tener 2-3 cilindros; el promedio de esos cilindros es el resultado del ensayo.
if res_col and toma_key:
    df28_ensayos = (
        df[df["Edad Estandar"] == 28]
        .dropna(subset=[toma_key, res_col])
        .groupby(toma_key)[res_col]
        .mean()
    )
    df28   = df28_ensayos          # Serie: un valor por ensayo
    prom28 = df28.mean() if not df28.empty else 0
    ds     = df28.std(ddof=1) if len(df28) > 1 else 0
    # N = número de ensayos (tomas) con datos a 28d
    n      = len(df28)
else:
    df28   = pd.Series()
    prom28 = 0
    ds     = 0
    n      = 0

cv    = ds / prom28 if prom28 else 0
fcr1  = fc_nominal + 1.34 * ds
fcr2  = fc_nominal + 2.33 * ds - 35
fcr   = max(fcr1, fcr2) if fc_nominal <= 350 else max(fcr1, 0.9 * fc_nominal + 2.33 * ds)

cumple_global = prom28 >= fcr
umbral_nsr    = fc_nominal - 35 if fc_nominal <= 350 else fc_nominal * 0.9
cal_cv, cls_cv = calidad_cv(cv)
cal_ds, cls_ds = calidad_ds(ds)

if cumple_global:
    nsr_reason = f"x={prom28:.1f} >= f'cr={fcr:.1f}"
else:
    nsr_reason = f"x={prom28:.1f} < f'cr={fcr:.1f} (faltan {fcr-prom28:.1f})"

# ─── TARJETAS KPI ────────────────────────────────────────────────────────────
st.markdown(f"### {proyecto_sel} &nbsp;·&nbsp; {tipo_sel}", unsafe_allow_html=True)

def card(col, label, value, sub="", cls="", reason=""):
    reason_html = f"<div class='metric-reason'>{reason}</div>" if reason else ""
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
        {reason_html}
    </div>""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
card(c1, "f'c Nominal",      f"{fc_nominal:.0f}",  "kg/cm²")
card(c2, "Promedio 28d",     f"{prom28:.1f}",       "kg/cm²")
card(c3, "Desv. Estandar",   f"{ds:.1f}",           cal_ds, cls_ds)
card(c4, "Coef. Variacion",  f"{cv*100:.1f}%",      cal_cv, cls_cv)
card(c5, "N Ensayos (Tomas)", str(n))
card(c6, "f'cr Diseno",      f"{fcr:.1f}",          "kg/cm²")
card(c7, "NSR-10 Global",
    "Cumple" if cumple_global else "No Cumple",
    sub="x vs f'cr estadistico",
    cls="cumple" if cumple_global else "nocumple",
    reason=nsr_reason)

st.markdown("<br>", unsafe_allow_html=True)

# ─── GRÁFICA 1: CONTROL DE RESISTENCIA ──────────────────────────────────────
# Unidad de análisis: ENSAYO (Toma).
# Cada punto = promedio de los cilindros de esa toma a una edad dada.
st.markdown('<div class="section-title">Control de Resistencia por Ensayo</div>', unsafe_allow_html=True)

# Tabla de ensayos: promedio por (toma_key, Edad Estandar)
df_ensayos = (
    df.dropna(subset=[toma_key, res_col, "Edad Estandar"])
    .groupby([toma_key, "Edad Estandar"], sort=True)[res_col]
    .mean()
    .reset_index()
    .rename(columns={res_col: "Prom Ensayo"})
    .sort_values(toma_key)
)

# Orden cronológico de tomas (o numérico si es número de cilindro)
toma_order  = list(dict.fromkeys(df_ensayos[toma_key].tolist()))  # orden de aparición
toma_to_idx = {t: i for i, t in enumerate(toma_order)}
df_ensayos["x_pos"] = df_ensayos[toma_key].map(toma_to_idx)

# Etiqueta del eje X: cilindro menor de cada toma (para referencia), o el valor de la toma
def label_toma(t):
    if cil_col:
        cils = df[df[toma_key] == t][cil_col].dropna()
        if not cils.empty:
            return str(int(cils.min()))
    # Si la toma es una fecha, formatear; si es número, mostrar tal cual
    if hasattr(t, "strftime"):
        return t.strftime("%d/%m")
    return str(t)

etiquetas_x = [label_toma(t) for t in toma_order]
indices_x   = list(range(len(toma_order)))

fig1 = go.Figure()

for edad in [14, 28, 56]:
    sub_edad     = df_ensayos[df_ensayos["Edad Estandar"] == edad]
    tomas_en_edad = set(sub_edad[toma_key])

    y_vals, x_pos, hover_texts = [], [], []

    for toma in toma_order:
        idx = toma_to_idx[toma]
        if toma in tomas_en_edad:
            val = sub_edad.loc[sub_edad[toma_key] == toma, "Prom Ensayo"].iloc[0]
            # Cuántos cilindros forman este ensayo a esta edad
            n_cil_ensayo = len(df[
                (df[toma_key] == toma) & (df["Edad Estandar"] == edad)
            ])
            lbl = label_toma(toma)
            y_vals.append(val)
            x_pos.append(idx)
            hover_texts.append(
                f"<b>Ensayo — {lbl}</b><br>"
                f"Edad: {int(edad)} días<br>"
                f"Promedio ensayo: {val:.1f} kg/cm²<br>"
                f"Cilindros promediados: {n_cil_ensayo}<br>"
                f"% f'c: {val/fc_nominal*100:.1f}%"
            )
        else:
            y_vals.append(None)
            x_pos.append(idx)
            lbl = label_toma(toma)
            hover_texts.append(f"<b>Ensayo — {lbl}</b><br>{int(edad)}d: Sin dato")

    fig1.add_trace(go.Scatter(
        x=x_pos, y=y_vals,
        mode="lines+markers", name=f"{int(edad)} días",
        line=dict(color=COLORS[edad], width=2, shape="spline", smoothing=0.5),
        marker=dict(size=6, color=COLORS[edad]),
        connectgaps=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ))

prom_general = df_ensayos[df_ensayos["Edad Estandar"] == 28]["Prom Ensayo"].mean()

_fc_pos   = "right top"    if fc_nominal >= prom_general else "right bottom"
_prom_pos = "right bottom" if fc_nominal >= prom_general else "right top"

fig1.add_hline(
    y=fc_nominal, line_dash="dot", line_color=HLINE_C,
    annotation_text=f"f'c: {fc_nominal:.0f}",
    annotation_font_color=HLINE_C,
    annotation_position=_fc_pos,
)
fig1.add_hline(
    y=prom_general, line_dash="dot", line_color=HLINE2_C,
    annotation_text=f"Prom. General: {prom_general:.2f}",
    annotation_font_color=HLINE2_C,
    annotation_position=_prom_pos,
)

n_ensayos = len(toma_order)
if n_ensayos <= 40:    step = 1
elif n_ensayos <= 80:  step = 2
elif n_ensayos <= 160: step = 5
else:                  step = 10

tick_vals_show = [indices_x[i] for i in range(0, n_ensayos, step)]
tick_text_show = [etiquetas_x[i] for i in range(0, n_ensayos, step)]

fig1.update_layout(**plotly_base(),
    xaxis=dict(
        title="Ensayo (etiqueta = cil. menor de la toma)",
        gridcolor=GRID, zerolinecolor=ZERO_LINE,
        tickmode="array", tickvals=tick_vals_show, ticktext=tick_text_show,
        tickangle=0,
    ),
    yaxis=dict(title="Promedio Resistencia por Ensayo (kg/cm²)", gridcolor=GRID, zerolinecolor=ZERO_LINE),
    height=420,
    hovermode="x unified",
)
st.plotly_chart(fig1, use_container_width=True)

# ─── GRÁFICAS 2 Y 3 ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-title">Distribucion Normal</div>', unsafe_allow_html=True)
    # La distribución se construye sobre los promedios de ensayo (no cilindros individuales)
    mu    = df28.mean() if not df28.empty else 0
    x_rel = np.linspace(-250, 250, 500)
    freq_counts, freq_bins = np.histogram(df28 - mu, bins=np.arange(-250, 260, 10))
    bin_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(
        x=bin_centers, y=freq_counts, name="Frecuencia",
        marker_color="rgba(74,144,217,0.35)",
        marker_line_color="rgba(74,144,217,0.7)", marker_line_width=1,
        hovertemplate="Intervalo: %{x:.0f}<br>Frecuencia: %{y}<extra></extra>",
    ), secondary_y=False)

    for nombre, sigma in {
        "Distribucion Real": ds if ds > 0 else 1,
        "Aceptable":         50,
        "Bueno":             45,
        "Muy Bueno":         37.5,
        "Excelente":         30,
    }.items():
        pdf = stats.norm.pdf(x_rel, 0, sigma)
        fig2.add_trace(go.Scatter(
            x=x_rel, y=pdf, mode="lines", name=nombre,
            line=dict(color=CURVE_COLORS.get(nombre, ACCENT), width=2,
                      dash="solid" if "Real" in nombre else "dot"),
            hovertemplate=f"{nombre}<br>x: %{{x:.1f}}<br>Densidad: %{{y:.5f}}<extra></extra>",
        ), secondary_y=True)

    fig2.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=TEXT, size=12),
        margin=dict(t=50, b=80, l=50, r=20), height=400,
        xaxis_title="x relativo (kg/cm²)",
        legend=dict(orientation="h", y=-0.38, font=dict(size=10, color=TEXT),
                    bgcolor=LEG_BG, bordercolor=BORDER, borderwidth=1),
    )
    fig2.update_yaxes(title_text="Frecuencia", gridcolor=GRID, zerolinecolor=ZERO_LINE, secondary_y=False)
    fig2.update_yaxes(title_text="Densidad",   gridcolor=GRID, zerolinecolor=ZERO_LINE, secondary_y=True)
    fig2.update_xaxes(gridcolor=GRID, zerolinecolor=ZERO_LINE)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown('<div class="section-title">Resistencia vs Tiempo</div>', unsafe_allow_html=True)

    # Promedio por edad estandarizada (promedio de promedios de ensayo)
    promedios_edad = []
    for edad in [14, 28, 56]:
        sub = df_ensayos[df_ensayos["Edad Estandar"] == edad]["Prom Ensayo"].dropna()
        if not sub.empty:
            promedios_edad.append({"Edad": edad, "Promedio": sub.mean()})
    df_evol = pd.DataFrame(promedios_edad)

    fig3 = go.Figure()
    if len(df_evol) >= 2:
        try:
            def log_func(x, a, b): return a * np.log(x) + b
            popt, _ = curve_fit(log_func, df_evol["Edad"], df_evol["Promedio"])
            x_curve = np.linspace(1, 70, 300)
            y_curve = log_func(x_curve, *popt)
            primer_val = df_evol["Promedio"].iloc[0]
            ultimo_val = df_evol["Promedio"].iloc[-1]
            rango_pct  = abs(ultimo_val - primer_val) / max(primer_val, ultimo_val) * 0.5
            y_upper    = y_curve * (1 + rango_pct)
            y_lower    = y_curve * (1 - rango_pct)

            fig3.add_trace(go.Scatter(
                x=np.concatenate([x_curve, x_curve[::-1]]),
                y=np.concatenate([y_upper, y_lower[::-1]]),
                fill="toself", fillcolor="rgba(74,144,217,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Rango ±", hoverinfo="skip",
            ))
            fig3.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode="lines", name="Regresion log",
                line=dict(color=HLINE_C, width=2, dash="dash"),
                hovertemplate="t=%{x:.0f}d<br>f(t)=%{y:.1f} kg/cm²<extra></extra>",
            ))
            eq_display = (
                f"<i>f</i>(t) = {popt[0]:.2f} · ln(t)"
                f" {'+ ' if popt[1] >= 0 else '− '}{abs(popt[1]):.2f}"
            )
            fig3.add_annotation(
                xref="paper", yref="paper", x=0.98, y=0.04,
                text=eq_display, showarrow=False,
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=12, color=HLINE_C, family="DM Mono"),
                xanchor="right", yanchor="bottom",
            )
        except Exception:
            pass

    fig3.add_trace(go.Scatter(
        x=df_evol["Edad"], y=df_evol["Promedio"],
        mode="markers+text", name="Promedio por edad",
        marker=dict(size=12, color=[COLORS.get(e, ACCENT) for e in df_evol["Edad"]],
                    line=dict(width=2, color=TEXT)),
        text=[f"{v:.1f}" for v in df_evol["Promedio"]],
        textposition="top center",
        textfont=dict(size=11, color=TEXT),
        hovertemplate="Edad: %{x}d<br>Promedio: %{y:.1f} kg/cm²<extra></extra>",
    ))
    fig3.add_hline(y=fc_nominal, line_dash="dot", line_color=HLINE_C,
                   annotation_text=f"f'c = {fc_nominal:.0f}",
                   annotation_font_color=HLINE_C)
    fig3.update_layout(**plotly_base(), height=420, showlegend=False)
    fig3.update_xaxes(tickvals=[14, 28, 56], title_text="Edad (días)", gridcolor=GRID, zerolinecolor=ZERO_LINE)
    fig3.update_yaxes(title_text="Promedio Resistencia (kg/cm²)", gridcolor=GRID, zerolinecolor=ZERO_LINE)
    st.plotly_chart(fig3, use_container_width=True)

# ─── TABLA DETALLE ───────────────────────────────────────────────────────────
# Cada fila = un ENSAYO (Toma).
# La resistencia reportada es el PROMEDIO de los cilindros de esa toma.
# La verificación NSR-10 individual se hace sobre ese promedio, no sobre cilindros sueltos.
st.markdown('<div class="section-title">Detalle por Ensayo (Toma)</div>', unsafe_allow_html=True)

n_no_cumple = 0
tabla_rows  = []

for toma in sorted(df[toma_key].dropna().unique()):
    try:
        df_toma = df[df[toma_key] == toma]
        lbl_toma = label_toma(toma)
        row = {"Ensayo (Toma)": lbl_toma}

        if loc_col:
            locs = df_toma[loc_col].dropna()
            row["Localización"] = locs.iloc[0] if not locs.empty else ""

        # Fecha de toma (si la columna es fecha)
        if toma_col and toma_col in df_toma.columns:
            fechas = df_toma[toma_col].dropna()
            if not fechas.empty:
                fecha = fechas.iloc[0]
                row["Fecha Toma"] = fecha.strftime("%Y-%m-%d") if hasattr(fecha, "strftime") and pd.notna(fecha) else ""
            else:
                row["Fecha Toma"] = ""

        # Cilindros que componen esta toma
        if cil_col:
            cils = df_toma[cil_col].dropna()
            row["Cilindros"] = ", ".join(str(int(c)) for c in sorted(cils.unique()))

        # Promedio por edad: media de todos los cilindros de esta toma a esa edad
        for edad in [14, 28, 56]:
            vals = df_toma[df_toma["Edad Estandar"] == edad][res_col].dropna()
            if not vals.empty:
                row[f"Prom {int(edad)}d (kg/cm²)"] = round(float(vals.mean()), 1)
                row[f"N cil {int(edad)}d"]          = len(vals)
            else:
                row[f"Prom {int(edad)}d (kg/cm²)"] = None
                row[f"N cil {int(edad)}d"]          = 0

        prom28_toma = row.get("Prom 28d (kg/cm²)")
        row["% f'c"] = f"{prom28_toma / fc_nominal * 100:.1f}%" if prom28_toma else None

        # Verificación NSR-10 individual: el promedio del ensayo debe ser > umbral
        if prom28_toma:
            cumple_toma = prom28_toma > umbral_nsr
            row["NSR-10"] = "✅ Cumple" if cumple_toma else "❌ No Cumple"
            if not cumple_toma:
                n_no_cumple += 1
        else:
            row["NSR-10"] = "—"

        tabla_rows.append(row)
    except Exception:
        continue

df_tabla    = pd.DataFrame(tabla_rows)
total_con28 = sum(1 for r in tabla_rows if r.get("Prom 28d (kg/cm²)"))
pct_cumple  = (total_con28 - n_no_cumple) / total_con28 * 100 if total_con28 else 0

m1, m2, m3 = st.columns(3)
m1.metric("Umbral individual NSR-10", f"{umbral_nsr:.0f} kg/cm²",
          delta=f"f'c {'− 35' if fc_nominal <= 350 else '× 0.9'}")
m2.metric("Ensayos que No Cumplen", str(n_no_cumple),
          delta=f"de {total_con28} ensayos con datos a 28d", delta_color="inverse")
m3.metric("% Cumplimiento por ensayo", f"{pct_cumple:.1f}%")

st.dataframe(df_tabla, use_container_width=True, hide_index=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Control Estadístico de Resistencia · NSR-10 / ACI 318")
