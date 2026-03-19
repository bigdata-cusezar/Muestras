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
section[data-testid="stSidebar"] {{ background: {BG2}; border-right: 1px solid {BORDER}; }}
.metric-card {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 16px 20px; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 100%;
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
.upload-area {{
    background: {BG2}; border: 2px dashed {BORDER};
    border-radius: 16px; padding: 48px 40px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
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
    "Distribucion Real":   "#1a73e8",
    "Aceptable (Ds=50)":   "#e74c3c",
    "Bueno (Ds=45)":       "#e67e22",
    "Muy Bueno (Ds=37.5)": "#f1c40f",
    "Excelente (Ds=30)":   "#2ecc71",
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

    df.columns      = [str(c).strip() for c in df.columns]
    col_res         = [c for c in df.columns if "kg/cm" in c]
    col_edad        = [c for c in df.columns if "Edad" in c and "dias" in c.lower().replace("í","i")]
    col_nominal     = [c for c in df.columns if "nominal" in c.lower()]

    for col in col_res + col_edad + col_nominal:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [c for c in df.columns if c in ["Toma", "Recepcion", "Rotura", "Recepción"]]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    edad_col             = col_edad[0] if col_edad else "Edad (dias)"
    df["Edad Estandar"]  = df[edad_col].apply(estandarizar_edad)
    df["Nombre Proyecto"]= df["Proyecto"].apply(extraer_nombre_proyecto)
    df["Clave Muestra"]  = df.apply(
        lambda r: f"{r['Tipo de mezcla']}-{r['Cilindro N']}"
        if pd.notna(r.get("Cilindro N")) else None, axis=1
    )
    return df

def get_res_col(df):
    cols = [c for c in df.columns if "kg/cm" in c]
    return cols[0] if cols else "Resistencia (kg/cm2)"

def get_nominal_col(df):
    cols = [c for c in df.columns if "nominal" in c.lower()]
    return cols[0] if cols else "Resistencia nominal (MPa)"

def get_cil_col(df):
    for c in df.columns:
        if "Cilindro" in c:
            return c
    return "Cilindro N"

def get_toma_col(df):
    for c in df.columns:
        if "Toma" in c:
            return c
    return None

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Control de Resistencia")
    st.markdown("---")
    archivo = st.file_uploader(
        "Cargar archivo Excel / CSV",
        type=["xlsx", "xls", "csv"],
        help="Debe contener los datos de ensayos de resistencia"
    )
    if archivo:
        st.markdown("---")
        st.caption("Usa los filtros en el dashboard para navegar.")

# ─── BIENVENIDA ──────────────────────────────────────────────────────────────
if not archivo:
    st.markdown(f"""
    <div class="upload-area">
        <h2 style="color:{ACCENT}; margin-bottom:8px;">🏗️ Control de Resistencia</h2>
        <p style="color:{TEXT2}; font-size:15px;">
            Carga tu archivo Excel con los datos de ensayos para comenzar
        </p>
        <p style="color:{TEXT2}; font-size:12px; margin-top:16px; opacity:0.7;">
            Columnas requeridas: Proyecto · OT · Cilindro N° · Tipo de mezcla · Localizacion<br>
            Toma · Edad (dias) · Resistencia (kg/cm2) · Resistencia nominal (MPa)
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── CARGA ───────────────────────────────────────────────────────────────────
df_raw    = cargar_datos(archivo)
cil_col   = get_cil_col(df_raw)
toma_col  = get_toma_col(df_raw)
proyectos = sorted(df_raw["Nombre Proyecto"].dropna().unique())
tipos     = sorted(df_raw["Tipo de mezcla"].dropna().unique())

# ─── FILTROS INLINE ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Filtros</div>', unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns([3, 2, 2])
with fc1:
    proyecto_sel = st.selectbox("Proyecto", proyectos)
with fc2:
    tipo_sel = st.selectbox("Tipo de mezcla", tipos)
with fc3:
    st.caption(f"**{len(df_raw)}** registros en el archivo")

df = df_raw[
    (df_raw["Nombre Proyecto"] == proyecto_sel) &
    (df_raw["Tipo de mezcla"]  == tipo_sel) &
    (df_raw["Edad Estandar"].notna())
].copy()

# ─── ESTADÍSTICOS ────────────────────────────────────────────────────────────
res_col    = get_res_col(df)
nom_col    = get_nominal_col(df)
fc_nominal = df[nom_col].dropna().iloc[0] * 10 if not df[nom_col].dropna().empty else 125

df28   = df[df["Edad Estandar"] == 28][res_col].dropna()
prom28 = df28.mean()
ds     = df28.std(ddof=1)
n      = df["Clave Muestra"].nunique()
cv     = ds / prom28 if prom28 else 0
fcr1   = fc_nominal + 1.34 * ds
fcr2   = fc_nominal + 2.33 * ds - 35
fcr    = max(fcr1, fcr2) if fc_nominal <= 350 else max(fcr1, 0.9 * fc_nominal + 2.33 * ds)
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

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
card(c1, "f'c Nominal",     f"{fc_nominal:.0f}",  "kg/cm2")
card(c2, "Promedio 28d",    f"{prom28:.1f}",       "kg/cm2")
card(c3, "Desv. Estandar",  f"{ds:.1f}",           cal_ds, cls_ds)
card(c4, "Coef. Variacion", f"{cv*100:.1f}%",      cal_cv, cls_cv)
card(c5, "N Muestras",      str(n))
card(c6, "f'cr Diseno",     f"{fcr:.1f}",          "kg/cm2")
card(c7, "NSR-10 Global",
    "Cumple" if cumple_global else "No Cumple",
    sub="x vs f'cr estadistico",
    cls="cumple" if cumple_global else "nocumple",
    reason=nsr_reason)

st.markdown("<br>", unsafe_allow_html=True)

# ─── GRÁFICA 1: CONTROL DE RESISTENCIA ──────────────────────────────────────
st.markdown('<div class="section-title">Control de Resistencia</div>', unsafe_allow_html=True)

todos_cilindros = sorted(df[cil_col].dropna().unique())
fig1 = go.Figure()

for edad in [14, 28, 56]:
    df_edad = df[df["Edad Estandar"] == edad].copy()
    if df_edad.empty:
        continue
    prom_cil = df_edad.groupby(cil_col)[res_col].mean().reindex(todos_cilindros)

    hover_texts = []
    for cil in todos_cilindros:
        val = prom_cil.get(cil)
        if pd.isna(val):
            hover_texts.append(f"<b>Cilindro {int(cil)}</b><br>{edad}d: Sin dato")
        else:
            hover_texts.append(
                f"<b>Cilindro {int(cil)}</b><br>"
                f"Edad: {edad} dias<br>"
                f"Resistencia: {val:.1f} kg/cm2<br>"
                f"% f'c: {val/fc_nominal*100:.1f}%"
            )

    fig1.add_trace(go.Scatter(
        x=todos_cilindros,
        y=prom_cil.values,
        mode="lines+markers",
        name=f"{edad} dias",
        line=dict(color=COLORS[edad], width=2),
        marker=dict(size=6, color=COLORS[edad]),
        connectgaps=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ))

prom_general = df[df["Edad Estandar"] == 28].groupby(cil_col)[res_col].mean().mean()
fig1.add_hline(y=fc_nominal, line_dash="dot", line_color=HLINE_C,
               annotation_text=f"f'c: {fc_nominal:.0f}", annotation_font_color=HLINE_C)
fig1.add_hline(y=prom_general, line_dash="dot", line_color=HLINE2_C,
               annotation_text=f"Prom. General: {prom_general:.2f}", annotation_font_color=HLINE2_C)
fig1.update_layout(**plotly_base(),
    xaxis=dict(title="Cilindro N", gridcolor=GRID, zerolinecolor=ZERO_LINE),
    yaxis=dict(title="Promedio Resistencia (kg/cm2)", gridcolor=GRID, zerolinecolor=ZERO_LINE),
    height=380,
    hovermode="x unified",
)
st.plotly_chart(fig1, use_container_width=True)

# ─── GRÁFICAS 2 Y 3 ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-title">Distribucion Normal</div>', unsafe_allow_html=True)
    mu = df28.mean()
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
        "Distribucion Real":   ds,
        "Aceptable (Ds=50)":   50,
        "Bueno (Ds=45)":       45,
        "Muy Bueno (Ds=37.5)": 37.5,
        "Excelente (Ds=30)":   30,
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
        margin=dict(t=50, b=80, l=50, r=20), height=380,
        xaxis_title="x relativo (kg/cm2)",
        legend=dict(orientation="h", y=-0.38, font=dict(size=10, color=TEXT),
                    bgcolor=LEG_BG, bordercolor=BORDER, borderwidth=1),
    )
    fig2.update_yaxes(title_text="Frecuencia", gridcolor=GRID, zerolinecolor=ZERO_LINE, secondary_y=False)
    fig2.update_yaxes(title_text="Densidad",   gridcolor=GRID, zerolinecolor=ZERO_LINE, secondary_y=True)
    fig2.update_xaxes(gridcolor=GRID, zerolinecolor=ZERO_LINE)
    st.plotly_chart(fig2, use_container_width=True)

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

            primer_val = df_evol["Promedio"].iloc[0]
            ultimo_val = df_evol["Promedio"].iloc[-1]
            rango_pct  = abs(ultimo_val - primer_val) / max(primer_val, ultimo_val) * 0.5
            y_upper    = y_curve * (1 + rango_pct)
            y_lower    = y_curve * (1 - rango_pct)

            fig3.add_trace(go.Scatter(
                x=np.concatenate([x_curve, x_curve[::-1]]),
                y=np.concatenate([y_upper, y_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(74,144,217,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Rango +-",
                hoverinfo="skip",
            ))
            eq_label = f"f(t) = {popt[0]:.2f}*ln(t) + {popt[1]:.2f}"
            fig3.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode="lines",
                name=eq_label,
                line=dict(color=HLINE_C, width=2, dash="dash"),
                hovertemplate="t=%{x:.0f}d<br>f(t)=%{y:.1f} kg/cm2<extra></extra>",
            ))
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
        hovertemplate="Edad: %{x}d<br>Promedio: %{y:.1f} kg/cm2<extra></extra>",
    ))
    fig3.add_hline(y=fc_nominal, line_dash="dot", line_color=HLINE_C,
                   annotation_text=f"f'c = {fc_nominal:.0f}",
                   annotation_font_color=HLINE_C)
    fig3.update_layout(**plotly_base(), height=380)
    fig3.update_xaxes(tickvals=[14, 28, 56], title_text="Edad (dias)", gridcolor=GRID, zerolinecolor=ZERO_LINE)
    fig3.update_yaxes(title_text="Promedio Resistencia (kg/cm2)", gridcolor=GRID, zerolinecolor=ZERO_LINE)
    st.plotly_chart(fig3, use_container_width=True)

# ─── TABLA DETALLE ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Detalle por Muestra</div>', unsafe_allow_html=True)

n_no_cumple = 0
tabla_rows  = []
for cil in sorted(df[cil_col].dropna().unique()):
    try:
        df_cil = df[df[cil_col] == cil]
        row    = {"N": int(cil)}
        loc    = df_cil["Localizacion"] if "Localizacion" in df_cil.columns else df_cil.get("Localización", pd.Series())
        row["Localizacion"] = loc.dropna().iloc[0] if not loc.dropna().empty else ""
        if toma_col and toma_col in df_cil.columns:
            fechas = df_cil[toma_col].dropna()
            fecha  = fechas.iloc[0] if not fechas.empty else None
            row["Fecha Toma"] = fecha.strftime("%Y-%m-%d") if fecha is not None and not pd.isna(fecha) else ""
        else:
            row["Fecha Toma"] = ""
        for edad in [14, 28, 56]:
            vals = df_cil[df_cil["Edad Estandar"] == edad][res_col].dropna()
            row[f"Prom {edad}d"] = round(float(vals.mean()), 1) if not vals.empty else None
        prom28_cil = row.get("Prom 28d")
        row["% f'c"] = f"{prom28_cil / fc_nominal * 100:.1f}%" if prom28_cil else None
        if prom28_cil:
            cumple_cil = prom28_cil > umbral_nsr
            row["NSR-10"] = "Cumple" if cumple_cil else "No Cumple"
            if not cumple_cil:
                n_no_cumple += 1
        else:
            row["NSR-10"] = "-"
        tabla_rows.append(row)
    except Exception:
        continue

df_tabla    = pd.DataFrame(tabla_rows)
total_con28 = sum(1 for r in tabla_rows if r.get("Prom 28d"))
pct_cumple  = (total_con28 - n_no_cumple) / total_con28 * 100 if total_con28 else 0

m1, m2, m3 = st.columns(3)
m1.metric("Umbral individual NSR-10", f"{umbral_nsr:.0f} kg/cm2",
          delta=f"f'c {'- 35' if fc_nominal <= 350 else 'x 0.9'}")
m2.metric("Muestras que No Cumplen", str(n_no_cumple),
          delta=f"de {total_con28} con datos a 28d", delta_color="inverse")
m3.metric("% Cumplimiento individual", f"{pct_cumple:.1f}%")

st.dataframe(df_tabla, use_container_width=True, hide_index=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Control Estadistico de Resistencia · NSR-10 / ACI 318")
