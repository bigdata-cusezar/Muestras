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

# 🔁 Conversión a MPa
fc_nominal_mpa = fc_nominal / 10
prom28_mpa     = prom28 / 10
ds_mpa         = ds / 10
fcr_mpa        = fcr / 10

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
card(c1, "f'c Nominal",      f"{fc_nominal_mpa:.2f}",  "MPa")
card(c2, "Promedio 28d",     f"{prom28_mpa:.2f}",      "MPa")
card(c3, "Desv. Estandar",   f"{ds_mpa:.2f}",          cal_ds, cls_ds)
card(c4, "Coef. Variacion",  f"{cv*100:.1f}%",         cal_cv, cls_cv)
card(c5, "N Ensayos (Tomas)", str(n))
card(c6, "f'cr Diseño",      f"{fcr_mpa:.2f}",         "MPa")
card(c7, "NSR-10 Global",
    "Cumple" if cumple_global else "No Cumple",
    sub="x vs f'cr estadistico",
    cls="cumple" if cumple_global else "nocumple",
    reason=nsr_reason)

# ─── TABLA DETALLE ───────────────────────────────────────────────────────────
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

        if toma_col and toma_col in df_toma.columns:
            fechas = df_toma[toma_col].dropna()
            if not fechas.empty:
                fecha = fechas.iloc[0]
                row["Fecha Toma"] = fecha.strftime("%Y-%m-%d") if hasattr(fecha, "strftime") else ""
            else:
                row["Fecha Toma"] = ""

        if cil_col:
            cils = df_toma[cil_col].dropna()
            row["Cilindros"] = ", ".join(str(int(c)) for c in sorted(cils.unique()))

        # 🔁 PROMEDIOS EN MPa
        for edad in [14, 28, 56]:
            vals = df_toma[df_toma["Edad Estandar"] == edad][res_col].dropna()
            if not vals.empty:
                valor_kg = float(vals.mean())
                valor_mpa = valor_kg / 10
                row[f"Prom {int(edad)}d (MPa)"] = round(valor_mpa, 2)
                row[f"N cil {int(edad)}d"] = len(vals)
            else:
                row[f"Prom {int(edad)}d (MPa)"] = None
                row[f"N cil {int(edad)}d"] = 0

        prom28_toma = row.get("Prom 28d (MPa)")
        row["% f'c"] = f"{prom28_toma / fc_nominal_mpa * 100:.1f}%" if prom28_toma else None

        # 🔁 VALIDACIÓN EN kg/cm²
        if prom28_toma:
            prom28_kg = prom28_toma * 10
            cumple_toma = prom28_kg > umbral_nsr
            row["NSR-10"] = "✅ Cumple" if cumple_toma else "❌ No Cumple"
            if not cumple_toma:
                n_no_cumple += 1
        else:
            row["NSR-10"] = "—"

        tabla_rows.append(row)

    except Exception:
        continue

df_tabla = pd.DataFrame(tabla_rows)

total_con28 = sum(1 for r in tabla_rows if r.get("Prom 28d (MPa)"))
pct_cumple  = (total_con28 - n_no_cumple) / total_con28 * 100 if total_con28 else 0

# 🔁 Umbral en MPa
umbral_mpa = umbral_nsr / 10

m1, m2, m3 = st.columns(3)
m1.metric("Umbral individual NSR-10", f"{umbral_mpa:.2f} MPa",
          delta=f"f'c {'− 3.5' if fc_nominal <= 350 else '× 0.9'}")
m2.metric("Ensayos que No Cumplen", str(n_no_cumple),
          delta=f"de {total_con28} ensayos con datos a 28d", delta_color="inverse")
m3.metric("% Cumplimiento por ensayo", f"{pct_cumple:.1f}%")

st.dataframe(df_tabla, use_container_width=True, hide_index=True)
