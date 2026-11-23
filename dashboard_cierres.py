"""SuperArgo · Dashboard de análisis de solicitudes PQR (archivo único)."""
from __future__ import annotations

import io
import re
import unicodedata
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN Y PALETA TEAL
# ═══════════════════════════════════════════════════════════════════════════════
APP_TITULO = "SuperArgo · Tablero de análisis de solicitudes (PQR)"
APP_ICON = "🩺"

C_NURC, C_UNIDAD, C_ESP, C_SOL, C_JUST, C_ADM, C_RIESGO = (
    "pqr_nurc", "UNIDAD", "Especialidad", "Solicitud", "Justificación",
    "Admitida", "TIPO DE RIESGO")
C_RASES, C_APROB, C_KEY, C_EN_MATRIZ = "RASES", "Aprobada", "_key_unidad", "EN_MATRIZ"
COLS_SOL = [C_NURC, C_UNIDAD, C_ESP, C_SOL, C_JUST, C_ADM, C_RIESGO]
COLS_REQ = [C_NURC, C_UNIDAD, C_ESP, C_ADM, C_RIESGO]
LARGO_NURC = 17

SI, NO = "Sí", "No"
APROB_CERRADA = "Cerrada por DISAN"
APROB_PARCIAL = "Parcialmente Cerrado"
APROB_SIN = "Sin Cierre de Fondo"
CAT_APROBADA = [APROB_CERRADA, APROB_PARCIAL, APROB_SIN]

RIESGO_VITAL, RIESGO_PRIOR, RIESGO_SIMPLE, SIN_CLASIFICAR = (
    "Riesgo Vital", "Priorizado", "Simple", "Sin Clasificar")
CAT_RIESGO = [RIESGO_VITAL, RIESGO_PRIOR, RIESGO_SIMPLE, SIN_CLASIFICAR]

SIN_RASES = "SIN CLASIFICAR"

TEAL_OSCURO  = "#064E5B"
TEAL_MEDIO1  = "#0E8B9E"
TEAL_MEDIO2  = "#3FB8AF"
TEAL_CLARO1  = "#7FD8D8"
TEAL_CLARO2  = "#CBF3F0"
TEAL_ESCALA  = [TEAL_CLARO2, TEAL_CLARO1, TEAL_MEDIO2, TEAL_MEDIO1, TEAL_OSCURO]

COLOR_APROBADA = {APROB_CERRADA: TEAL_OSCURO, APROB_PARCIAL: TEAL_MEDIO1, APROB_SIN: TEAL_CLARO1}
COLOR_RIESGO   = {RIESGO_VITAL: TEAL_OSCURO, RIESGO_PRIOR: TEAL_MEDIO1,
                  RIESGO_SIMPLE: TEAL_CLARO1, SIN_CLASIFICAR: "#E6F4F1"}

ALIAS_SOL = {
    C_NURC:   ["pqr_nurc", "pqrnurc", "nurc", "pqr", "radicado", "pqr nurc"],
    C_UNIDAD: ["unidad", "upres", "unidad_upres", "unidad (upres)", "institucion"],
    C_ESP:    ["especialidad", "especialidad_servicio", "servicio"],
    C_SOL:    ["solicitud", "detalle_solicitud", "requerimiento", "peticion"],
    C_JUST:   ["justificacion", "justificación", "motivo", "observaciones", "detalle"],
    C_ADM:    ["admitida", "admitido", "admite", "admision"],
    C_RIESGO: ["tipo de riesgo", "tipo_riesgo", "tiporiesgo", "riesgo", "clase_riesgo"],
}

_MAP_SI = {"si", "s", "sí", "yes", "y", "true", "1", "1.0", "admitida", "admitido", "x"}
_MAP_NO = {"no", "n", "false", "0", "0.0", "no admitida", "negada", ""}
_MAP_RIESGO = {
    "riesgo vital": RIESGO_VITAL, "vital": RIESGO_VITAL, "alto": RIESGO_VITAL,
    "priorizado": RIESGO_PRIOR, "riesgo priorizado": RIESGO_PRIOR, "prioritaria": RIESGO_PRIOR,
    "simple": RIESGO_SIMPLE, "no priorizado": RIESGO_SIMPLE, "ordinario": RIESGO_SIMPLE,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MATRIZ RASES-UPRES EMBEBIDA (35 pares)
# ═══════════════════════════════════════════════════════════════════════════════
MATRIZ_RASES_UPRES_DATA = [
    ("REGIONAL 1", "AMAZONAS"),
    ("REGIONAL 6", "ANTIOQUIA"),
    ("REGIONAL 5", "ARAUCA"),
    ("REGIONAL 8", "ATLÁNTICO"),
    ("REGIONAL 1", "BOGOTÁ. D.C."),
    ("REGIONAL 8", "BOLÍVAR"),
    ("REGIONAL 1", "BOYACÁ"),
    ("REGIONAL 3", "CALDAS"),
    ("REGIONAL 2", "CAQUETÁ"),
    ("REGIONAL 7", "CASANARE"),
    ("REGIONAL 4", "CAUCA"),
    ("REGIONAL 8", "CESAR"),
    ("REGIONAL 6", "CHOCÓ"),
    ("REGIONAL 6", "CÓRDOBA"),
    ("REGIONAL 1", "CUNDINAMARCA"),
    ("REGIONAL 1", "GUAINÍA"),
    ("REGIONAL 1", "GUAVIARE"),
    ("REGIONAL 1", "HOCEN"),
    ("REGIONAL 2", "HUILA"),
    ("REGIONAL 8", "LA GUAJIRA"),
    ("REGIONAL 8", "MAGDALENA"),
    ("REGIONAL 7", "META"),
    ("REGIONAL 4", "NARIÑO"),
    ("REGIONAL 5", "NORTE DE SANTANDER"),
    ("REGIONAL 2", "PUTUMAYO"),
    ("REGIONAL 3", "QUINDÍO"),
    ("REGIONAL 3", "RISARALDA"),
    ("REGIONAL 1", "SAN ANDRÉS"),
    ("REGIONAL 5", "SANTANDER"),
    ("REGIONAL 8", "SUCRE"),
    ("REGIONAL 2", "TOLIMA"),
    ("REGIONAL 6", "URABÁ"),
    ("REGIONAL 4", "VALLE DEL CAUCA"),
    ("REGIONAL 7", "VAUPÉS"),
    ("REGIONAL 1", "VICHADA"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════
def _sin_acentos(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(txt))
                   if not unicodedata.combining(c))

def _norm_key(valor) -> str:
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return ""
    return re.sub(r"\s+", " ", _sin_acentos(valor).upper()).strip()

def _norm_col(nombre) -> str:
    return re.sub(r"[^a-z0-9]", "", _sin_acentos(nombre).lower())

def _limpiar(serie: pd.Series) -> pd.Series:
    return (serie.astype("string").str.replace(r"\s+", " ", regex=True)
            .str.strip().fillna(""))

def _renombrar(df: pd.DataFrame, alias: dict) -> pd.DataFrame:
    lookup = {canon: {_norm_col(canon), *(_norm_col(a) for a in lista)}
              for canon, lista in alias.items()}
    mapa = {}
    for col in df.columns:
        n = _norm_col(col)
        for canon, opciones in lookup.items():
            if n in opciones and canon not in mapa.values():
                mapa[col] = canon
                break
    return df.rename(columns=mapa)

def _norm_admitida(v) -> str:
    v = _sin_acentos(str(v or "")).lower().strip()
    return SI if v in _MAP_SI else NO

def _norm_riesgo(v) -> str:
    v = _sin_acentos(str(v or "")).lower().strip()
    return _MAP_RIESGO.get(v, SIN_CLASIFICAR)

def _moda(s: pd.Series):
    s = s.dropna()
    s = s[s.astype(str) != ""]
    return s.mode().iloc[0] if not s.empty else ""

def _con_totales(tabla: pd.DataFrame, etiqueta: str = "TOTAL") -> pd.DataFrame:
    if tabla.empty:
        return tabla
    out = tabla.copy()
    out[etiqueta] = out.sum(axis=1)
    out.loc[etiqueta] = out.sum(axis=0)
    return out

def _pct(parte: int, total: int) -> str:
    return f"{(parte / total * 100):.1f} %" if total else "0.0 %"


def construir_matriz() -> pd.DataFrame:
    """Construye la matriz canónica desde la constante embebida."""
    if not MATRIZ_RASES_UPRES_DATA:
        return pd.DataFrame(columns=[C_KEY, "RASES", "UNIDAD"])
    df = pd.DataFrame(MATRIZ_RASES_UPRES_DATA, columns=["RASES", "UNIDAD"])
    df["RASES"] = df["RASES"].astype(str).str.strip().str.upper()
    df["UNIDAD"] = df["UNIDAD"].astype(str).str.strip()
    df[C_KEY] = df["UNIDAD"].map(_norm_key)
    return df[[C_KEY, "RASES", "UNIDAD"]].drop_duplicates(subset=[C_KEY], keep="first").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LECTURA Y VALIDACIÓN DE EXCEL
# ═══════════════════════════════════════════════════════════════════════════════
class ErrorDatos(Exception): pass

@st.cache_data(show_spinner=False)
def _leer_excel(file_bytes: bytes, hoja=0) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=hoja, dtype=str)
    return df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

def preparar_solicitudes(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _renombrar(df_raw, ALIAS_SOL)
    faltan = [c for c in COLS_REQ if c not in df.columns]
    if faltan:
        raise ErrorDatos(f"Faltan columnas: {', '.join(faltan)}. Detectadas: {list(df_raw.columns)}")
    for c in COLS_SOL:
        if c not in df.columns:
            df[c] = ""
    sol = df[COLS_SOL].copy()
    for c in COLS_SOL:
        sol[c] = _limpiar(sol[c])
    sol = sol[sol[C_NURC] != ""].reset_index(drop=True)
    if sol.empty:
        raise ErrorDatos("El archivo no contiene filas con pqr_nurc.")
    avisos = []
    mal = sol.loc[sol[C_NURC].str.len() != LARGO_NURC, C_NURC].unique()
    if len(mal):
        avisos.append(f"{len(mal)} pqr_nurc no tienen {LARGO_NURC} caracteres.")
    crudo = sol[C_ADM].copy()
    sol[C_ADM] = crudo.map(_norm_admitida)
    no_rec = crudo[~crudo.map(lambda v: _sin_acentos(str(v or "")).lower().strip()).isin(_MAP_SI | _MAP_NO)].unique()
    if len(no_rec):
        avisos.append(f"Valores de 'Admitida' no reconocidos → 'No': {', '.join(map(str, no_rec[:6]))}.")
    sol[C_RIESGO] = sol[C_RIESGO].map(_norm_riesgo)
    sin_r = int((sol[C_RIESGO] == SIN_CLASIFICAR).sum())
    if sin_r:
        avisos.append(f"{sin_r} solicitudes con riesgo fuera del catálogo → '{SIN_CLASIFICAR}'.")
    sol[C_KEY] = sol[C_UNIDAD].map(_norm_key)
    dup = int(sol.duplicated().sum())
    if dup:
        sol = sol.drop_duplicates().reset_index(drop=True)
        avisos.append(f"{dup} filas duplicadas fueron eliminadas.")
    return sol, avisos

def enriquecer_con_matriz(sol: pd.DataFrame, matriz: pd.DataFrame) -> pd.DataFrame:
    """Agrega RASES y la UNIDAD canónica desde la matriz embebida."""
    out = sol.copy()
    if matriz is None or matriz.empty:
        out[C_RASES] = SIN_RASES
        out[C_EN_MATRIZ] = False
        return out

    mat_renombrada = matriz.rename(columns={"UNIDAD": "_unidad_mat"})
    out = out.merge(mat_renombrada, on=C_KEY, how="left")
    out[C_EN_MATRIZ] = out["_unidad_mat"].notna()
    out[C_RASES] = out["RASES"].fillna(SIN_RASES)
    out[C_UNIDAD] = np.where(out[C_EN_MATRIZ], out["_unidad_mat"], out[C_UNIDAD])
    return out.drop(columns=["_unidad_mat"])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LÓGICA DE NEGOCIO
# ═══════════════════════════════════════════════════════════════════════════════
def construir_aprobaciones(sol: pd.DataFrame) -> pd.DataFrame:
    if sol.empty:
        return pd.DataFrame(columns=[C_NURC, C_UNIDAD, C_RASES, C_RIESGO, C_APROB,
                                     "Solicitudes", "Admitidas", "No admitidas"])
    tmp = sol.assign(_si=(sol[C_ADM] == SI).astype(int), _no=(sol[C_ADM] == NO).astype(int))
    apr = (tmp.groupby(C_NURC, sort=True)
           .agg(**{C_UNIDAD: (C_UNIDAD, _moda), C_RASES: (C_RASES, _moda),
                   C_RIESGO: (C_RIESGO, _moda),
                   "Solicitudes": (C_ADM, "size"),
                   "Admitidas": ("_si", "sum"), "No admitidas": ("_no", "sum")})
           .reset_index())
    apr[C_APROB] = np.select(
        [apr["No admitidas"].eq(0) & apr["Admitidas"].gt(0),
         apr["Admitidas"].gt(0) & apr["No admitidas"].gt(0)],
        [APROB_CERRADA, APROB_PARCIAL], default=APROB_SIN)
    return apr[[C_NURC, C_UNIDAD, C_RASES, C_RIESGO, C_APROB,
                "Solicitudes", "Admitidas", "No admitidas"]]

def aplicar_filtros(sol, apr, filtros, recalcular=False):
    mask = pd.Series(True, index=sol.index)
    for col, clave in ((C_RASES, "rases"), (C_UNIDAD, "unidades"),
                       (C_ESP, "especialidades"), (C_RIESGO, "riesgos")):
        v = filtros.get(clave)
        if v:
            mask &= sol[col].isin([v] if isinstance(v, str) else v)
    sol_f = sol[mask]
    apr_f = (construir_aprobaciones(sol_f) if recalcular
             else apr[apr[C_NURC].isin(sol_f[C_NURC].unique())])
    if filtros.get("aprobadas"):
        apr_f = apr_f[apr_f[C_APROB].isin(filtros["aprobadas"])]
        sol_f = sol_f[sol_f[C_NURC].isin(apr_f[C_NURC])]
    return sol_f.reset_index(drop=True), apr_f.reset_index(drop=True)

def admitidas_por_especialidad(sol):
    if sol.empty:
        return pd.DataFrame()
    si = sol[sol[C_ADM] == SI]
    return (si.groupby(C_ESP)
            .agg(**{"Solicitudes admitidas": (C_NURC, "size"),
                    "pqr_nurc admitidos": (C_NURC, "nunique")})
            .sort_values("pqr_nurc admitidos", ascending=False))

def nurc_unidad_especialidad(sol):
    if sol.empty:
        return pd.DataFrame()
    uniq = sol[[C_NURC, C_UNIDAD, C_ESP]].drop_duplicates()
    return pd.crosstab(uniq[C_UNIDAD], uniq[C_ESP])

def _pivot(df, index, columns, orden):
    if df.empty:
        return pd.DataFrame()
    p = pd.crosstab(df[index], df[columns])
    presentes = [c for c in orden if c in p.columns]
    extras = [c for c in p.columns if c not in presentes]
    return p.reindex(columns=presentes + extras, fill_value=0)

def conteo_categoria(apr, col, orden):
    if apr.empty:
        return pd.DataFrame(columns=[col, "Cantidad", "%"])
    s = apr.groupby(col).size()
    cats = [c for c in orden if c in s.index] + [c for c in s.index if c not in orden]
    df = s.reindex(cats, fill_value=0).astype(int).rename("Cantidad").reset_index()
    total = int(df["Cantidad"].sum())
    df["%"] = (df["Cantidad"] / total * 100).round(1) if total else 0.0
    return df

def a_largo(pivot, index, color):
    if pivot.empty:
        return pd.DataFrame(columns=[index, color, "Cantidad"])
    long = pivot.reset_index().melt(id_vars=index, var_name=color, value_name="Cantidad")
    return long[long["Cantidad"] > 0]

def exportar_excel(hojas):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for n, df, idx in hojas:
            (df if df is not None else pd.DataFrame()).to_excel(w, sheet_name=n[:31], index=idx)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════
def _base(fig, titulo, height=430):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=titulo, x=0.01, font=dict(size=18, color=TEAL_OSCURO, family="Segoe UI, Arial")),
        height=height, margin=dict(l=10, r=10, t=60, b=10),
        font=dict(family="Segoe UI, Arial", size=14, color="#0A3A40"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0,
                    font=dict(size=13)),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig

def fig_treemap(datos, por_rases=True):
    path = [C_RASES, C_UNIDAD] if por_rases else [C_UNIDAD]
    fig = px.treemap(datos, path=path, values="Cantidad",
                     color="Cantidad", color_continuous_scale=TEAL_ESCALA)
    fig.update_traces(
        texttemplate="%{label}<br><b>%{value}</b>",
        textfont=dict(size=13, family="Segoe UI, Arial"),
        hovertemplate="<b>%{label}</b><br>Reclamos: %{value}"
                      "<br>%{percentRoot:.1%} del total<extra></extra>",
        marker=dict(cornerradius=4))
    fig.update_coloraxes(showscale=False)
    return _base(fig, "DISTRIBUCIÓN DE LOS RECLAMOS POR UNIDAD", 470)

def fig_torta(datos, categoria, colores, titulo):
    """Torta ampliada con texto más grande."""
    fig = px.pie(datos, names=categoria, values="Cantidad", color=categoria,
                 color_discrete_map=colores,
                 category_orders={categoria: list(datos[categoria])})
    fig.update_traces(
        textinfo="percent+label", textposition="inside",
        textfont=dict(size=16, family="Segoe UI, Arial", color="#FFFFFF"),
        hovertemplate="<b>%{label}</b><br>Reclamos: %{value} (%{percent})<extra></extra>",
        marker=dict(line=dict(color="#FFFFFF", width=3)), sort=False)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0,
                    font=dict(size=14, family="Segoe UI, Arial")),
    )
    return _base(fig, titulo, 540)

def fig_columnas_agrupadas(datos, eje, color, colores, orden_color, titulo):
    """Columnas agrupadas: mismo grupo pegado, separador vertical entre unidades,
       eje Y entero, etiquetas a 90°."""
    totales = datos.groupby(eje)["Cantidad"].sum().sort_values(ascending=False)
    orden_eje = list(totales.index)

    fig = px.bar(datos, x=eje, y="Cantidad", color=color,
                 barmode="group",
                 color_discrete_map=colores,
                 category_orders={eje: orden_eje, color: orden_color},
                 text="Cantidad")
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12, color="#064E5B"),
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{value}<extra></extra>")

    n_unidades = len(orden_eje)
    shapes = []
    for i in range(1, n_unidades):
        shapes.append(dict(
            type="line",
            xref="x", yref="paper",
            x0=i - 0.5, y0=0, x1=i - 0.5, y1=1,
            line=dict(color="#CBD5E1", width=1, dash="dot"),
        ))

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Cantidad de pqr_nurc",
        xaxis=dict(
            tickangle=-90,
            tickfont=dict(size=12, family="Segoe UI, Arial"),
            showgrid=False,
        ),
        yaxis=dict(
            dtick=1,
            rangemode="tozero",
            showgrid=True,
            gridcolor="#E2E8F0",
            zeroline=True,
            zerolinecolor="#CBD5E1",
        ),
        shapes=shapes,
        uniformtext=dict(minsize=10, mode="hide"),
        bargap=0.25,
        bargroupgap=0,
        margin=dict(l=10, r=10, t=60, b=80),
    )
    return _base(fig, titulo, 520)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SIDEBAR: SELECTORES
# ═══════════════════════════════════════════════════════════════════════════════
def sidebar_filtros(sol, apr):
    with st.sidebar:
        st.markdown("""
            <style>
                .stSelectbox label { font-size: 15px !important; }
                .stSelectbox > div > div { font-size: 14px !important; }
                .stCheckbox label { font-size: 14px !important; }
                .stButton > button { font-size: 14px !important; }
            </style>
        """, unsafe_allow_html=True)
        st.markdown("### 🔎 Filtros")
        if sol.empty:
            st.caption("Carga un archivo para habilitar los filtros.")
            return {}, False

        rases_opts = sorted(sol[C_RASES].dropna().unique().tolist())
        rases_sel = st.selectbox("RASES", ["Todas"] + rases_opts, key="f_rases")
        pool = sol if rases_sel == "Todas" else sol[sol[C_RASES] == rases_sel]

        unidades_opts = sorted(pool[C_UNIDAD].dropna().unique().tolist())
        unidad_sel = st.selectbox("UNIDAD (UPRES)", ["Todas"] + unidades_opts, key="f_unidades")

        esp_opts = sorted(sol[C_ESP].dropna().unique().tolist())
        esp_sel = st.selectbox("Especialidad", ["Todas"] + esp_opts, key="f_especialidades")

        riesgo_opts = [c for c in CAT_RIESGO if c in apr[C_RIESGO].unique()] \
                      + [c for c in apr[C_RIESGO].unique() if c not in CAT_RIESGO]
        riesgo_sel = st.selectbox("Tipo de Riesgo", ["Todos"] + riesgo_opts, key="f_riesgos")

        aprob_opts = [c for c in CAT_APROBADA if c in apr[C_APROB].unique()] \
                     + [c for c in apr[C_APROB].unique() if c not in CAT_APROBADA]
        aprob_sel = st.selectbox("Tipo Aprobada", ["Todas"] + aprob_opts, key="f_aprobadas")

        recalcular = st.checkbox(
            "Recalcular 'Aprobada' con los filtros", value=False, key="f_recalc",
            help="Desactivado (recomendado): la clasificación se calcula con todas las solicitudes del nurc.")

        if st.button("↩️ Limpiar filtros", use_container_width=True, key="btn_limpiar_filtros"):
            for k in ("f_rases", "f_unidades", "f_especialidades", "f_riesgos", "f_aprobadas"):
                st.session_state.pop(k, None)
            st.rerun()

        filtros = {
            "rases":         [] if rases_sel == "Todas" else [rases_sel],
            "unidades":      [] if unidad_sel == "Todas" else [unidad_sel],
            "especialidades":[] if esp_sel == "Todas" else [esp_sel],
            "riesgos":       [] if riesgo_sel == "Todos" else [riesgo_sel],
            "aprobadas":     [] if aprob_sel == "Todas" else [aprob_sel],
        }
        return filtros, recalcular


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DIÁLOGO DE CARGA
# ═══════════════════════════════════════════════════════════════════════════════
@st.dialog("Cargar SOLICITUDES", width="large")
def dialogo_carga():
    st.caption("Selecciona el archivo Excel de la tabla SOLICITUDES y la fecha del análisis.")
    col1, col2 = st.columns([3, 1.2])
    archivo = col1.file_uploader("Excel SOLICITUDES", type=["xlsx", "xls"], key="dlg_up")
    fecha = col2.date_input("Fecha del análisis", value=date.today(),
                            format="DD/MM/YYYY", key="dlg_fecha")
    hoja = 0
    if archivo is not None:
        try:
            disponibles = list(pd.ExcelFile(io.BytesIO(archivo.getvalue())).sheet_names)
            hoja = st.selectbox("Hoja", disponibles, key="dlg_hoja")
        except Exception as e:
            st.error(f"No se pudo abrir el archivo: {e}")
    st.divider()
    if st.button("✅ Procesar archivo", type="primary", use_container_width=True):
        if archivo is None:
            st.warning("Selecciona un archivo antes de procesar.")
            return
        try:
            sol, avisos = preparar_solicitudes(_leer_excel(archivo.getvalue(), hoja))
            st.session_state.dataset = {
                "solicitudes": sol, "fecha": fecha,
                "archivo": archivo.name, "avisos": avisos,
                "cargado": datetime.now()}
            st.session_state.flash = [("warning", a) for a in avisos]
            st.rerun()
        except ErrorDatos as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error inesperado: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. RENDER DEL TABLERO
# ═══════════════════════════════════════════════════════════════════════════════
def render_tablero(sol, apr, filtros, recalcular):
    sol_f, apr_f = aplicar_filtros(sol, apr, filtros, recalcular)
    if apr_f.empty:
        st.warning("Ningún reclamo cumple los filtros seleccionados.")
        return

    n_casos, n_sol = len(apr_f), len(sol_f)
    n_si = int((sol_f[C_ADM] == SI).sum())
    conteo = apr_f[C_APROB].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reclamos analizados", f"{n_casos:,}".replace(",", "."))
    c2.metric("Solicitudes evaluadas", f"{n_sol:,}".replace(",", "."))
    c3.metric("Solicitudes admitidas", f"{n_si:,}".replace(",", "."), _pct(n_si, n_sol))
    c4.metric("Cerradas por DISAN", int(conteo.get(APROB_CERRADA, 0)),
              _pct(int(conteo.get(APROB_CERRADA, 0)), n_casos))

    st.markdown("""
        <style>
            div[data-testid="stMetricValue"] { font-size: 2rem !important; }
            div[data-testid="stMetricLabel"] { font-size: 1rem !important; }
            div[data-testid="stMetricDelta"] { font-size: 1rem !important; }
        </style>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Treemap con toggle: apagado = solo UNIDAD, encendido = RASES → UNIDAD ──
    agrupar_por_rases = st.toggle(
        "Agrupar por RASES",
        value=False,
        key="tm_rases",
        help="Apagado: distribución directa por UNIDAD. Encendido: jerarquía RASES → UNIDAD."
    )
    rutas = [C_RASES, C_UNIDAD] if agrupar_por_rases else [C_UNIDAD]
    datos_unidad = apr_f.groupby(rutas).size().reset_index(name="Cantidad")
    st.plotly_chart(fig_treemap(datos_unidad, agrupar_por_rases),
                    use_container_width=True, key="fig_treemap")

    # ── Tortas (más grandes) ──────────────────────────────────────────────────
    t_aprob = conteo_categoria(apr_f, C_APROB, CAT_APROBADA)
    t_riesgo = conteo_categoria(apr_f, C_RIESGO, CAT_RIESGO)
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(fig_torta(t_aprob, C_APROB, COLOR_APROBADA,
                                  "DISTRIBUCIÓN DE CAUSALIDAD DE CASOS"),
                        use_container_width=True, key="fig_pie_aprob")
        st.dataframe(t_aprob, hide_index=True, use_container_width=True)
    with g2:
        st.plotly_chart(fig_torta(t_riesgo, C_RIESGO, COLOR_RIESGO,
                                  "DISTRIBUCIÓN DEL RIESGO DE CASOS"),
                        use_container_width=True, key="fig_pie_riesgo")
        st.dataframe(t_riesgo, hide_index=True, use_container_width=True)

    # ── Columnas agrupadas ────────────────────────────────────────────────────
    p_aprob = _pivot(apr_f, C_UNIDAD, C_APROB, CAT_APROBADA)
    p_riesgo = _pivot(apr_f, C_UNIDAD, C_RIESGO, CAT_RIESGO)

    st.plotly_chart(fig_columnas_agrupadas(
        a_largo(p_aprob, C_UNIDAD, C_APROB), C_UNIDAD, C_APROB, COLOR_APROBADA, CAT_APROBADA,
        "CARACTERIZACIÓN DE LA APROBACIÓN POR UPRES"),
        use_container_width=True, key="fig_bar_aprob")

    st.plotly_chart(fig_columnas_agrupadas(
        a_largo(p_riesgo, C_UNIDAD, C_RIESGO), C_UNIDAD, C_RIESGO, COLOR_RIESGO, CAT_RIESGO,
        "CARACTERIZACIÓN DEL RIESGO POR UPRES"),
        use_container_width=True, key="fig_bar_riesgo")

    # ── Tablas resumen ────────────────────────────────────────────────────────
    t_esp = admitidas_por_especialidad(sol_f)
    t_uni_esp = nurc_unidad_especialidad(sol_f)

    st.markdown("""
        <style>
            .dataframe { font-size: 14px !important; }
            .dataframe th { font-size: 14px !important; }
            .dataframe td { font-size: 14px !important; }
        </style>
    """, unsafe_allow_html=True)

    with st.expander("📋 Tablas resumen", expanded=False):
        a, b, c, d = st.tabs(["Admitidas por Especialidad", "UNIDAD × Especialidad",
                              "Aprobada por UNIDAD", "Riesgo por UNIDAD"])
        with a: st.dataframe(t_esp, use_container_width=True)
        with b: st.dataframe(_con_totales(t_uni_esp), use_container_width=True)
        with c: st.dataframe(_con_totales(p_aprob), use_container_width=True)
        with d: st.dataframe(_con_totales(p_riesgo), use_container_width=True)

    with st.expander(f"✅ APROBACIONES ({len(apr_f):,} pqr_nurc únicos)".replace(",", ".")):
        st.dataframe(apr_f, hide_index=True, use_container_width=True, height=340)
    with st.expander(f"🗂️ SOLICITUDES filtradas ({len(sol_f):,} filas)".replace(",", ".")):
        st.dataframe(sol_f.drop(columns=[C_KEY, C_EN_MATRIZ], errors="ignore"),
                     hide_index=True, use_container_width=True, height=380)

    # ── Descarga Excel ────────────────────────────────────────────────────────
    meta = pd.DataFrame([
        {"Campo": "Archivo", "Valor": st.session_state.dataset["archivo"]},
        {"Campo": "Fecha análisis", "Valor": st.session_state.dataset["fecha"].strftime("%d/%m/%Y")},
        {"Campo": "Reclamos", "Valor": n_casos},
        {"Campo": "Solicitudes", "Valor": n_sol},
        {"Campo": "Recalculado con filtros", "Valor": "Sí" if recalcular else "No"},
        *[{"Campo": f"Filtro {k}", "Valor": ", ".join(v) or "(todos)"} for k, v in filtros.items()],
    ])
    hojas = [
        ("SOLICITUDES", sol_f.drop(columns=[C_KEY, C_EN_MATRIZ], errors="ignore"), False),
        ("APROBACIONES", apr_f, False),
        ("Admitidas por Especialidad", t_esp, True),
        ("NURC Unidad x Especialidad", _con_totales(t_uni_esp), True),
        ("Aprobada por UNIDAD", _con_totales(p_aprob), True),
        ("Por Aprobada", t_aprob, False),
        ("Riesgo por UNIDAD", _con_totales(p_riesgo), True),
        ("Por Tipo de Riesgo", t_riesgo, False),
        ("Parametros", meta, False),
    ]
    fecha_str = st.session_state.dataset["fecha"].strftime("%Y%m%d")
    st.download_button("⬇️ Descargar tablas (Excel)", exportar_excel(hojas),
                       f"SuperArgo_{fecha_str}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="dl_excel", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title=APP_TITULO, page_icon=APP_ICON, layout="wide")
    for k, v in (("dataset", None), ("flash", [])):
        st.session_state.setdefault(k, v)

    if "matriz" not in st.session_state:
        st.session_state.matriz = construir_matriz()
    matriz = st.session_state.matriz

    with st.sidebar:
        st.markdown("### 🗺️ Matriz RASES-UPRES")
        if matriz.empty:
            st.warning("⚠️ La matriz embebida está vacía.")
        else:
            st.success(f"{matriz['UNIDAD'].nunique()} UPRES · {matriz['RASES'].nunique()} RASES")
        st.divider()

    st.markdown(f"""
        <h1 style="font-size: 2.2rem; color: {TEAL_OSCURO};">{APP_ICON} SuperArgo · Dashboard de cierres</h1>
    """, unsafe_allow_html=True)
    st.markdown(f"""
        <p style="font-size: 1.1rem; color: #475569;">Cargue el Excel de SOLICITUDES. La matriz RASES-UPRES ya está incluida en el script.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        ds = st.session_state.dataset
        if ds:
            st.markdown(f"""
                <div style="background-color: #ECFDF5; padding: 12px 16px; border-radius: 8px;
                            border-left: 4px solid {TEAL_MEDIO1}; font-size: 1rem;">
                    📄 <b>{ds['archivo']}</b> · 🗓️ análisis: <b>{ds['fecha'].strftime('%d/%m/%Y')}</b> ·
                    {len(ds['solicitudes']):,} solicitudes cargadas
                </div>
            """, unsafe_allow_html=True)
    with col2:
        st.write("")
        if st.button("📂 Cargar / actualizar Excel", type="primary", use_container_width=True):
            dialogo_carga()

    for nivel, mensaje in st.session_state.flash:
        getattr(st, nivel)(mensaje)
    st.session_state.flash = []

    if st.session_state.dataset is None:
        st.info("👉 Pulsa **Cargar / actualizar Excel** y sube el archivo SOLICITUDES.")
        st.stop()

    sol = enriquecer_con_matriz(st.session_state.dataset["solicitudes"], matriz)
    apr = construir_aprobaciones(sol)

    fuera = sorted(sol.loc[~sol[C_EN_MATRIZ], C_UNIDAD].unique())
    if fuera:
        with st.expander(f"⚠️ {len(fuera)} UNIDAD no están en la matriz RASES-UPRES"):
            st.write(", ".join(fuera))

    filtros, recalcular = sidebar_filtros(sol, apr)
    render_tablero(sol, apr, filtros, recalcular)


if __name__ == "__main__":
    main()