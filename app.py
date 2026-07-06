# -*- coding: utf-8 -*-
"""
Dashboard de monitoreo RECLIMA — SCALL y Agrícola
Streamlit app: sube la exportación .xlsx (Kobo/ODK o el archivo de monitoreo)
y genera KPIs, flags de calidad y desgloses por distrito / encuestador / día.
"""

import io
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Monitoreo RECLIMA", page_icon="🌱", layout="wide")

# ----------------------------------------------------------------------------
# Acceso con contraseña (definir APP_PASSWORD en Secrets de Streamlit Cloud)
# ----------------------------------------------------------------------------
def check_password() -> bool:
    try:
        expected = st.secrets.get("APP_PASSWORD", "reclima2026")
    except Exception:
        expected = "reclima2026"
    if st.session_state.get("auth_ok"):
        return True
    st.title("🌱 Monitoreo RECLIMA")
    pwd = st.text_input("Contraseña", type="password")
    if pwd:
        if pwd == expected:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    return False


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def resolve(df: pd.DataFrame, key, exact: bool = False):
    """Devuelve la primera columna que coincide (exacto primero, luego subcadena,
    sin acentos ni mayúsculas). `key` puede ser una cadena o una lista de
    alternativas (para soportar distintas versiones del cuestionario)."""
    keys = [key] if isinstance(key, str) else list(key)
    for k in keys:
        nk = _norm(k)
        for c in df.columns:
            if _norm(c) == nk:
                return c
    if exact:
        return None
    for k in keys:
        nk = _norm(k)
        for c in df.columns:
            if nk in _norm(c):
                return c
    return None


# Alternativas por versión del cuestionario
K_ENUM = ["Nombre del encuestador", "Nombre del enumerador"]
K_NOMBRE = ["nombre completo del productor",
            "nombre completo de la jefa o jefe", "nombre completo"]


def col(df, key, exact=False):
    """Serie de la columna resuelta, o serie de NaN si no existe."""
    c = resolve(df, key, exact)
    if c is None:
        return pd.Series(np.nan, index=df.index)
    return df[c]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def skip(s: pd.Series, *sentinels) -> pd.Series:
    """NaN donde el valor es vacío o centinela (9, 99, 9999...)."""
    n = num(s)
    for v in sentinels:
        n = n.mask(n == v)
    return n


def flag_telefono(s: pd.Series) -> pd.Series:
    def bad(v):
        if pd.isna(v) or str(v).strip() == "":
            return False
        t = str(v).strip()
        if t.endswith(".0"):
            t = t[:-2]
        t = t.replace("-", "").replace(" ", "")
        return (not t.isdigit()) or len(t) > 8
    return s.map(bad)


def duracion_min(d: pd.DataFrame) -> pd.Series:
    if resolve(d, "start", exact=True) and resolve(d, "end", exact=True):
        return (pd.to_datetime(d["end"], errors="coerce")
                - pd.to_datetime(d["start"], errors="coerce")).dt.total_seconds() / 60
    return pd.Series(np.nan, index=d.index)


def flags_comunes(d: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Flags aplicables a cualquier módulo (v2/v3)."""
    f["G01 Duración <15 min"] = duracion_min(d) < 15
    fe = pd.to_datetime(col(d, "Fecha de la entrevista"), errors="coerce")
    f["G02 Fecha de entrevista vacía"] = fe.isna()
    nom = col(d, K_NOMBRE).astype(str).str.strip().str.upper()
    f["G03 Nombre de prueba"] = nom.isin(["NOMBRE", "PRUEBA", "TEST", "XXX", "N/A"])
    return f


def distinciones(d: pd.DataFrame) -> pd.DataFrame:
    """NO son errores: encuestas dejadas en borrador y complementadas por
    teléfono después. Se reportan como característica del levantamiento."""
    n = pd.DataFrame(index=d.index)
    n["Completada en borrador (>240 min)"] = (duracion_min(d) > 240).fillna(False)
    fe = pd.to_datetime(col(d, "Fecha de la entrevista"), errors="coerce")
    if resolve(d, "today", exact=True):
        ty = pd.to_datetime(d["today"], errors="coerce")
        n["Enviada en fecha distinta a la entrevista"] = (
            fe.notna() & ty.notna() & (fe.dt.date != ty.dt.date)).fillna(False)
    return n


@st.cache_data(show_spinner=False)
def load_book(file_bytes: bytes) -> dict:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return {name: xls.parse(name) for name in xls.sheet_names}


def pick_sheet(book: dict, *candidates):
    for cand in candidates:
        for name in book:
            if _norm(cand) == _norm(name):
                return book[name]
    for cand in candidates:
        for name in book:
            if _norm(cand) in _norm(name):
                return book[name]
    return None


def detect_module(book: dict):
    data = pick_sheet(book, "DATA")
    if data is None:  # exportación cruda: usar la hoja más ancha
        data = max(book.values(), key=lambda d: d.shape[1])
    joined = " ".join(_norm(c) for c in data.columns)
    if "recoleccion de lluvia" in joined or "scall" in joined:
        return "SCALL", data
    if "terrenos o parcelas" in joined or pick_sheet(book, "roster_parcela") is not None:
        return "AGRICOLA", data
    return None, data


# ----------------------------------------------------------------------------
# Definición de flags (replican y extienden las fórmulas del Excel de monitoreo)
# ----------------------------------------------------------------------------
def flags_scall(d: pd.DataFrame) -> pd.DataFrame:
    an = num(col(d, "personas habitan al dia de hoy"))
    ao = num(col(d, "cuantas son mujeres"))
    ap = num(col(d, "cuantas son hombres"))
    f = pd.DataFrame(index=d.index)
    f["F01 Teléfono inválido"] = flag_telefono(col(d, "numero de telefono del productor"))
    f["F02 Hogar >10 personas"] = an > 10
    f["F03 Mujeres > total"] = ao > an
    f["F04 Hombres+Mujeres ≠ Total"] = (ao + ap).notna() & an.notna() & ((ao + ap) != an)
    f["F05 Ayudantes > total hogar"] = num(col(d, "ayudantes del productor")) > an
    f["F06 Año SCALL < 2020"] = skip(col(d, "año que le instalaron"), 9) < 2020
    bn = skip(col(d, "tiempo ida y vuelta para acarrear agua"), 9999)
    f["F07 Tiempo acarreo sospechoso"] = (bn > 240) | (bn < 10)
    f["F08 Días agua >30"] = skip(col(d, "cuantos dias hubo agua disponible"), 99) > 30
    f["F09 Litros SCALL >10,000"] = skip(col(d, "cuantos litros de agua aporta"), 9999) > 10000
    bv = skip(col(d, ["cuantos meses aporta agua", "meses al año su scall aporta"]), 99)
    f["F10 Meses fuera 0-12"] = (bv < 0) | (bv > 12)
    ce = skip(col(d, ["tiempo actual ida y vuelta", "tiempo actual de ida y vuelta"]), 9999)
    f["F11 Tiempo actual sospechoso"] = (ce < 5) | (ce > 1440)
    cf = num(col(d, "gasto mensual actual en compra"))
    f["F12 Gasto agua atípico"] = (cf > 0) & ((cf < 1) | (cf > 1000))
    cg = skip(col(d, "capacidad total de almacenamiento"), 9999, 999)
    f["F13 Almacenamiento <100 L"] = (cg > 0) & (cg < 100)
    ct = num(col(d, "gasto estimado en mantenimiento"))
    f["F14 Gasto mant. atípico"] = (ct > 0) & ((ct < 10) | (ct > 10000))
    # v3 (jun 2026)
    f["F15 Días tanque lleno >365"] = skip(col(d, "dias le duraria un tanque lleno"), 9999) > 365
    f["F16 Centinela 9999/999 sin depurar"] = (
        (num(col(d, "dias le duraria un tanque lleno")) == 9999)
        | (num(col(d, ["tiempo actual ida y vuelta", "tiempo actual de ida y vuelta"])) == 9999)
        | (num(col(d, "capacidad total de almacenamiento")).isin([9999, 999]))
        | (num(col(d, "dias estuvo fuera de servicio")) == 999)
    )
    return flags_comunes(d, f).fillna(False)


def flags_agricola(d: pd.DataFrame) -> pd.DataFrame:
    edad = num(col(d, "edad del productor"))
    gs_raw = num(col(d, "compra de la SEMILLA"))
    gf_raw = num(col(d, "compra de FERTILIZANTES"))
    ga_raw = num(col(d, "AGROQUIMICOS"))
    gm_raw = num(col(d, "mano de obra o jornales"))
    ing_raw = num(col(d, "ingreso total obtenido por la venta"))
    # 9999 es centinela de "no sabe": no debe contar como gasto/ingreso real
    gs, gf, ga = gs_raw.mask(gs_raw == 9999), gf_raw.mask(gf_raw == 9999), ga_raw.mask(ga_raw == 9999)
    gm, ing = gm_raw.mask(gm_raw == 9999), ing_raw.mask(ing_raw == 9999)
    gastos = gs.fillna(0) + gf.fillna(0) + ga.fillna(0) + gm.fillna(0)
    parc = num(col(d, "TERRENOS o PARCELAS"))
    cult = num(col(d, "CULTIVOS tuvo en total"))
    f = pd.DataFrame(index=d.index)
    f["F01 Teléfono inválido"] = flag_telefono(col(d, "numero de telefono del productor"))
    f["F02 Edad atípica"] = (edad < 15) | (edad > 100)
    f["F03 Hogar >15 personas"] = num(col(d, "personas habitan al dia de hoy")) > 15
    f["F04 Gasto semilla >$5,000"] = gs > 5000
    f["F05 Gasto fertilizantes >$5,000"] = gf > 5000
    f["F06 Gasto agroquímicos >$5,000"] = ga > 5000
    f["F07 Gasto mano obra >$10,000"] = gm > 10000
    f["F08 Ingreso ventas >$50,000"] = ing > 50000
    f["F09 Ingresos >> gastos x10"] = gs.notna() & ing.notna() & (gastos > 0) & (ing > gastos * 10)
    f["F10 Más parcelas que cultivos"] = (parc > 0) & (cult > 0) & (parc > cult)
    f["F11 Centinela 9999 en gastos/ingreso"] = (
        (gs_raw == 9999) | (gf_raw == 9999) | (ga_raw == 9999)
        | (gm_raw == 9999) | (ing_raw == 9999))
    return flags_comunes(d, f).fillna(False)


def flags_parcela(r: pd.DataFrame) -> pd.DataFrame:
    area = num(col(r, "el area de"))
    unidad = col(r, "M1_Q6b", exact=True).astype(str)
    ax = num(col(r, "cuantos arboles tenia"))
    ay = num(col(r, "cuantos arboles planto"))
    f = pd.DataFrame(index=r.index)
    f["P01 Área parcela = 0"] = area <= 0
    f["P02 Área parcela >500 mz"] = (unidad == "Manzanas") & (area > 500)
    f["P03 Árboles plantados > existentes"] = (ax > 0) & (ay > ax)
    f["P04 Qty fertilizante = 9999"] = num(col(r, "cantidad total de fertilizantes")) == 9999
    f["P05 Qty agroquímico = 9999"] = num(col(r, "cantidad de ${M1_Q20}")) == 9999
    return f.fillna(False)


def flags_cultivo(r: pd.DataFrame) -> pd.DataFrame:
    semb = num(col(r, "area total sembrada"))
    u_semb = col(r, ["M2_Q2b", "M2s_Q2b"], exact=True).astype(str)
    cos = num(col(r, "area total cosechada"))
    u_cos = col(r, ["M2_Q3b", "M2s_Q3b"], exact=True).astype(str)
    prod = num(col(r, "la produccion de"))
    f = pd.DataFrame(index=r.index)
    f["C01 Cosechada > Sembrada"] = (u_semb == u_cos) & (cos > semb)
    f["C02 Producción=0, área>0"] = (semb > 0) & (prod == 0)
    f["C03 Qty semilla = 9999"] = num(col(r, "cuanta semilla")) == 9999
    f["C04 Área sembrada >500 mz"] = (u_semb == "Manzanas") & (semb > 500)
    return f.fillna(False)


# ----------------------------------------------------------------------------
# Render del dashboard
# ----------------------------------------------------------------------------
CHART = dict(height=340)


def kpis_y_desgloses(d: pd.DataFrame, flags: pd.DataFrame, notas: pd.DataFrame):
    enum_c = resolve(d, K_ENUM)
    fecha_c = resolve(d, "Fecha de la entrevista")
    distr_c = resolve(d, "Distrito", exact=True) or resolve(d, "Distrito")
    fechas = pd.to_datetime(d[fecha_c], errors="coerce") if fecha_c else pd.Series(dtype="datetime64[ns]")

    # KPIs
    dur = duracion_min(d)
    n_flag = flags.any(axis=1).sum()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Encuestas", len(d))
    c2.metric("Encuestadores", d[enum_c].nunique() if enum_c else "—")
    c3.metric("Distritos", d[distr_c].nunique() if distr_c else "—")
    c4.metric("Días de campo", fechas.dt.date.nunique())
    c5.metric("Con ≥1 flag", f"{n_flag} ({n_flag/len(d):.0%})" if len(d) else "0")
    c6.metric("Duración mediana", f"{dur.median():.0f} min" if dur.notna().any() else "—")

    # Resumen de flags
    st.subheader("🔍 Resumen de flags de calidad")
    resumen = flags.sum().sort_values(ascending=False)
    resumen = resumen[resumen > 0]
    if resumen.empty:
        st.success("Ningún flag activo. 🎉")
    else:
        t, g = st.columns([2, 3])
        t.dataframe(resumen.rename("Casos ⚠").to_frame(), width="stretch")
        fig = px.bar(resumen.sort_values(), orientation="h",
                     labels={"value": "Casos", "index": ""}, **CHART)
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        g.plotly_chart(fig, width="stretch")

    # Distinciones de levantamiento (informativo, no errores)
    if not notas.empty and notas.any().any():
        partes = [f"{int(notas[c].sum())} {c.lower()}" for c in notas.columns if notas[c].sum() > 0]
        st.info("📱 **Distinciones del levantamiento** (no son errores — encuestas "
                "que quedan en borrador y se complementan por teléfono): "
                + " · ".join(partes) + ".")

    # Desgloses
    st.subheader("📈 Avance de campo")
    a, b = st.columns(2)
    if distr_c:
        vc = d[distr_c].value_counts()
        fig = px.bar(vc, labels={"value": "Encuestas", "index": "Distrito"},
                     title="Por distrito", **CHART)
        fig.update_layout(showlegend=False)
        a.plotly_chart(fig, width="stretch")
    if enum_c:
        vc = d[enum_c].value_counts()
        fig = px.bar(vc, labels={"value": "Encuestas", "index": "Encuestador"},
                     title="Por encuestador", **CHART)
        fig.update_layout(showlegend=False)
        b.plotly_chart(fig, width="stretch")
    if fechas.notna().any():
        por_dia = fechas.dt.date.value_counts().sort_index()
        fig = px.bar(por_dia, labels={"value": "Encuestas", "index": "Día"},
                     title="Encuestas por día", height=300)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")


def tabla_flags(d: pd.DataFrame, flags: pd.DataFrame):
    st.subheader("⚠ Registros con flags")
    idx_c = resolve(d, "_index", exact=True)
    base = pd.DataFrame({
        "#": d[idx_c] if idx_c else d.index + 1,
        "Encuestador": col(d, K_ENUM),
        "Fecha": pd.to_datetime(col(d, "Fecha de la entrevista"), errors="coerce").dt.strftime("%d/%m/%Y"),
        "Productor": col(d, K_NOMBRE),
        "Distrito": col(d, "Distrito"),
    })
    activos = flags[flags.any(axis=1)]
    if activos.empty:
        st.info("Sin registros con flags.")
        return
    sel = st.multiselect("Filtrar por flag", list(flags.columns),
                         key=f"filtro_{id(flags)}")
    mask = flags[sel].any(axis=1) if sel else flags.any(axis=1)
    out = base[mask].copy()
    out["Flags activos"] = flags[mask].apply(
        lambda r: ", ".join(f.split(" ")[0] for f in flags.columns if r[f]), axis=1)
    show = pd.concat([out, flags[mask].replace({True: "⚠", False: ""})], axis=1)
    st.dataframe(show, width="stretch", hide_index=True)


def ficha_encuestado(d: pd.DataFrame, flags: pd.DataFrame, book: dict = None,
                     modulo: str = ""):
    st.subheader("👤 Ficha del encuestado")
    nom_c = resolve(d, K_NOMBRE)
    idx_c = resolve(d, "_index", exact=True)
    if nom_c is None:
        st.info("No se encontró la columna de nombre.")
        return
    ids = d[idx_c].astype(str) if idx_c else (d.index + 1).astype(str)
    etiquetas = ids + " — " + d[nom_c].astype(str)
    elegido = st.selectbox("Seleccione encuestado (escriba para buscar)",
                           etiquetas, key=f"ficha_{modulo}")
    i = etiquetas[etiquetas == elegido].index[0]
    fila = d.loc[i]

    # Flags de este registro
    activos = [f for f in flags.columns if flags.at[i, f]]
    if activos:
        st.warning("⚠ Flags de este registro: " + ", ".join(activos))
    else:
        st.success("✔ Sin flags de calidad en este registro.")

    # Todas las respuestas (Pregunta / Respuesta)
    ficha = fila.dropna()
    ficha = ficha[ficha.astype(str).str.strip() != ""]
    ficha.index.name = "Pregunta"
    st.dataframe(ficha.rename("Respuesta").to_frame().astype(str),
                 width="stretch", height=450)

    # Parcelas y cultivos del encuestado (Agrícola)
    if book is not None and idx_c is not None:
        rp = pick_sheet(book, "roster_parcela")
        if rp is not None and resolve(rp, "_parent_index", exact=True):
            mias = rp[rp["_parent_index"] == fila[idx_c]]
            if len(mias):
                st.markdown(f"**🌾 Parcelas de este encuestado ({len(mias)})**")
                st.dataframe(mias.dropna(axis=1, how="all").astype(str),
                             width="stretch", hide_index=True)
                idxs = set(mias["_index"]) if "_index" in mias.columns else set()
                for sh, tit in [("roster_cultivos", "🌽 Cultivos — parcela principal"),
                                ("roster_cultivo", "🌽 Cultivos — otras parcelas")]:
                    rc = pick_sheet(book, sh)
                    if rc is not None and "_parent_index" in rc.columns and idxs:
                        c = rc[rc["_parent_index"].isin(idxs)]
                        if len(c):
                            st.markdown(f"**{tit} ({len(c)})**")
                            st.dataframe(c.dropna(axis=1, how="all").astype(str),
                                         width="stretch", hide_index=True)


def seccion_roster(book: dict, d: pd.DataFrame, sheet: str, flag_fn, titulo: str,
                   cols_id: list, via_sheet: str = None):
    r = pick_sheet(book, sheet)
    if r is None:
        st.caption(f"(No se encontró la hoja {sheet} en el archivo.)")
        return
    fl = flag_fn(r)
    st.subheader(titulo)
    resumen = fl.sum()
    resumen = resumen[resumen > 0]
    if resumen.empty:
        st.success("Ningún flag activo.")
        return
    st.dataframe(resumen.rename("Casos ⚠").to_frame(), width=380)
    # detalle con nombre del productor vía _parent_index
    mask = fl.any(axis=1)
    det = pd.DataFrame(index=r.index)
    for etiqueta, key in cols_id:
        es_exacta = isinstance(key, str) and (key.startswith("_") or key in ("CL", "CLs", "M1_Q6b"))
        det[etiqueta] = col(r, key, exact=es_exacta)
    parent_c = resolve(r, "_parent_index", exact=True)
    idx_c = resolve(d, "_index", exact=True)
    nom_c = resolve(d, K_NOMBRE)
    if parent_c and idx_c and nom_c:
        parents = r[parent_c]
        if via_sheet:  # el roster cuelga de otro roster (ej. cultivo → parcela)
            v = pick_sheet(book, via_sheet)
            if v is not None and "_index" in v.columns and "_parent_index" in v.columns:
                parents = parents.map(v.set_index(v["_index"])["_parent_index"])
        mapa = d.set_index(d[idx_c])[nom_c]
        det["Productor"] = parents.map(mapa)
    show = pd.concat([det[mask], fl[mask].replace({True: "⚠", False: ""})], axis=1)
    st.dataframe(show, width="stretch", hide_index=True)


def render_modulo(book: dict, esperado: str, nombre: str):
    detectado, d = detect_module(book)
    if detectado and detectado != esperado:
        st.error(f"Este archivo parece del módulo **{detectado}**, no de {nombre}. "
                 "Súbelo en la otra pestaña.")
        return
    flags = flags_scall(d) if esperado == "SCALL" else flags_agricola(d)
    notas = distinciones(d)
    kpis_y_desgloses(d, flags, notas)
    tabla_flags(d, flags)
    if esperado == "AGRICOLA":
        st.divider()
        seccion_roster(book, d, "roster_parcela", flags_parcela,
                       "🌾 Flags de parcelas",
                       [("Parcela", ["nombre de su parcela", "nombre de la parcela"]),
                        ("Área", "el area de"), ("Unidad", "M1_Q6b")])
        seccion_roster(book, d, "roster_cultivos", flags_cultivo,
                       "🌽 Flags de cultivos — parcela principal",
                       [("Cultivo", "CLs"), ("Área sembrada", "area total sembrada")],
                       via_sheet="roster_parcela")
        seccion_roster(book, d, "roster_cultivo", flags_cultivo,
                       "🌽 Flags de cultivos — otras parcelas",
                       [("Cultivo", "CL"), ("Área sembrada", "area total sembrada")],
                       via_sheet="roster_parcela")
    seccion_reporte(d, flags, notas, "SCALL" if esperado == "SCALL" else "Agrícola")
    st.divider()
    ficha_encuestado(d, flags, book if esperado == "AGRICOLA" else None, esperado)


# ----------------------------------------------------------------------------
# Reporte acumulado para supervisores (.docx)
# ----------------------------------------------------------------------------
def _doc_tabla(doc, filas, headers):
    from docx.shared import Pt
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        run = t.rows[0].cells[i].paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
    for fila in filas:
        celdas = t.add_row().cells
        for i, v in enumerate(fila):
            run = celdas[i].paragraphs[0].add_run("" if pd.isna(v) else str(v))
            run.font.size = Pt(9)
    return t


def reporte_acumulado_docx(d: pd.DataFrame, flags: pd.DataFrame,
                           notas: pd.DataFrame, modulo: str,
                           extra: str = "") -> bytes:
    from docx import Document
    from docx.shared import Pt

    enum_c = resolve(d, K_ENUM)
    distr_c = resolve(d, "Distrito", exact=True) or resolve(d, "Distrito")
    nom_c = resolve(d, K_NOMBRE)
    fechas = pd.to_datetime(col(d, "Fecha de la entrevista"), errors="coerce").dt.date
    con_flag = flags.any(axis=1)

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)

    doc.add_heading(f"Reporte de campo — RECLIMA {modulo}", 0)
    rango = ""
    fv = fechas.dropna()
    if len(fv):
        rango = f"Periodo: {fv.min():%d/%m/%Y} – {fv.max():%d/%m/%Y}   ·   "
    doc.add_paragraph(
        rango + f"Generado: {datetime.now():%d/%m/%Y %H:%M}   ·   "
        "Proyecto RECLIMA — Evaluación final, corredor seco de El Salvador"
    )

    # 1. Avance
    doc.add_heading("1. Avance de campo (acumulado)", 1)
    doc.add_paragraph(
        f"Se han levantado {len(d)} encuestas con "
        f"{d[enum_c].nunique() if enum_c else 0} encuestadores en "
        f"{d[distr_c].nunique() if distr_c else 0} distritos, a lo largo de "
        f"{fechas.nunique()} días de campo. "
        f"{int(con_flag.sum())} encuestas ({con_flag.mean():.0%}) tienen al menos "
        "un flag de calidad pendiente de revisión."
    )
    if enum_c:
        g = d.groupby(d[enum_c]).size().sort_values(ascending=False)
        doc.add_paragraph("Encuestas por encuestador:", style="Intense Quote")
        _doc_tabla(doc, list(g.items()), ["Encuestador", "Encuestas"])
    if distr_c:
        g = d.groupby(d[distr_c]).size().sort_values(ascending=False)
        doc.add_paragraph("Encuestas por distrito:", style="Intense Quote")
        _doc_tabla(doc, list(g.items()), ["Distrito", "Encuestas"])

    # 2. Flags a atender
    doc.add_heading("2. Flags a atender", 1)
    resumen = flags.sum().sort_values(ascending=False)
    resumen = resumen[resumen > 0]
    if resumen.empty:
        doc.add_paragraph("No hay flags de calidad activos. ✔")
    else:
        doc.add_paragraph(
            "Casos por flag, con los productores afectados y su encuestador "
            "(para ubicar y verificar cada registro):")
        filas = []
        for flag_name, n in resumen.items():
            afectados = []
            for i in d.index[flags[flag_name]]:
                p = d.at[i, nom_c] if nom_c else f"fila {i}"
                e = d.at[i, enum_c] if enum_c else ""
                afectados.append(f"{p} ({e})" if e else str(p))
            filas.append((flag_name, int(n), "; ".join(map(str, afectados))))
        _doc_tabla(doc, filas, ["Flag", "Casos", "Productores afectados (encuestador)"])
    if extra:
        doc.add_paragraph(extra)

    # 3. Distinciones del levantamiento (informativo)
    if not notas.empty and notas.any().any():
        doc.add_heading("3. Distinciones del levantamiento (informativo)", 1)
        doc.add_paragraph(
            "Las siguientes no son errores: corresponden a encuestas que se dejan "
            "abiertas en borrador y luego se complementan por teléfono. Se "
            "reportan solo como característica del levantamiento:")
        _doc_tabla(doc, [(c, int(notas[c].sum()),
                          f"{notas[c].mean():.0%} del total")
                         for c in notas.columns if notas[c].sum() > 0],
                   ["Distinción", "Encuestas", "Proporción"])

    # 4. Ranking de encuestadores
    doc.add_heading("4. Ranking de encuestadores", 1)
    if enum_c:
        rk = pd.DataFrame({
            "Encuestas": d.groupby(d[enum_c]).size(),
            "Con flags": con_flag.groupby(d[enum_c]).sum().astype(int),
        })
        rk["% flags"] = (rk["Con flags"] / rk["Encuestas"]).map("{:.0%}".format)
        rk = rk.sort_values("Encuestas", ascending=False)
        _doc_tabla(doc, [(i, r["Encuestas"], r["Con flags"], r["% flags"])
                         for i, r in rk.iterrows()],
                   ["Encuestador", "Encuestas", "Con flags", "% con flags"])
        doc.add_paragraph(
            "Nota: un % alto de flags no siempre implica mal desempeño — puede reflejar "
            "zonas con casos atípicos reales. Usar como insumo de supervisión, no de sanción.")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def seccion_reporte(d: pd.DataFrame, flags: pd.DataFrame, notas: pd.DataFrame,
                    modulo: str):
    st.divider()
    st.subheader("📄 Reporte para supervisores (acumulado)")
    extra = st.text_input("Observaciones (opcional, se incluyen en el reporte)",
                          key=f"rep_obs_{modulo}")
    data = reporte_acumulado_docx(d, flags, notas, modulo, extra)
    st.download_button(
        "⬇️ Descargar reporte (.docx)",
        data=data,
        file_name=f"Reporte_RECLIMA_{modulo}_{datetime.now():%Y-%m-%d}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"rep_dl_{modulo}",
    )


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
if check_password():
    st.title("🌱 Dashboard de monitoreo RECLIMA")
    st.caption("Evaluación final — corredor seco de El Salvador. "
               "Sube la exportación .xlsx de KoboToolbox (o el archivo de monitoreo) en la pestaña del módulo.")

    tab_scall, tab_agri = st.tabs(["💧 SCALL", "🌾 Agrícola"])

    with tab_scall:
        up = st.file_uploader("Base SCALL (.xlsx)", type="xlsx", key="up_scall")
        if up:
            render_modulo(load_book(up.getvalue()), "SCALL", "SCALL")
        else:
            st.info("Sube la base para generar el dashboard.")

    with tab_agri:
        up = st.file_uploader("Base Agrícola (.xlsx)", type="xlsx", key="up_agri")
        if up:
            render_modulo(load_book(up.getvalue()), "AGRICOLA", "Agrícola")
        else:
            st.info("Sube la base para generar el dashboard.")
