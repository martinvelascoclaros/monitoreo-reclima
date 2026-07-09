# -*- coding: utf-8 -*-
"""
Dashboard de monitoreo RECLIMA — SCALL y Agrícola
Publica los datos el administrador (carpeta data/ del repo); el equipo
consulta con enlace + contraseña. El tablero NO muestra datos personales
de los entrevistados (solicitud FAO): solo el ID de Kobo (_index).
"""

import io
import os
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
    """Primera columna que coincide (exacto primero, luego subcadena, sin
    acentos ni mayúsculas). `key` puede ser cadena o lista de alternativas."""
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
    c = resolve(df, key, exact)
    if c is None:
        return pd.Series(np.nan, index=df.index)
    return df[c]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def skip(s: pd.Series, *sentinels) -> pd.Series:
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


def id_encuesta(d: pd.DataFrame) -> pd.Series:
    """ID para rastrear el registro en Kobo (sin datos personales)."""
    idx_c = resolve(d, "_index", exact=True)
    return d[idx_c].astype("Int64").astype(str) if idx_c else (d.index + 1).astype(str)


# ----------------------------------------------------------------------------
# Protección de datos personales (solicitud FAO): el tablero solo muestra el
# ID de Kobo; nombres, teléfonos, direcciones y coordenadas se excluyen.
# ----------------------------------------------------------------------------
PATRON_SENSIBLE = re.compile(
    r"nombre|telefono|celular|correo|contacto|domicilio|direccion|geoloc"
    r"|latitud|longitud|latitude|longitude|altitude|precision|gps|poligono|polygon|shape",
    re.I)


def es_sensible(nombre_col: str) -> bool:
    n = _norm(str(nombre_col))
    if "encuestador" in n or "supervisor" in n or "enumerador" in n:
        return False  # personal de campo, no beneficiarios
    return bool(PATRON_SENSIBLE.search(n))


def quitar_sensibles(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not es_sensible(c)]]


# ----------------------------------------------------------------------------
# Preguntas clave para el análisis de valores faltantes
# ----------------------------------------------------------------------------
CLAVES_SCALL = [
    ("Fecha de la entrevista", "Fecha de la entrevista"),
    ("Distrito", "Distrito"),
    ("Miembros del hogar", "personas habitan al dia de hoy"),
    ("Mujeres en el hogar", "cuantas son mujeres"),
    ("Año de instalación del SCALL", "año que le instalaron"),
    ("Uso actual del agua del SCALL", "el hogar usa agua del SCALL"),
    ("Meses al año que aporta agua", ["cuantos meses aporta agua", "meses al año su scall aporta"]),
    ("Capacidad de almacenamiento", "capacidad total de almacenamiento"),
    ("Gasto en mantenimiento", "gasto estimado en mantenimiento"),
]
CLAVES_AGRI = [
    ("Fecha de la entrevista", "Fecha de la entrevista"),
    ("Distrito", "Distrito"),
    ("Edad del productor", "edad del productor"),
    ("Miembros del hogar", "personas habitan al dia de hoy"),
    ("N.º de parcelas", "TERRENOS o PARCELAS"),
    ("N.º de cultivos", "CULTIVOS tuvo en total"),
    ("Gasto en semilla", "compra de la SEMILLA"),
    ("Gasto en fertilizantes", "compra de FERTILIZANTES"),
    ("Ingreso por ventas", "ingreso total obtenido por la venta"),
    ("Fuente principal de ingresos", "principal fuente de ingresos"),
]


def vacios(serie: pd.Series) -> pd.Series:
    return serie.isna() | serie.astype(str).str.strip().isin(["", "nan", "None"])


def faltantes_por_fila(d: pd.DataFrame, claves) -> pd.Series:
    cols = [resolve(d, k) for _, k in claves]
    cols = [c for c in cols if c is not None]
    if not cols:
        return pd.Series(0, index=d.index)
    return pd.concat([vacios(d[c]) for c in cols], axis=1).sum(axis=1)


def flags_comunes(d: pd.DataFrame, f: pd.DataFrame, claves=None) -> pd.DataFrame:
    """Flags aplicables a cualquier módulo."""
    f["G01 Duración <15 min"] = duracion_min(d) < 15
    fe = pd.to_datetime(col(d, "Fecha de la entrevista"), errors="coerce")
    f["G02 Fecha de entrevista vacía"] = fe.isna()
    nom = col(d, K_NOMBRE).astype(str).str.strip().str.upper()
    f["G03 Nombre de prueba"] = nom.isin(["NOMBRE", "PRUEBA", "TEST", "XXX", "N/A"])
    if claves:
        f["G04 ≥3 preguntas clave sin dato"] = faltantes_por_fila(d, claves) >= 3
    return f


def distinciones(d: pd.DataFrame) -> pd.DataFrame:
    """NO son errores: encuestas dejadas en borrador y complementadas por
    teléfono. La fecha válida es siempre la registrada en Kobo."""
    n = pd.DataFrame(index=d.index)
    n["Completada en borrador (>240 min)"] = (duracion_min(d) > 240).fillna(False)
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
# Definición de flags
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
    f["F15 Días tanque lleno >365"] = skip(col(d, "dias le duraria un tanque lleno"), 9999) > 365
    f["F16 Centinela 9999/999 sin depurar"] = (
        (num(col(d, "dias le duraria un tanque lleno")) == 9999)
        | (num(col(d, ["tiempo actual ida y vuelta", "tiempo actual de ida y vuelta"])) == 9999)
        | (num(col(d, "capacidad total de almacenamiento")).isin([9999, 999]))
        | (num(col(d, "dias estuvo fuera de servicio")) == 999)
    )
    return flags_comunes(d, f, CLAVES_SCALL).fillna(False)


def flags_agricola(d: pd.DataFrame) -> pd.DataFrame:
    edad = num(col(d, "edad del productor"))
    gs_raw = num(col(d, "compra de la SEMILLA"))
    gf_raw = num(col(d, "compra de FERTILIZANTES"))
    ga_raw = num(col(d, "AGROQUIMICOS"))
    gm_raw = num(col(d, "mano de obra o jornales"))
    ing_raw = num(col(d, "ingreso total obtenido por la venta"))
    # 9999 es centinela de "no sabe": no cuenta como monto real
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
    return flags_comunes(d, f, CLAVES_AGRI).fillna(False)


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
# Diccionario de flags (explicación para el equipo)
# ----------------------------------------------------------------------------
FLAG_DESC = {
    "G01 Duración <15 min": "El tiempo entre que se abrió y se envió el formulario fue menor a 15 minutos. Una encuesta completa difícilmente se levanta tan rápido, así que puede estar incompleta o haberse llenado sin entrevistar realmente. Verificar con el encuestador si fue un reinicio o una prueba.",
    "G02 Fecha de entrevista vacía": "El campo 'Fecha de la entrevista' quedó sin llenar en Kobo. Esa fecha es la que usa el dashboard para medir el avance por día, así que sin ella la encuesta no aparece en la gráfica diaria. Pedir al encuestador que la complete.",
    "G03 Nombre de prueba": "El registro corresponde a un texto de prueba ('NOMBRE', 'PRUEBA', 'TEST', etc.). Casi seguro es un registro de práctica o capacitación que quedó en la base. Confirmar y eliminarlo antes del análisis.",
    "G04 ≥3 preguntas clave sin dato": "La encuesta tiene tres o más preguntas clave sin respuesta. Puede deberse a la lógica de salto del formulario, pero también a un problema de comprensión de la pregunta o a una encuesta incompleta. Revisar el registro en Kobo y la sección de valores faltantes del tablero.",
    "F01 Teléfono inválido": "El número de teléfono tiene letras, símbolos o más de 8 dígitos (el estándar en El Salvador es de 8). Un teléfono mal capturado impide recontactar al hogar para verificaciones o para complementar datos por llamada.",
    "F02 Hogar >10 personas": "El hogar reporta más de 10 miembros, un tamaño poco común. Puede ser real, pero también un error de dedo (ej. 12 en vez de 2). Confirmar el dato.",
    "F03 Mujeres > total": "El número de mujeres reportado es mayor que el total de personas del hogar, lo cual es imposible. Alguno de los dos números quedó mal capturado y hay que corregirlo.",
    "F04 Hombres+Mujeres ≠ Total": "La suma de hombres más mujeres no coincide con el total de miembros del hogar. Indica error de captura o de conteo en alguno de los tres campos; revisar cuál es el correcto.",
    "F05 Ayudantes > total hogar": "Se reportan más miembros ayudando en las actividades productivas que personas viviendo en el hogar. Es una inconsistencia lógica que requiere verificar ambos números.",
    "F06 Año SCALL < 2020": "El año de instalación del sistema de captación es anterior al periodo esperado de entregas del proyecto. Puede tratarse de un sistema previo (no de RECLIMA) o de un año mal recordado o mal digitado.",
    "F07 Tiempo acarreo sospechoso": "El tiempo de ida y vuelta para acarrear agua antes del proyecto es menor a 10 minutos o mayor a 4 horas. Los extremos suelen ser errores de unidad (horas vs. minutos) o estimaciones poco fiables.",
    "F08 Días agua >30": "Se reportan más de 30 días con agua disponible en el último mes, cuando el máximo posible es 30-31. Es un error de captura o de interpretación de la pregunta.",
    "F09 Litros SCALL >10,000": "El aporte anual de agua reportado supera los 10,000 litros, muy por encima de la capacidad típica de un SCALL domiciliar. Probable error de unidad o estimación exagerada.",
    "F10 Meses fuera 0-12": "Los meses al año que el SCALL aporta agua están fuera del rango 0 a 12, lo cual es imposible. Corregir el dato.",
    "F11 Tiempo actual sospechoso": "El tiempo actual de ida y vuelta para conseguir agua fuera del hogar es menor a 5 minutos o mayor a 24 horas. Los extremos sugieren error de unidad o de digitación.",
    "F12 Gasto agua atípico": "El gasto mensual en compra o transporte de agua es menor a $1 o mayor a $1,000. Montos así de extremos casi siempre son errores de captura (ej. centavos vs. dólares).",
    "F13 Almacenamiento <100 L": "La capacidad total de almacenamiento reportada es menor a 100 litros, muy por debajo de cualquier tanque SCALL real. Probable error de unidad o dato incompleto.",
    "F14 Gasto mant. atípico": "El gasto anual en mantenimiento es menor a $10 o mayor a $10,000. Fuera de ese rango lo esperable es un error de digitación o una interpretación distinta de la pregunta.",
    "F15 Días tanque lleno >365": "Los días que duraría un tanque lleno superan los 365, es decir más de un año con una sola llenada, lo cual no es plausible. Revisar si se entendió la pregunta.",
    "F16 Centinela 9999/999 sin depurar": "El registro contiene códigos 9999 o 999 que significan 'no sabe / no responde' en campos numéricos (tanque, tiempo, almacenamiento, días fuera de servicio). No son valores reales: hay que depurarlos antes de calcular promedios o totales.",
    "F02 Edad atípica": "La edad del productor es menor a 15 o mayor a 100 años. Puede ser un error de dedo o que se registró a la persona equivocada como productor principal.",
    "F03 Hogar >15 personas": "El hogar reporta más de 15 miembros, un tamaño excepcional. Puede ser real, pero conviene confirmar que no sea un error de captura.",
    "F04 Gasto semilla >$5,000": "El gasto en semilla supera los $5,000 en la temporada, muy alto para un pequeño productor. Verificar si es real (productor grande) o un error de monto.",
    "F05 Gasto fertilizantes >$5,000": "El gasto en fertilizantes supera los $5,000 en la temporada. Igual que con semilla: puede ser real en casos excepcionales, pero lo usual es un error de captura.",
    "F06 Gasto agroquímicos >$5,000": "El gasto en agroquímicos (sin contar fertilizantes) supera los $5,000. Es un monto atípico para la escala de los beneficiarios; confirmar.",
    "F07 Gasto mano obra >$10,000": "El gasto en jornales o mano de obra contratada supera los $10,000 en la temporada. Verificar unidad y monto con el encuestador.",
    "F08 Ingreso ventas >$50,000": "El ingreso por venta de cultivos supera los $50,000, fuera de la escala esperada de los beneficiarios. Puede ser un error de digitación (un cero de más).",
    "F09 Ingresos >> gastos x10": "El ingreso por ventas es más de 10 veces la suma de todos los gastos productivos. Una rentabilidad así de alta es improbable y sugiere que algún monto (ingreso o gastos) está mal capturado.",
    "F10 Más parcelas que cultivos": "Se reportan más parcelas que cultivos en total, lo que implicaría parcelas enteras sin ningún cultivo. Es posible, pero conviene confirmar que no se invirtieron los dos números.",
    "F11 Centinela 9999 en gastos/ingreso": "Algún campo de gastos o de ingreso tiene el código 9999 que significa 'no sabe'. No es un monto real: hay que depurarlo antes de sumar o promediar y, de ser posible, recuperar el dato.",
    "P01 Área parcela = 0": "La parcela tiene área cero o negativa, lo cual no es posible si se cultivó en ella. Falta el dato real de superficie; recuperarlo con el encuestador.",
    "P02 Área parcela >500 mz": "La parcela supera las 500 manzanas, una extensión enorme para el perfil de los beneficiarios. Casi seguro es un error de unidad o de digitación.",
    "P03 Árboles plantados > existentes": "Se reportan más árboles plantados en los últimos 12 meses que árboles existentes en total en la parcela. Como los plantados deberían estar incluidos en los existentes, hay una inconsistencia que revisar.",
    "P04 Qty fertilizante = 9999": "La cantidad de fertilizante tiene el código 9999 de 'no sabe'. Depurar antes de usar el dato y, si se puede, recuperarlo.",
    "P05 Qty agroquímico = 9999": "La cantidad de agroquímico tiene el código 9999 de 'no sabe'. Depurar antes de usar el dato y, si se puede, recuperarlo.",
    "C01 Cosechada > Sembrada": "El área cosechada es mayor que el área sembrada (medidas en la misma unidad), lo cual no es posible. Uno de los dos valores está mal capturado.",
    "C02 Producción=0, área>0": "Se sembró un área mayor a cero pero la producción reportada es cero. Puede ser una pérdida total real (sequía, plaga) o un dato faltante; conviene confirmar cuál de las dos.",
    "C03 Qty semilla = 9999": "La cantidad de semilla tiene el código 9999 de 'no sabe'. No es un valor real; depurar y de ser posible recuperar el dato.",
    "C04 Área sembrada >500 mz": "El área sembrada del cultivo supera las 500 manzanas, fuera de toda escala esperada. Error de unidad o de digitación casi seguro.",
}


def diccionario_flags(nombres):
    con_desc = [(n, FLAG_DESC[n]) for n in nombres if n in FLAG_DESC]
    if not con_desc:
        return
    with st.expander("ℹ️ ¿Qué significa cada flag? (guía para el equipo)"):
        for nombre, desc in con_desc:
            st.markdown(f"**{nombre}** — {desc}")


# ----------------------------------------------------------------------------
# Render del dashboard
# ----------------------------------------------------------------------------
CHART = dict(height=340)


def kpis_y_desgloses(d: pd.DataFrame, flags: pd.DataFrame, notas: pd.DataFrame):
    enum_c = resolve(d, K_ENUM)
    fecha_c = resolve(d, "Fecha de la entrevista")
    distr_c = resolve(d, "Distrito", exact=True) or resolve(d, "Distrito")
    fechas = pd.to_datetime(d[fecha_c], errors="coerce") if fecha_c else pd.Series(dtype="datetime64[ns]")

    dur = duracion_min(d)
    n_flag = flags.any(axis=1).sum()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Encuestas", len(d))
    c2.metric("Encuestadores", d[enum_c].nunique() if enum_c else "—")
    c3.metric("Distritos", d[distr_c].nunique() if distr_c else "—")
    c4.metric("Días de campo", fechas.dt.date.nunique())
    c5.metric("Con ≥1 flag", f"{n_flag} ({n_flag/len(d):.0%})" if len(d) else "0")
    c6.metric("Duración mediana", f"{dur.median():.0f} min" if dur.notna().any() else "—")

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

    if not notas.empty and notas.any().any():
        partes = [f"{int(notas[c].sum())} {c.lower()}" for c in notas.columns if notas[c].sum() > 0]
        st.info("📱 **Distinciones del levantamiento** (no son errores — encuestas "
                "que quedan en borrador y se complementan por teléfono): "
                + " · ".join(partes) + ".")

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
                     title="Encuestas por día (según fecha registrada en Kobo)", height=300)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")


def _pct(serie: pd.Series, condicion) -> str:
    v = serie.dropna().astype(str).str.strip()
    v = v[v != ""]
    if v.empty:
        return "—"
    return f"{condicion(v).mean():.0%}"


def seccion_preliminares(d: pd.DataFrame, modulo: str):
    st.subheader("🧪 Datos preliminares (resultados)")
    st.caption("Calculados sobre la base tal como está — sin depuración final ni "
               "ponderación. Referencia de avance, no resultados de la evaluación.")
    c1, c2, c3, c4 = st.columns(4)
    if modulo == "SCALL":
        c1.metric("Hogares que usan agua del SCALL",
                  _pct(col(d, "el hogar usa agua del SCALL"),
                       lambda v: v.str.startswith("Sí")))
        meses = skip(col(d, ["cuantos meses aporta agua",
                             "meses al año su scall aporta"]), 99)
        c2.metric("Meses al año que aporta agua (prom.)",
                  f"{meses.mean():.1f}" if meses.notna().any() else "—")
        c3.metric("Perciben mejor disponibilidad de agua",
                  _pct(col(d, "la disponibilidad de agua para beber es"),
                       lambda v: v.isin(["Mejor", "Mucho mejor"])))
        c4.metric("Creen que funcionará en 2 años",
                  _pct(col(d, "seguira funcionando dentro de 2"),
                       lambda v: v == "Sí"))
    else:
        c1.metric("Productoras mujeres",
                  _pct(col(d, "sexo del productor"), lambda v: v == "Mujer"))
        parc = num(col(d, "TERRENOS o PARCELAS"))
        cult = num(col(d, "CULTIVOS tuvo en total"))
        c2.metric("Parcelas / cultivos por productor (prom.)",
                  f"{parc.mean():.1f} / {cult.mean():.1f}"
                  if parc.notna().any() else "—")
        ing = skip(col(d, "ingreso total obtenido por la venta"), 9999)
        c3.metric("Vendieron parte de su cosecha",
                  f"{(ing > 0).sum() / ing.notna().sum():.0%}"
                  if ing.notna().any() else "—")
        fuente = col(d, "principal fuente de ingresos").dropna().astype(str)
        if len(fuente):
            top = fuente.value_counts()
            c4.metric("Principal fuente de ingresos (moda)",
                      top.index[0],
                      f"{top.iloc[0] / len(fuente):.0%} de los hogares",
                      delta_color="off")
        else:
            c4.metric("Principal fuente de ingresos (moda)", "—")


def seccion_faltantes(d: pd.DataFrame, claves, modulo: str):
    st.subheader("🕳 Valores faltantes en preguntas clave")
    st.caption("Encuestas sin dato en preguntas clave, por pregunta y por encuestador. "
               "Nota: parte de los vacíos puede deberse a la lógica de salto del "
               "formulario, no a errores — el valor de esta tabla está en detectar "
               "patrones (una pregunta o un encuestador con faltantes sistemáticos).")
    filas = []
    for etiqueta, key in claves:
        c = resolve(d, key)
        if c is None:
            filas.append((etiqueta, "—", "columna no encontrada"))
            continue
        v = vacios(d[c])
        filas.append((etiqueta, int(v.sum()), f"{v.mean():.0%}"))
    t1 = pd.DataFrame(filas, columns=["Pregunta clave", "Sin dato", "% del total"])
    a, b = st.columns(2)
    a.markdown("**Por pregunta**")
    a.dataframe(t1, hide_index=True, width="stretch")
    enum_c = resolve(d, K_ENUM)
    if enum_c:
        n_falt = faltantes_por_fila(d, claves)
        t2 = pd.DataFrame({
            "Encuestas": d.groupby(d[enum_c]).size(),
            "Prom. preguntas clave sin dato": n_falt.groupby(d[enum_c]).mean().round(1),
            "Encuestas con ≥3 sin dato": (n_falt >= 3).groupby(d[enum_c]).sum().astype(int),
        }).sort_values("Prom. preguntas clave sin dato", ascending=False)
        b.markdown("**Por encuestador**")
        b.dataframe(t2, width="stretch")


def tabla_flags(d: pd.DataFrame, flags: pd.DataFrame):
    st.subheader("⚠ Registros con flags")
    st.caption("Sin datos personales: use el ID para ubicar el registro en Kobo.")
    base = pd.DataFrame({
        "ID": id_encuesta(d),
        "Encuestador": col(d, K_ENUM),
        "Fecha": pd.to_datetime(col(d, "Fecha de la entrevista"), errors="coerce").dt.strftime("%d/%m/%Y"),
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
    st.subheader("👤 Ficha de la encuesta")
    st.caption("Identificada solo por ID (sin datos personales). Los campos de "
               "nombres, teléfonos, direcciones y coordenadas están excluidos "
               "del tablero; consúltelos directamente en Kobo si es necesario.")
    ids = id_encuesta(d)
    elegido = st.selectbox("Seleccione la encuesta por ID",
                           "ID " + ids, key=f"ficha_{modulo}")
    i = ids.index[("ID " + ids) == elegido][0]
    fila = quitar_sensibles(d.to_frame().T if isinstance(d, pd.Series) else d).loc[i]

    activos = [f for f in flags.columns if flags.at[i, f]]
    if activos:
        st.warning("⚠ Flags de este registro: " + ", ".join(activos))
    else:
        st.success("✔ Sin flags de calidad en este registro.")

    ficha = fila.dropna()
    ficha = ficha[ficha.astype(str).str.strip() != ""]
    ficha.index.name = "Pregunta"
    st.dataframe(ficha.rename("Respuesta").to_frame().astype(str),
                 width="stretch", height=450)

    # Parcelas y cultivos de la encuesta (Agrícola)
    idx_c = resolve(d, "_index", exact=True)
    if book is not None and idx_c is not None:
        rp = pick_sheet(book, "roster_parcela")
        if rp is not None and resolve(rp, "_parent_index", exact=True):
            mias = rp[rp["_parent_index"] == d.at[i, idx_c]]
            if len(mias):
                st.markdown(f"**🌾 Parcelas de esta encuesta ({len(mias)})**")
                st.dataframe(quitar_sensibles(mias).dropna(axis=1, how="all").astype(str),
                             width="stretch", hide_index=True)
                idxs = set(mias["_index"]) if "_index" in mias.columns else set()
                for sh, tit in [("roster_cultivos", "🌽 Cultivos — parcela principal"),
                                ("roster_cultivo", "🌽 Cultivos — otras parcelas")]:
                    rc = pick_sheet(book, sh)
                    if rc is not None and "_parent_index" in rc.columns and idxs:
                        c = rc[rc["_parent_index"].isin(idxs)]
                        if len(c):
                            st.markdown(f"**{tit} ({len(c)})**")
                            st.dataframe(quitar_sensibles(c).dropna(axis=1, how="all").astype(str),
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
    mask = fl.any(axis=1)
    det = pd.DataFrame(index=r.index)
    for etiqueta, key in cols_id:
        es_exacta = isinstance(key, str) and (key.startswith("_") or key in ("CL", "CLs", "M1_Q6b"))
        det[etiqueta] = col(r, key, exact=es_exacta)
    parent_c = resolve(r, "_parent_index", exact=True)
    if parent_c:
        parents = r[parent_c]
        if via_sheet:  # el roster cuelga de otro roster (ej. cultivo → parcela)
            v = pick_sheet(book, via_sheet)
            if v is not None and "_index" in v.columns and "_parent_index" in v.columns:
                parents = parents.map(v.set_index(v["_index"])["_parent_index"])
        det["ID Encuesta"] = parents  # rastrear en Kobo por _index
    show = pd.concat([quitar_sensibles(det[mask]),
                      fl[mask].replace({True: "⚠", False: ""})], axis=1)
    st.dataframe(show, width="stretch", hide_index=True)


def render_modulo(book: dict, esperado: str, nombre: str):
    detectado, d = detect_module(book)
    if detectado and detectado != esperado:
        st.error(f"Este archivo parece del módulo **{detectado}**, no de {nombre}. "
                 "Súbelo en la otra pestaña.")
        return
    flags = flags_scall(d) if esperado == "SCALL" else flags_agricola(d)
    claves = CLAVES_SCALL if esperado == "SCALL" else CLAVES_AGRI
    notas = distinciones(d)
    kpis_y_desgloses(d, flags, notas)
    seccion_preliminares(d, esperado)
    nombres_flags = list(flags.columns)
    if esperado == "AGRICOLA":
        nombres_flags += [n for n in FLAG_DESC
                          if n[:1] in ("P", "C") and n[1:3].isdigit()]
    diccionario_flags(nombres_flags)
    tabla_flags(d, flags)
    seccion_faltantes(d, claves, esperado)
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
# Reporte acumulado para supervisores (.docx) — sin datos personales
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
    ids = id_encuesta(d)
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
        "Proyecto RECLIMA — Evaluación final, corredor seco de El Salvador. "
        "Este reporte no contiene datos personales: los registros se identifican "
        "por su ID de Kobo (_index)."
    )

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

    doc.add_heading("2. Flags a atender", 1)
    resumen = flags.sum().sort_values(ascending=False)
    resumen = resumen[resumen > 0]
    if resumen.empty:
        doc.add_paragraph("No hay flags de calidad activos. ✔")
    else:
        doc.add_paragraph(
            "Casos por flag, con el ID de Kobo de cada registro afectado y su "
            "encuestador (para ubicar y verificar en Kobo):")
        filas = []
        for flag_name, n in resumen.items():
            afectados = []
            for i in d.index[flags[flag_name]]:
                e = d.at[i, enum_c] if enum_c else ""
                afectados.append(f"#{ids.loc[i]} ({e})" if e else f"#{ids.loc[i]}")
            filas.append((flag_name, int(n), "; ".join(afectados)))
        _doc_tabla(doc, filas, ["Flag", "Casos", "IDs afectados (encuestador)"])
        doc.add_paragraph("Descripción de los flags activos:", style="Intense Quote")
        for flag_name in resumen.index:
            if flag_name in FLAG_DESC:
                pr = doc.add_paragraph()
                pr.add_run(flag_name + ": ").bold = True
                pr.add_run(FLAG_DESC[flag_name])
    if extra:
        doc.add_paragraph(extra)

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
# Datos publicados por el administrador (carpeta data/ del repo de GitHub)
# ----------------------------------------------------------------------------
DATA_DIR = "data"
ARCHIVO_PUBLICADO = {"SCALL": "scall.xlsx", "AGRICOLA": "agricola.xlsx"}


@st.cache_data(show_spinner=False)
def libro_publicado(nombre_archivo: str, mtime: float):
    with open(os.path.join(DATA_DIR, nombre_archivo), "rb") as fh:
        return load_book(fh.read())


def pestana_modulo(esperado: str, nombre: str, key: str):
    publicado = os.path.join(DATA_DIR, ARCHIVO_PUBLICADO[esperado])
    with st.expander("🔄 Ver otra base (opcional — no reemplaza la publicada)"):
        up = st.file_uploader(f"Base {nombre} (.xlsx)", type="xlsx", key=f"up_{key}")
    if up:
        st.caption("Mostrando la base subida en esta sesión (no queda guardada).")
        render_modulo(load_book(up.getvalue()), esperado, nombre)
    elif os.path.exists(publicado):
        mt = os.path.getmtime(publicado)
        st.caption(f"📌 Base publicada por el administrador — actualizada el "
                   f"{datetime.fromtimestamp(mt):%d/%m/%Y %H:%M}.")
        render_modulo(libro_publicado(ARCHIVO_PUBLICADO[esperado], mt), esperado, nombre)
    else:
        st.info(f"Aún no hay base publicada para {nombre}. El administrador debe subir "
                f"`data/{ARCHIVO_PUBLICADO[esperado]}` al repo de GitHub, o puedes subir "
                "un archivo en la sección de arriba para verlo en esta sesión.")


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
if check_password():
    st.title("🌱 Dashboard de monitoreo RECLIMA")
    st.caption("Evaluación final — corredor seco de El Salvador. Los datos los "
               "publica el administrador y se actualizan automáticamente. "
               "El tablero no muestra datos personales de los entrevistados.")

    tab_scall, tab_agri = st.tabs(["💧 SCALL", "🌾 Agrícola"])

    with tab_scall:
        pestana_modulo("SCALL", "SCALL", "scall")

    with tab_agri:
        pestana_modulo("AGRICOLA", "Agrícola", "agri")
