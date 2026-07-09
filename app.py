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


# La variable de identificación oficial es PRODUCTOS-ID ENCUESTA
# ("Identificador de encuestado" en la exportación). Si está vacía, se usa
# el índice de Kobo como respaldo, marcado con "k".
K_ID = ["PRODUCTOS-ID ENCUESTA", "PRODUCTOS-ID", "ID ENCUESTA",
        "Identificador de encuestado"]


def id_encuesta(d: pd.DataFrame) -> pd.Series:
    """ID para rastrear el registro (sin datos personales)."""
    idx_c = resolve(d, "_index", exact=True)
    kobo = (d[idx_c].astype("Int64").astype(str) if idx_c
            else (d.index + 1).astype(str))
    id_c = resolve(d, K_ID)
    if id_c is not None:
        ids = d[id_c].astype(str).str.strip()
        sin_id = ids.isin(["", "nan", "None"])
        return ids.mask(sin_id, "s/ID·k" + kobo)
    return kobo


# ----------------------------------------------------------------------------
# Protección de datos personales (solicitud FAO): el tablero solo muestra el
# ID de Kobo; nombres, teléfonos, direcciones y coordenadas se excluyen.
# ----------------------------------------------------------------------------
PATRON_SENSIBLE = re.compile(
    r"nombre|telefono|celular|correo|contacto|domicilio|direccion|geoloc"
    r"|latitud|longitud|latitude|longitude|altitude|precision|gps|poligono|polygon|shape|coordenadas",
    re.I)


def es_sensible(nombre_col: str) -> bool:
    n = _norm(str(nombre_col))
    if "encuestador" in n or "supervisor" in n or "enumerador" in n:
        return False  # personal de campo, no beneficiarios
    return bool(PATRON_SENSIBLE.search(n))


def quitar_sensibles(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not es_sensible(c)]]


def vacios(serie: pd.Series) -> pd.Series:
    return serie.isna() | serie.astype(str).str.strip().isin(["", "nan", "None"])


# ----------------------------------------------------------------------------
# Preguntas clave (SCALL: lista definida por ADEPRO/FAO, jul 2026)
# ----------------------------------------------------------------------------
OBLIGATORIAS_SCALL = [
    ("Tiene instalado SCALL", "Tiene instalado en su hogar un sistema"),
    ("Nombre de pila del jefe de hogar", "nombre de pila o nombre propio"),
    ("Sexo del jefe de hogar (M1)", "sexo de ${M1_Q2}"),
    ("Años cumplidos (M1)", "años cumplidos tiene ${M1_Q2}"),
    ("Grado escolar (M1)", "grado o año escolar más alto"),
    ("Pertenece a Pueblo Indígena", "pertenece o se identifica con un Pueblo Indígena"),
    ("Personas en el hogar", "personas habitan al día de hoy"),
    ("Mujeres en el hogar", "cuántas son mujeres"),
    ("res_miembros", "res_miembros"),
    ("Hombres en el hogar", ["cuántos son hombres", "cuántas son hombres"]),
    ("Miembros de 15 años o menos", "15 años o menos"),
    ("Ayudantes del productor", "ayudantes del productor"),
    ("Fuente principal de ingresos", "principal fuente de ingresos"),
]
CLAVES_SCALL = [
    ("Fecha de la entrevista", "Fecha de la entrevista"),
    ("Nombre del encuestador", K_ENUM),
    ("Cantón", "Cantón"),
    ("Caserío", "Caserío"),
    ("Identificador de encuestado", "Identificador de encuestado"),
    ("Geolocalización del hogar", "Registrar geolocalización del hogar"),
    ("Relación con el jefe de hogar", "relación con el jefe de hogar"),
    ("Edad del jefe de hogar", ["edad de la jefa o jefe de hogar", "edad del productor"]),
] + OBLIGATORIAS_SCALL

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


def faltantes_por_fila(d: pd.DataFrame, claves) -> pd.Series:
    cols = [resolve(d, k) for _, k in claves]
    cols = [c for c in cols if c is not None]
    if not cols:
        return pd.Series(0, index=d.index)
    return pd.concat([vacios(d[c]) for c in cols], axis=1).sum(axis=1)


def flags_comunes(d: pd.DataFrame, f: pd.DataFrame, claves=None) -> pd.DataFrame:
    """Flags aplicables al módulo Agrícola (SCALL usa su propia lista)."""
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
# Flags SCALL — lista definitiva acordada con FAO (jul 2026)
# ----------------------------------------------------------------------------
def flags_scall(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    f["S01 Sin nombre de encuestador"] = vacios(col(d, K_ENUM))
    edad = num(col(d, ["edad de la jefa o jefe de hogar", "edad del productor"]))
    f["S02 Edad <18, >90 o faltante"] = edad.isna() | (edad < 18) | (edad > 90)
    f["S03 Cantón o caserío vacío"] = (vacios(col(d, "Cantón", exact=True))
                                       | vacios(col(d, "Caserío", exact=True)))
    f["S04 Sin identificador del encuestado"] = vacios(col(d, "Identificador de encuestado"))
    f["S05 Sin geolocalización del hogar"] = vacios(col(d, "Registrar geolocalización del hogar"))
    f["S06 Sin relación con el jefe de hogar"] = vacios(col(d, "relación con el jefe de hogar"))

    # S07: preguntas obligatorias del módulo de hogar sin respuesta.
    # Las de Pueblo Indígena son condicionales: solo cuentan si aplica la rama.
    faltas = faltantes_por_fila(d, OBLIGATORIAS_SCALL)
    pert = col(d, "pertenece o se identifica con un Pueblo Indígena").astype(str)
    cond_pueblo = vacios(col(d, "A qué Pueblo Indígena")) & pert.str.startswith("Sí")
    pueblo = col(d, "A qué Pueblo Indígena").astype(str)
    cond_esp = vacios(col(d, "Especificar Pueblo Indígena")) & pueblo.str.contains("Otro", na=False)
    f["S07 Obligatorias del hogar sin respuesta"] = (faltas >= 1) | cond_pueblo | cond_esp

    s_an = col(d, "año que le instalaron")
    if pd.api.types.is_datetime64_any_dtype(s_an):
        an = s_an.dt.year.astype(float)  # Kobo exporta el año como fecha
    else:
        an = num(s_an)
    f["S08 Año SCALL ≤2022 o no es un año"] = an.notna() & ((an <= 2022) | (an > 2100))
    f["S09 Días tanque lleno >365"] = num(col(d, "dias le duraria un tanque lleno")) > 365
    f["S10 Meses de aporte >12"] = num(col(d, ["meses al año su scall aporta",
                                               "cuantos meses aporta agua"])) > 12

    # S11: obligatorias que no se desplegaron (posible falla de lógica del
    # formulario), desglosadas para ver exactamente cuál pregunta faltó
    recibio = col(d, "Recibió capacitación en instalación").astype(str)
    f["S11a No desplegada: quién realiza el mantenimiento"] = vacios(
        col(d, "Quién realiza el mantenimiento habitual"))
    f["S11b No desplegada: tipo de capacitación"] = (
        vacios(col(d, "Qué tipo de capacitación recibió"))
        & recibio.str.startswith("Sí"))
    f["S11c No desplegada: componentes observados"] = vacios(
        col(d, "componentes observó a partir de su inspección visual"))
    f["S11d No desplegada: frecuencia de limpieza"] = vacios(
        col(d, "frecuencia se da limpieza"))
    fe = pd.to_datetime(col(d, "Fecha de la entrevista"), errors="coerce")
    f["S12 Sin fecha de entrevista"] = fe.isna()
    return f.fillna(False)


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
    # SCALL (lista definitiva)
    "S01 Sin nombre de encuestador": "El registro no tiene el nombre del encuestador. Sin ese dato no se puede atribuir la encuesta ni dar seguimiento a la supervisión. Completarlo en Kobo.",
    "S02 Edad <18, >90 o faltante": "La edad del jefe o jefa de hogar falta o está fuera del rango 18-90 años. Un menor de edad no debería ser el entrevistado principal, y edades mayores a 90 suelen ser errores de digitación. Verificar el registro.",
    "S03 Cantón o caserío vacío": "El cantón o el caserío del hogar quedaron sin nombre. Son datos necesarios para ubicar el hogar en campo y para los desgloses territoriales; completarlos en Kobo.",
    "S04 Sin identificador del encuestado": "Falta el identificador del encuestado que entrega el supervisor. Sin este ID no se puede vincular la encuesta con el listado de beneficiarios. Es prioritario recuperarlo.",
    "S05 Sin geolocalización del hogar": "No se registró la coordenada GPS del hogar. La geolocalización es obligatoria para la verificación en campo; el encuestador debe capturarla al momento de la visita.",
    "S06 Sin relación con el jefe de hogar": "No se registró la relación del informante con el jefe o jefa de hogar. El dato es necesario para validar que el informante sea idóneo.",
    "S07 Obligatorias del hogar sin respuesta": "Una o más preguntas obligatorias del módulo de hogar están sin respuesta (instalación del SCALL, características del jefe de hogar, composición del hogar, fuente de ingresos). Si esto ocurre de forma masiva, el problema es la lógica de salto del formulario, no los encuestadores. Las preguntas de Pueblo Indígena solo cuentan cuando aplica la rama.",
    "S08 Año SCALL ≤2022 o no es un año": "El año de instalación del SCALL es menor o igual a 2022, o el valor capturado no es un año válido (por ejemplo, un dígito suelto o un código). Verificar con el hogar el año real de entrega del sistema.",
    "S09 Días tanque lleno >365": "Los días que duraría un tanque lleno superan los 365 — más de un año con una sola llenada no es plausible. Incluye valores centinela (9999) que deben depurarse.",
    "S10 Meses de aporte >12": "Los meses al año que el SCALL aporta agua superan los 12, lo cual es imposible. Incluye valores centinela (99) que deben depurarse.",
    "S11a No desplegada: quién realiza el mantenimiento": "La pregunta '¿Quién realiza el mantenimiento habitual?' quedó vacía. Es obligatoria, así que el vacío sugiere que el formulario no la desplegó — revisar la lógica de salto del cuestionario en Kobo.",
    "S11b No desplegada: tipo de capacitación": "La pregunta '¿Qué tipo de capacitación recibió?' quedó vacía en un hogar que SÍ reportó haber recibido capacitación. La rama debió desplegarse; revisar la lógica del cuestionario.",
    "S11c No desplegada: componentes observados": "La verificación visual del encuestador ('¿Qué componentes observó a partir de su inspección visual?') quedó vacía. Es un paso obligatorio de la visita; revisar si el formulario la desplegó o si el encuestador la omitió.",
    "S11d No desplegada: frecuencia de limpieza": "La pregunta '¿Con qué frecuencia se da limpieza a canaletas, tanque o filtros?' quedó vacía. Es obligatoria del bloque de mantenimiento; revisar la lógica del cuestionario.",
    "S12 Sin fecha de entrevista": "El campo 'Fecha de la entrevista' quedó sin llenar en Kobo. Esa fecha es la que usa el tablero para medir el avance por día; sin ella la encuesta no aparece en la gráfica diaria.",
    # Comunes (módulo Agrícola)
    "G01 Duración <15 min": "El tiempo entre que se abrió y se envió el formulario fue menor a 15 minutos. Una encuesta completa difícilmente se levanta tan rápido, así que puede estar incompleta o haberse llenado sin entrevistar realmente.",
    "G02 Fecha de entrevista vacía": "El campo 'Fecha de la entrevista' quedó sin llenar en Kobo. Esa fecha es la que usa el tablero para medir el avance por día. Pedir al encuestador que la complete.",
    "G03 Nombre de prueba": "El registro corresponde a un texto de prueba ('NOMBRE', 'PRUEBA', 'TEST', etc.). Casi seguro es un registro de práctica que quedó en la base. Confirmar y eliminarlo antes del análisis.",
    "G04 ≥3 preguntas clave sin dato": "La encuesta tiene tres o más preguntas clave sin respuesta. Puede deberse a la lógica de salto del formulario, pero también a una encuesta incompleta. Revisar el registro en Kobo y la sección de valores faltantes.",
    # Agrícola
    "F01 Teléfono inválido": "El número de teléfono tiene letras, símbolos o más de 8 dígitos (el estándar en El Salvador es de 8). Un teléfono mal capturado impide recontactar al hogar para verificaciones.",
    "F02 Edad atípica": "La edad del productor es menor a 15 o mayor a 100 años. Puede ser un error de dedo o que se registró a la persona equivocada como productor principal.",
    "F03 Hogar >15 personas": "El hogar reporta más de 15 miembros, un tamaño excepcional. Puede ser real, pero conviene confirmar que no sea un error de captura.",
    "F04 Gasto semilla >$5,000": "El gasto en semilla supera los $5,000 en la temporada, muy alto para un pequeño productor. Verificar si es real (productor grande) o un error de monto.",
    "F05 Gasto fertilizantes >$5,000": "El gasto en fertilizantes supera los $5,000 en la temporada. Puede ser real en casos excepcionales, pero lo usual es un error de captura.",
    "F06 Gasto agroquímicos >$5,000": "El gasto en agroquímicos (sin contar fertilizantes) supera los $5,000. Es un monto atípico para la escala de los beneficiarios; confirmar.",
    "F07 Gasto mano obra >$10,000": "El gasto en jornales o mano de obra contratada supera los $10,000 en la temporada. Verificar unidad y monto con el encuestador.",
    "F08 Ingreso ventas >$50,000": "El ingreso por venta de cultivos supera los $50,000, fuera de la escala esperada de los beneficiarios. Puede ser un error de digitación (un cero de más).",
    "F09 Ingresos >> gastos x10": "El ingreso por ventas es más de 10 veces la suma de todos los gastos productivos. Una rentabilidad así de alta es improbable y sugiere que algún monto está mal capturado.",
    "F10 Más parcelas que cultivos": "Se reportan más parcelas que cultivos en total, lo que implicaría parcelas enteras sin ningún cultivo. Es posible, pero conviene confirmar que no se invirtieron los dos números.",
    "F11 Centinela 9999 en gastos/ingreso": "Algún campo de gastos o de ingreso tiene el código 9999 que significa 'no sabe'. No es un monto real: hay que depurarlo antes de sumar o promediar.",
    # Parcelas
    "P01 Área parcela = 0": "La parcela tiene área cero o negativa, lo cual no es posible si se cultivó en ella. Falta el dato real de superficie; recuperarlo con el encuestador.",
    "P02 Área parcela >500 mz": "La parcela supera las 500 manzanas, una extensión enorme para el perfil de los beneficiarios. Casi seguro es un error de unidad o de digitación.",
    "P03 Árboles plantados > existentes": "Se reportan más árboles plantados en los últimos 12 meses que árboles existentes en total en la parcela. Como los plantados deberían estar incluidos en los existentes, hay una inconsistencia que revisar.",
    "P04 Qty fertilizante = 9999": "La cantidad de fertilizante tiene el código 9999 de 'no sabe'. Depurar antes de usar el dato y, si se puede, recuperarlo.",
    "P05 Qty agroquímico = 9999": "La cantidad de agroquímico tiene el código 9999 de 'no sabe'. Depurar antes de usar el dato y, si se puede, recuperarlo.",
    # Cultivos
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


ESTILOS_MAPA = {"⬜ Blanco": "white-bg", "🩶 Claro": "carto-positron",
                "🗺 Calles": "open-street-map"}


def selector_estilo(key: str) -> str:
    op = st.radio("Fondo del mapa", list(ESTILOS_MAPA), horizontal=True, key=key)
    return ESTILOS_MAPA[op]


def seccion_mapa(d: pd.DataFrame, modulo: str = ""):
    """Mapa de puntos GPS para control de calidad de campo. Los puntos se
    identifican SOLO por ID (sin nombres). Uso interno del equipo."""
    lat_c = resolve(d, "_latitude")
    lon_c = resolve(d, "_longitude")
    if lat_c is None or lon_c is None:
        return
    pts = pd.DataFrame({
        "lat": num(d[lat_c]),
        "lon": num(d[lon_c]),
        "ID": id_encuesta(d),
        "Encuestador": col(d, K_ENUM).astype(str),
        "Distrito": col(d, "Distrito").astype(str),
    }).dropna(subset=["lat", "lon"])
    if pts.empty:
        return
    st.subheader("🗺 Mapa de puntos GPS (control de calidad)")
    st.caption("Puntos identificados únicamente por ID, para verificar cobertura "
               "y ubicaciones atípicas. Uso interno del equipo — no difundir "
               "capturas de este mapa fuera del proyecto.")
    # El Salvador: lat 12.9–14.5, lon -90.2 – -87.6
    fuera = ~(pts["lat"].between(12.9, 14.5) & pts["lon"].between(-90.2, -87.6))
    if fuera.any():
        st.warning("⚠ Punto(s) fuera de El Salvador — revisar GPS: IDs "
                   + ", ".join("#" + i for i in pts.loc[fuera, "ID"]))
    estilo = selector_estilo(f"estilo_pts_{modulo}")
    try:
        fig = px.scatter_map(pts, lat="lat", lon="lon", color="Encuestador",
                             hover_name="ID", hover_data={"lat": False, "lon": False,
                                                          "Distrito": True},
                             zoom=8, height=480)
        fig.update_layout(map_style=estilo,
                          margin=dict(l=0, r=0, t=10, b=0))
    except Exception:  # versiones previas de plotly
        fig = px.scatter_mapbox(pts, lat="lat", lon="lon", color="Encuestador",
                                hover_name="ID", zoom=8, height=480)
        fig.update_layout(mapbox_style=estilo,
                          margin=dict(l=0, r=0, t=10, b=0))
    fig.update_traces(marker=dict(size=10))
    st.plotly_chart(fig, width="stretch")
    sin_gps = len(d) - len(pts)
    if sin_gps > 0:
        st.caption(f"{sin_gps} encuesta(s) sin coordenadas no aparecen en el mapa.")


# ----------------------------------------------------------------------------
# Polígonos de parcela (geoshape de Kobo): visualización y control de calidad
# ----------------------------------------------------------------------------
M2_POR_MANZANA = 6989.0
M2_POR_TAREA = 437.0


def parse_geoshape(s):
    """'lat lon alt prec;lat lon alt prec;...' -> lista de (lat, lon, prec)."""
    pts = []
    for seg in str(s).split(";"):
        p = seg.strip().split()
        if len(p) >= 2:
            try:
                prec = float(p[3]) if len(p) > 3 else np.nan
                pts.append((float(p[0]), float(p[1]), prec))
            except ValueError:
                pass
    return pts


def area_m2(pts) -> float:
    """Área del polígono (m²) por fórmula del polígono en proyección local."""
    import math
    if len(pts) < 3:
        return 0.0
    lat0 = sum(p[0] for p in pts) / len(pts)
    xs = [p[1] * 111320 * math.cos(math.radians(lat0)) for p in pts]
    ys = [p[0] * 110540 for p in pts]
    a = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2


def seccion_poligonos(d: pd.DataFrame, book: dict):
    geo_c = resolve(d, "coordenadas de la esquina de la parcela")
    if geo_c is None:
        return
    ids = id_encuesta(d)
    idx_c = resolve(d, "_index", exact=True)
    rp = pick_sheet(book, "roster_parcela")

    regs, shapes = [], []
    for i in d.index:
        v = d.at[i, geo_c]
        if pd.isna(v) or ";" not in str(v):
            continue
        pts = parse_geoshape(v)
        if not pts:
            continue
        a_m2 = area_m2(pts)
        precs = [p[2] for p in pts if not pd.isna(p[2])]
        prec_media = float(np.mean(precs)) if precs else np.nan
        # vértices consecutivos idénticos (encuestador sin moverse)
        dup = sum(1 for k in range(1, len(pts))
                  if (pts[k][0], pts[k][1]) == (pts[k-1][0], pts[k-1][1]))
        # área reportada de la primera parcela del productor (referencia)
        rep_txt, ratio = "—", np.nan
        if rp is not None and idx_c is not None and "_parent_index" in rp.columns:
            mias = rp[rp["_parent_index"] == d.at[i, idx_c]]
            if len(mias):
                ar = num(mias[resolve(mias, "el area de")]).iloc[0] if resolve(mias, "el area de") else np.nan
                un = str(mias[resolve(mias, "M1_Q6b", exact=True)].iloc[0]) if resolve(mias, "M1_Q6b", exact=True) else ""
                if not pd.isna(ar):
                    rep_txt = f"{ar:g} {un}"
                    factor = {"Manzanas": M2_POR_MANZANA, "Tareas": M2_POR_TAREA}.get(un)
                    if factor and ar > 0:
                        ratio = a_m2 / (ar * factor)
        alertas = []
        if len(pts) < 3:
            alertas.append("menos de 3 vértices")
        if a_m2 < 50:
            alertas.append("área ≈ 0")
        if not pd.isna(prec_media) and prec_media > 15:
            alertas.append(f"precisión GPS mala ({prec_media:.0f} m)")
        if dup > 0:
            alertas.append(f"{dup} vértice(s) duplicado(s)")
        if not pd.isna(ratio) and (ratio > 3 or ratio < 1/3):
            alertas.append(f"difiere del área reportada (x{ratio:.1f})")
        regs.append({
            "ID": ids.loc[i],
            "Vértices": len(pts),
            "Área medida (mz)": round(a_m2 / M2_POR_MANZANA, 2),
            "Área medida (m²)": round(a_m2),
            "Área reportada (1ª parcela)": rep_txt,
            "Precisión GPS media (m)": round(prec_media, 1) if not pd.isna(prec_media) else "—",
            "⚠ Revisar": "; ".join(alertas) if alertas else "✔",
        })
        shapes.append((ids.loc[i], pts))
    if not regs:
        return

    st.subheader("📐 Polígonos de parcela principal (control de calidad)")
    st.caption("Polígonos registrados con geoshape, identificados solo por ID. "
               "El área medida se calcula a partir de los vértices GPS; compárela "
               "con el área reportada por el productor. Uso interno del equipo.")
    st.dataframe(pd.DataFrame(regs), hide_index=True, width="stretch")

    import plotly.graph_objects as go
    usa_map = hasattr(go, "Scattermap")
    estilo = selector_estilo("estilo_poly")
    fig = go.Figure()
    for pid, pts in shapes:
        lats = [p[0] for p in pts] + [pts[0][0]]
        lons = [p[1] for p in pts] + [pts[0][1]]
        tr = dict(lat=lats, lon=lons, mode="lines+markers", fill="toself",
                  name=f"#{pid}", hovertext=f"ID #{pid}")
        fig.add_trace(go.Scattermap(**tr) if usa_map else go.Scattermapbox(**tr))
    todas_lat = [p[0] for _, pts in shapes for p in pts]
    todas_lon = [p[1] for _, pts in shapes for p in pts]
    centro = dict(lat=float(np.mean(todas_lat)), lon=float(np.mean(todas_lon)))
    if usa_map:
        fig.update_layout(map=dict(style=estilo, center=centro, zoom=13),
                          height=500, margin=dict(l=0, r=0, t=10, b=0))
    else:
        fig.update_layout(mapbox=dict(style=estilo, center=centro, zoom=13),
                          height=500, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    # Visor de polígono individual
    st.markdown("**🔎 Ver el polígono de una observación**")
    opciones = [pid for pid, _ in shapes]
    sel = st.selectbox("Seleccione la encuesta (ID)", opciones, key="poly_sel")
    pts_sel = dict(shapes)[sel]
    fila_sel = next(r for r in regs if r["ID"] == sel)
    st.dataframe(pd.DataFrame([fila_sel]), hide_index=True, width="stretch")
    lats = [p[0] for p in pts_sel] + [pts_sel[0][0]]
    lons = [p[1] for p in pts_sel] + [pts_sel[0][1]]
    tr = dict(lat=lats, lon=lons, mode="lines+markers", fill="toself",
              name=f"#{sel}", hovertext=f"ID #{sel}")
    fig2 = go.Figure(go.Scattermap(**tr) if usa_map else go.Scattermapbox(**tr))
    centro2 = dict(lat=float(np.mean([p[0] for p in pts_sel])),
                   lon=float(np.mean([p[1] for p in pts_sel])))
    if usa_map:
        fig2.update_layout(map=dict(style=estilo, center=centro2, zoom=17),
                           height=450, margin=dict(l=0, r=0, t=10, b=0),
                           showlegend=False)
    else:
        fig2.update_layout(mapbox=dict(style=estilo, center=centro2, zoom=17),
                           height=450, margin=dict(l=0, r=0, t=10, b=0),
                           showlegend=False)
    st.plotly_chart(fig2, width="stretch")
    st.caption("Un polígono válido debe cerrar sobre sí mismo y tener un "
               "área coherente con lo reportado.")


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
    idx_c0 = resolve(d, "_index", exact=True)
    kobo = (d[idx_c0].astype("Int64").astype(str) if idx_c0
            else (d.index + 1).astype(str))
    etiquetas = ids.where(ids.str.startswith("s/ID"), ids + " · k" + kobo)
    elegido = st.selectbox("Seleccione la encuesta por ID (PRODUCTOS-ID ENCUESTA)",
                           etiquetas, key=f"ficha_{modulo}")
    i = etiquetas.index[etiquetas == elegido][0]
    fila = quitar_sensibles(d).loc[i]

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
    seccion_mapa(d, esperado)
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
        seccion_poligonos(d, book)
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
