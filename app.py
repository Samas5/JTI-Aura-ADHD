import os
import io
import base64
from flask import Flask, render_template, session, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from aura.reader import connect_to_aura, get_eeg_samples
from aura.processor import calculate_tbr

"""
app.py (versión completa)
=========================
• Lector CSV inteligente (UTF‑8, UTF‑16‑LE, Latin‑1) y eliminación de fila de unidades.
• Detección flexible de canal EEG (F3/F4… o primera columna numérica != tiempo).
• Endpoints de calibración, TBR en vivo, datos para la gráfica y página realtime.
"""

# -------------------------------------------------------------------
# Utilidades de lectura de CSV
# -------------------------------------------------------------------

def leer_csv_auto(file_storage, header_row: int = 0, units_row: int | None = 1) -> pd.DataFrame:
    """Lee un CSV Aura probando codificaciones habituales.

    - `header_row`: fila donde están los nombres de columna.
    - `units_row`: fila con las unidades (µV, etc.) que se eliminará si se indica.
    """
    for enc in ("utf-8", "utf-16-le", "latin-1"):
        try:
            file_storage.seek(0)  # volver al inicio del stream
            df = pd.read_csv(
                file_storage,
                encoding=enc,
                header=header_row,
                skiprows=[units_row] if units_row is not None else None,
            )
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("No se pudo detectar la codificación del CSV")


def detectar_canal_eeg(df: pd.DataFrame) -> str | None:
    """Devuelve el nombre de un canal EEG presente en el DataFrame.

    1. Busca en una lista estándar (F3, F4, Fp1, …).
    2. Si no hay coincidencia, toma la primera columna numérica que no se llame
       "Time" ni "Timestamp".
    """
    estandar = [
        "F3", "F4", "F5", "F6", "F7", "F8", "Fp1", "Fp2",
        "C3", "C4", "P3", "P4", "O1", "O2",
    ]
    df.columns = df.columns.str.strip().str.replace("'", "", regex=False)
    for c in estandar:
        if c in df.columns:
            return c
    # Fallback: primera columna numérica (excluyendo tiempo)
    for col in df.columns:
        if col.lower() in {"time", "timestamp"}:
            continue
        try:
            pd.to_numeric(df[col].dropna().head(10))
            return col
        except ValueError:
            continue
    return None

# -------------------------------------------------------------------
# Utilidades de señal
# -------------------------------------------------------------------

def calcular_theta_beta(signal: np.ndarray, fs: int) -> tuple[float, float]:
    """Devuelve potencia theta y beta en una ventana de señal."""
    from scipy.signal import welch

    freqs, psd = welch(signal, fs=fs, nperseg=fs * 2)
    theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
    beta_power = np.sum(psd[(freqs >= 12) & (freqs <= 30)])
    return theta_power, beta_power


def procesar_eeg_dataframe(df: pd.DataFrame, canal: str, fs: int = 256, ventana_seg: int = 2) -> pd.DataFrame:
    """Extrae theta y beta de un DataFrame Aura para un canal dado."""
    df.columns = df.columns.str.strip().str.replace("'", "", regex=False)

    if canal not in df.columns:
        raise ValueError(f"El canal '{canal}' no está presente en el archivo.")

    senal = df[canal].dropna().astype(float).values
    muestras = fs * ventana_seg
    theta_vals, beta_vals = [], []

    for i in range(0, len(senal), muestras):
        ventana = senal[i : i + muestras]
        if len(ventana) < muestras:
            break
        theta, beta = calcular_theta_beta(ventana, fs)
        theta_vals.append(theta)
        beta_vals.append(beta)

    return pd.DataFrame({"theta": theta_vals, "beta": beta_vals})

# -------------------------------------------------------------------
# Configuración Flask + BD
# -------------------------------------------------------------------

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.secret_key = "clave_super_secreta_123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)  # (no se usa activamente, pero queda listo)

# Conexión con el dispositivo EEG
inlet = connect_to_aura()

# -------------------------------------------------------------------
# Rutas Web
# -------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------- 1. Calibración personal --------------------------------

@app.route("/calib")
def calib_page():
    return render_template("calib.html")


@app.route("/calibrar_tbr", methods=["POST"])
def calibrar_tbr():
    """Recibe 2 CSV (relajación y actividad) y guarda tbr_min / tbr_max."""
    act_file = request.files.get("actividad")
    relax_file = request.files.get("relajacion")
    if not act_file or not relax_file:
        return jsonify({"error": "Faltan archivos"}), 400

    # Leer CSV con detección de codificación
    df_act_raw = leer_csv_auto(act_file)
    df_rel_raw = leer_csv_auto(relax_file)

    # Detectar canal
    canal_eeg = detectar_canal_eeg(df_act_raw)
    if canal_eeg is None:
        return jsonify({
            "error": "No se encontró un canal EEG en el CSV.",
            "cols": list(df_act_raw.columns)
        }), 400

    # Procesar DataFrames
    df_act = procesar_eeg_dataframe(df_act_raw, canal=canal_eeg)
    df_rel = procesar_eeg_dataframe(df_rel_raw, canal=canal_eeg)

    tbr_min = round((df_act.theta / df_act.beta).mean(), 2)
    tbr_max = round((df_rel.theta / df_rel.beta).mean(), 2)

    session.update({"tbr_min": tbr_min, "tbr_max": tbr_max, "canal_eeg": canal_eeg})

    return jsonify({"ok": True, "tbr_min": tbr_min, "tbr_max": tbr_max, "canal": canal_eeg})


# ---------- 2. TBR en vivo + umbrales ------------------------------

@app.route("/tbr_value")
def tbr_value():
    """Calcula TBR en vivo y envía también los umbrales calibrados."""
    try:
        tbr_live = calculate_tbr(get_eeg_samples(inlet, 2, 256))
        return jsonify({
            "tbr": tbr_live,
            "tbr_min": session.get("tbr_min", 1.5),
            "tbr_max": session.get("tbr_max", 4.5),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ---------- 3. Endpoints auxiliares (gráfica, datos) ---------------

@app.route("/eeg_plot")
def eeg_plot():
    """Muestra una imagen PNG con 5 s de señal EEG."""
    try:
        eeg = np.array(get_eeg_samples(inlet, duration_sec=5, fs=256))
        signal = eeg[:, 0]
        tiempo = np.linspace(0, 5, len(signal))

        plt.figure(figsize=(10, 4))
        plt.plot(tiempo, signal, label="EEG Signal")
        plt.title("EEG Signal Variations Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        return render_template("eeg_plot.html", plot_url=img_b64)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/eeg_data")
def eeg_data():
    """Devuelve 1 s de señal (para Chart.js realtime)."""
    try:
        eeg = np.array(get_eeg_samples(inlet, duration_sec=1, fs=256))
        signal = eeg[:, 0].tolist()
        tiempo = np.linspace(0, 1, len(signal)).tolist()
        return jsonify({"time": tiempo, "signal": signal})
    except Exception as e:
        return jsonify({"error": str(e)})


# ---------- 4. Página realtime -------------------------------------

@app.route("/realtime")
def realtime():
    return render_template("realtime.html")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
