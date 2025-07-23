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
import time
from collections import deque
import json, pathlib
STATS_FILE = pathlib.Path("music_stats.json")
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


def load_stats():
    if STATS_FILE.exists():
        return json.loads(STATS_FILE.read_text())
    return []

music_stats = load_stats() 

def save_stats():
    STATS_FILE.write_text(json.dumps(music_stats, indent=2))

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

def color_tbr(v: float, tmin: float, tmax: float) -> str:
    if v > tmax:                       return "rojo"
    elif v > (tmin + tmax) / 2:        return "naranja"
    elif v >= tmin:                    return "verde"
    else:                              return "azul"

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


tbr_buf  = deque(maxlen=120)
time_buf = deque(maxlen=120)
# -------------------------------------------------------------------
# Rutas Web
# -------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/informacion")
def informacion():
    return render_template("informacion.html")

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


@app.route("/cerrar_medicion", methods=["POST"])
def cerrar_medicion():
    data = request.get_json()
    valores = data["datos"]
    if not valores:
        return jsonify({"error": "no_data"}), 400

    tmin, tmax = session["tbr_min"], session["tbr_max"]
    # clasificar colores
    colormap = {"rojo":0,"naranja":0,"verde":0,"azul":0}
    for v in valores:
        colormap[color_tbr(v, tmin, tmax)] += 1
    total = len(valores)

    # calcular estadísticos
    sesion = {
        "fecha":   time.strftime("%Y-%m-%d %H:%M"),
        "cancion": data["cancion"],
        "actividad": data["actividad"],
        "duracion": total / 2,                     # 0.5 s paso
        "promedio": round(sum(valores)/total, 2),
        "maximo":   round(max(valores), 2),
        "minimo":   round(min(valores), 2),
        "pct_rojo":   round(colormap["rojo"]/total*100, 1),
        "pct_naranja":round(colormap["naranja"]/total*100, 1),
        "pct_verde":  round(colormap["verde"]/total*100, 1),
        "pct_azul":   round(colormap["azul"]/total*100, 1)
    }
    music_stats.append(sesion)   # ① guarda en RAM
    save_stats()                 # ② persiste en JSON
    return jsonify({"ok": True})

@app.route("/resultados_musica")
def resultados_musica():
    return render_template("resultados_musica.html", sesiones=music_stats[::-1])

#Reiniciar stats de musica
@app.route("/reset_stats")
def reset_stats():
    music_stats.clear()
    save_stats()
    return "stats cleared"

# ---------- 2. TBR en vivo + umbrales ------------------------------

@app.route("/tbr_value")
def tbr_value():
    # ← NUEVO: verifica que existan los umbrales en sesión
    if "tbr_min" not in session or "tbr_max" not in session:
        return jsonify({"error": "not_calibrated"}), 400

    # --- lo que ya tenías ---
    try:
        tbr_live = calculate_tbr(get_eeg_samples(inlet, 2, 256))
        return jsonify({
            "tbr": tbr_live,
            "tbr_min": session["tbr_min"],   # usa los reales, ya sabemos que existen
            "tbr_max": session["tbr_max"],
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/tbr_stream")
def tbr_stream():
    if "tbr_min" not in session or "tbr_max" not in session:
        return jsonify({"error": "not_calibrated"}), 400

    """Devuelve arrays de tiempo (s) y TBR, más umbrales calibrados."""
    try:
        tbr_live = calculate_tbr(get_eeg_samples(inlet, 0.5, 256))  # 0.5 s
        ts = time.time()

        tbr_buf.append(tbr_live)
        time_buf.append(ts)

        if not time_buf:
            return jsonify({"error": "Sin datos aún"})

        t0 = time_buf[0]
        times = [round(t - t0, 1) for t in time_buf]  # relativo a t0

        return jsonify({
            "t": times,
            "v": list(tbr_buf),
            "tbr_min": session["tbr_min"],
            "tbr_max": session["tbr_max"],
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/tbr_stats")
def tbr_stats():
    if "tbr_min" not in session or "tbr_max" not in session:
        return jsonify({"error": "not_calibrated"}), 400
    if not tbr_buf:
        return jsonify({"error": "no_data"}), 400

    tmin, tmax = session["tbr_min"], session["tbr_max"]
    valores = list(tbr_buf)

    # clasificación por color
    colormap = {"rojo":0,"naranja":0,"verde":0,"azul":0}
    for v in valores:
        colormap[color_tbr(v, tmin, tmax)] += 1

    total = len(valores)
    porcentajes = {c: round(colormap[c] / total * 100, 1) for c in colormap}

    stats = {
        "promedio": round(sum(valores)/total, 2),
        "pico_max": round(max(valores), 2),
        "pico_min": round(min(valores), 2),
        "porcentajes": porcentajes
    }
    return jsonify(stats)

# ---------- 3. Endpoints auxiliares (gráfica, datos) ---------------
@app.route("/resultados")
def resultados():
    return render_template("resultados.html")

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
