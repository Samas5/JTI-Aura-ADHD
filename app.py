import os
import io
import base64
from flask import Flask, render_template,session, jsonify,request
from flask_sqlalchemy import SQLAlchemy
from matplotlib import pyplot as plt
import numpy as np
from aura.reader import connect_to_aura, get_eeg_samples
from aura.processor import calculate_tbr
import time
import pandas as pd


def calcular_theta_beta(signal, fs):
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
    beta_power = np.sum(psd[(freqs >= 12) & (freqs <= 30)])
    return theta_power, beta_power

def procesar_eeg_dataframe(df, canal='F5', fs=256, ventana_seg=2):
    # Limpiar nombres de columna
    df.columns = df.columns.str.strip().str.replace("'", "")

    if canal not in df.columns:
        raise ValueError(f"El canal '{canal}' no está presente en el archivo.")

    señal = df[canal].dropna().astype(float).values
    muestras_por_ventana = fs * ventana_seg

    theta_vals = []
    beta_vals = []

    for i in range(0, len(señal), muestras_por_ventana):
        ventana = señal[i:i + muestras_por_ventana]
        if len(ventana) < muestras_por_ventana:
            break
        theta, beta = calcular_theta_beta(ventana, fs)
        theta_vals.append(theta)
        beta_vals.append(beta)

    df_out = pd.DataFrame({'theta': theta_vals, 'beta': beta_vals})
    return df_out



# Configuración de la base de datos
basedir = os.path.abspath(os.path.dirname(__file__))
db = SQLAlchemy()

# Crear la app de Flask
app = Flask(__name__)
app.secret_key = 'clave_super_secreta_123'  # <-- AGREGA ESTA LÍNEA
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Conectar con el dispositivo EEG
inlet = connect_to_aura()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/tbr")
def tbr():
    try:
        # Obtener datos EEG
        eeg_data = get_eeg_samples(inlet, duration_sec=2, fs=256)
        
        # Calcular la relación Theta/Beta
        tbr_value = calculate_tbr(eeg_data, fs=256, channel_index=0)
        
        return jsonify({"TBR": tbr_value})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/eeg_plot")
def eeg_plot():
    try:
        # Obtener datos EEG (usamos 5 segundos para visualización)
        eeg_data = get_eeg_samples(inlet, duration_sec=5, fs=256)
        
        # Convertir los datos a un array de numpy (para facilitar la manipulación)
        eeg_data = np.array(eeg_data)
        
        # Seleccionar un canal específico para graficar (por ejemplo, el canal 0)
        signal = eeg_data[:, 0]  # Asumiendo que canal 0 es el que te interesa
        
        # Crear un tiempo para cada punto de datos
        time = np.linspace(0, 5, len(signal))  # 5 segundos de duración, ajusta según sea necesario
        
        # Crear la figura para la gráfica
        plt.figure(figsize=(10, 4))
        plt.plot(time, signal, label="EEG Signal")
        plt.title("EEG Signal Variations Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("EEG Signal Amplitude")
        plt.grid(True)
        plt.legend()
        
        # Guardar la gráfica en un buffer de memoria
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        
        # Codificar la imagen en base64 para mostrarla en el navegador
        img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        img_buf.close()
        
        # Pasar la imagen a la plantilla
        return render_template('eeg_plot.html', plot_url=img_base64)
        
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/eeg_data")
def eeg_data():
    try:
        eeg = get_eeg_samples(inlet, duration_sec=1, fs=256)  # 1 segundo
        eeg = np.array(eeg)
        signal = eeg[:, 0]
        time = np.linspace(0, 1, len(signal)).tolist()
        signal = signal.tolist()
        return jsonify({"time": time, "signal": signal})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/realtime")
def realtime():
    return render_template("realtime.html")

@app.route("/popup")
def popup():
    return render_template("realtime_popup.html")


    
@app.route('/calib')
def calibrar():
    return render_template('calib.html')

@app.route('/calibrar_tbr', methods=['POST'])
def calibrar_tbr():
    try:
        actividad_file = request.files['actividad']
        relajacion_file = request.files['relajacion']

        if not actividad_file or not relajacion_file:
            return jsonify({"error": "Faltan archivos"}), 400

        df_actividad = pd.read_csv(actividad_file, skiprows=2)
        df_relajacion = pd.read_csv(relajacion_file, skiprows=2)

        # 🔧 LIMPIAR LOS NOMBRES DE COLUMNA AQUÍ TAMBIÉN
        df_actividad.columns = df_actividad.columns.str.strip().str.replace("'", "")
        df_relajacion.columns = df_relajacion.columns.str.strip().str.replace("'", "")

        canal = 'F3'

        df_act = procesar_eeg_dataframe(df_actividad, canal)
        df_rel = procesar_eeg_dataframe(df_relajacion, canal)

        # Calcular TBR (theta/beta) promedio
        tbrs_actividad = (df_act["theta"] / df_act["beta"]).dropna()
        tbrs_relajacion = (df_rel["theta"] / df_rel["beta"]).dropna()

        if len(tbrs_actividad) == 0 or len(tbrs_relajacion) == 0:
            return jsonify({"error": "No se pudieron calcular TBRs"}), 400

        tbr_max = round(tbrs_relajacion.mean(), 2)  # relajación → más theta
        tbr_min = round(tbrs_actividad.mean(), 2)   # concentración → menos theta

        # Guardar en sesión
        session['tbr_min'] = tbr_min
        session['tbr_max'] = tbr_max

        return jsonify({
            "status": "ok",
            "tbr_min": tbr_min,
            "tbr_max": tbr_max
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
