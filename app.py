import os
import io
import base64
from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from matplotlib import pyplot as plt
import numpy as np
from aura.reader import connect_to_aura, get_eeg_samples
from aura.processor import calculate_tbr

# Configuración de la base de datos
basedir = os.path.abspath(os.path.dirname(__file__))
db = SQLAlchemy()

# Crear la app de Flask
app = Flask(__name__)
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

if __name__ == '__main__':
    app.run(debug=True)
