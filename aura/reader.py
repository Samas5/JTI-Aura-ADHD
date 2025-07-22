from pylsl import resolve_byprop, StreamInlet

def connect_to_aura(name='AURA', stream_type='EEG'):
    """
    Conecta con el stream LSL que emite Aura.
    Si no encuentra, imprime qué streams están disponibles.
    """
    try:
        print(f"🔍 Buscando stream con nombre '{name}' y tipo '{stream_type}'...")

        # Intentar encontrar el stream con el nombre específico.
        streams = resolve_byprop('name', name, timeout=5.0)

        if not streams:
            print(f"⚠️ No se encontró ningún stream llamado '{name}'. Buscando todos los streams disponibles...")
            
            # Buscar todos los streams de tipo 'EEG' y comprobar si se encuentra el correcto.
            all_streams = resolve_byprop('type', stream_type, timeout=5.0)

            if not all_streams:
                raise RuntimeError(f"❌ No se encontró ningún stream de tipo '{stream_type}'.")

            print(f"📡 Se detectaron {len(all_streams)} stream(s) de tipo '{stream_type}':")
            for i, s in enumerate(all_streams):
                print(f"Stream {i+1}:")
                print(f"  Name: {s.name()}")
                print(f"  Type: {s.type()}")
                print(f"  Source ID: {s.source_id()}")
                print(f"  Channel Count: {s.channel_count()}")
                print(f"  Nominal Sampling Rate: {s.nominal_srate()} Hz")
                print("-----")

            raise RuntimeError(f"🎯 El stream con el nombre '{name}' no está activo. Verifica el dispositivo.")

        # Si encontramos el stream por nombre específico, conectamos.
        inlet = StreamInlet(streams[0])
        print("✅ Conectado al stream de Aura.")
        return inlet

    except Exception as e:
        print(f"❌ Error al conectar: {e}")
        raise


def get_eeg_samples(inlet, duration_sec=2, fs=256):
    """
    Lee datos EEG desde Aura durante cierto tiempo (en segundos).
    Devuelve una lista de muestras, cada muestra es una lista por canal.
    """
    total_samples = int(duration_sec * fs)
    eeg_data = []

    for _ in range(total_samples):
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample:
            eeg_data.append(sample)

    return eeg_data  # lista de listas: [[ch1, ch2, ...], [ch1, ch2, ...], ...]
