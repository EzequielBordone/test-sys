import soundfile as sf
import sounddevice as sd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Importa pyplot de matplotlib

#Punto 1

def ruidoRosa_voss(t, fs):
    """
    Genera ruido rosa utilizando el algoritmo de Voss-McCartney(https://www.dsprelated.com/showabstract/3933.php).
    Parametros
    ----------
    t : float
        Valor temporal en segundos, este determina la duración del ruido generado.
    fs : int, opcional
        Frecuencia de muestreo en Hz de la señal. Por defecto, el valor es 44100 Hz.
    
    returns: NumPy array
        Datos de la señal generada.
    """
    
    nrows = int(t * fs)
    ncols = 16  # Puedes ajustar el número de columnas según tus necesidades
    
    array = np.full((nrows, ncols), np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # el numero total de cambios es nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)
    
    df = pd.DataFrame(array)
    filled = df.fillna(method='ffill', axis=0)
    total = filled.sum(axis=1)
    
    ## Centrado de el array en 0
    total = total - total.mean()
    
    ## Normalizado
    valor_max = max(abs(max(total)), abs(min(total)))
    total = total / valor_max
    
    # Generar el archivo de audio .wav
    sf.write('ruidoRosa.wav', total, fs)
    
    return total

#LLamar  la funcion
ruido_rosa = ruidoRosa_voss(t=20, fs=44100)


#Realizar una función para visualizar el dominio temporal de la señal.
def visualizar_dominio_temporal(senal, fs):
  
    # Calcula el eje de tiempo en segundos
    tiempo = np.arange(0, len(senal)) / fs

    # Crea una figura y un eje
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo, senal, color='b')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Dominio Temporal de la Señal de Audio Generada, Ruido Rosa')
    plt.grid(True)

    # Muestra la gráfica
    plt.show()

#LLamar  la funcion
graf_ruido_rosa = visualizar_dominio_temporal(ruido_rosa, fs=44100)


'''recordar de instalar audacity y hacer screenshot de la envolvente del ruido rosa'''


#reproducir 
#completar
sd.play(ruido_rosa)
sd.wait()





# Completar
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd

def generate_and_plot_sine_sweep(fs, T, w1, w2):
    # Cálculo de K y L
    R = np.log(w2 / w1)
    K = T * w1 / R
    L = T / R

    # Generación del sine sweep
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    x = np.sin(K * (np.exp(t / L) - 1))

    # Cálculo de la frecuencia instantánea w(t)
    w_t = K / L * np.exp(t / L)

    # Cálculo de la modulación m(t)
    m_t = w1 / (2 * np.pi * w_t)

    # Generación del filtro inverso k(t)
    k = m_t * x[::-1]

    # Normalizacion del filtro inverso
    k_n = k / np.max(k)

    # Guardar los audios como archivos WAV
    sf.write("sine_sweep.wav", x, fs)
    sf.write("filtro_inverso.wav", k_n, fs)

    # Graficar el resultado
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.title("Sine Sweep Logarítmico")
    plt.subplot(2, 1, 2)
    plt.plot(k_n)
    plt.title("Filtro Inverso")
    plt.tight_layout()
    plt.show()
    return x , k_n
#Llamar a la funcion
sine_sweep, filtro_inverso = generate_and_plot_sine_sweep(44100, 40, w1 = 2 * np.pi * 20 , w2 = 2 * np.pi * 20000 )


'''recordar de instalar audacity y hacer screenshot de la envolvente del ruido rosa'''


# sd.play(sine_sweep)
# sd.wait()
# sd.play(filtro_inverso)
# sd.wait()
