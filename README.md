# holtwinters

import math
import pandas as pd
import numpy as np
import statistics
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go


#Holt-Winters
#Para Holt-Winters utilizaremos la función ExponentialSmoothing del paquete statsmodels.
#Instalación

pip install statsmodels

#Carga del paquete

from statsmodels.tsa.api import ExponentialSmoothing


#Ejemplo de Uso
#Para este ejemplo utilizaremos la tabla de datos “AirPassengers”, la cual trata sobre totales mensuales de pasajeros en aerolíneas internacionales 1949–1960.

AirPassengers = pd.read_csv("../../../datos/AirPassengers.csv", delimiter = ',', decimal = ".", header = 0)
AirPassengers_ts = pd.Series(AirPassengers.Passengers.values, index = pd.DatetimeIndex(AirPassengers.TravelDate, freq = "MS"))

AirPassengers_train = AirPassengers_ts.head(len(AirPassengers_ts) - 12)
AirPassengers_test  = AirPassengers_ts.tail(12)

modelo = ExponentialSmoothing(AirPassengers_train, trend = 'add', seasonal = 'add')
modelo_fit = modelo.fit()
pred = modelo_fit.forecast(12)
pred



#Holt-Winters Calibrado
#Para encontrar los mejores parámetros en Holt-Winters, podemos utilizar la fuerza bruta. Para ello, podemos utilizar la siguiente función:

class HW_Prediccion(Prediccion):
  def __init__(self, modelo, alpha, beta, gamma):
    super().__init__(modelo)
    self.__alpha = alpha
    self.__beta  = beta
    self.__gamma = gamma
  
  @property
  def alpha(self):
    return self.__alpha
  
  @property
  def beta(self):
    return self.__beta 
  
  @property
  def gamma(self):
    return self.__gamma
  
  def forecast(self, steps = 1):
    res = self.modelo.forecast(steps)
    return(res)
  
class HW_calibrado(Modelo):
  def __init__(self, ts, test, trend = 'add', seasonal = 'add'):
    super().__init__(ts)
    self.__test = test
    self.__modelo = ExponentialSmoothing(ts, trend = trend, seasonal = seasonal)
  
  @property
  def test(self):
    return self.__test  
  
  @test.setter
  def test(self, test):
    if(isinstance(test, pd.core.series.Series)):
      if(test.index.freqstr != None):
        self.__test = test
      else:
        warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
    else:
      warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')
  
  def fit(self, paso = 0.1):
    error = float("inf")
    n = np.append(np.arange(0, 1, paso), 1)
    for alpha in n:
      for beta in n:
        for gamma in n:
          model_fit = self.__modelo.fit(smoothing_level = alpha, smoothing_trend = beta, smoothing_seasonal = gamma)
          pred      = model_fit.forecast(len(self.test))
          mse       = sum((pred - self.test)**2)
          if mse < error:
            res_alpha = alpha
            res_beta  = beta
            res_gamma = gamma
            error = mse
            res = model_fit
    return(HW_Prediccion(res, res_alpha, res_beta, res_gamma))

modelo_calibrado     = HW_calibrado(AirPassengers_train, AirPassengers_test)
modelo_calibrado_fit = modelo_calibrado.fit(0.05)

modelo_calibrado_fit.alpha

modelo_calibrado_fit.beta

modelo_calibrado_fit.gamma

pred_calibrado = modelo_calibrado_fit.forecast(12)
pred_calibrado

Gráfico de la predicción

fig = go.Figure()
no_plot = fig.add_trace(
  go.Scatter(x = AirPassengers_train.index.tolist(), y = AirPassengers_train.values.tolist(), 
             mode = 'lines+markers', name = "Entrenamiento")
)
no_plot = fig.add_trace(
  go.Scatter(x = AirPassengers_test.index.tolist(), y = AirPassengers_test.values.tolist(), 
             mode = 'lines+markers', name = "Prueba")
)
no_plot = fig.add_trace(
  go.Scatter(x = pred.index.tolist(), y = pred.values.tolist(), 
             mode = 'lines+markers', name = "Holt-Winters")
)
no_plot = fig.add_trace(
  go.Scatter(x = pred_calibrado.index.tolist(), y = pred_calibrado.values.tolist(), 
             mode = 'lines+markers', name = "Holt-Winters Calibrado")
)

no_plot = fig.update_xaxes(rangeslider_visible=True)
fig.show()

Medición del error

errores = ts_error([pred, pred_calibrado], AirPassengers_test, ["Holt-Winters", "Holt-Winters Calibrado"])
errores.df_errores()


errores.plot_errores()

Predicción final

modelo = ExponentialSmoothing(AirPassengers_ts, trend = 'add', seasonal = 'add')
modelo_fit = modelo.fit(smoothing_level = 0.25, smoothing_slope = 0.05, smoothing_seasonal = 0.75)
pred = modelo_fit.forecast(6)
pred

fig = go.Figure()
no_plot = fig.add_trace(
  go.Scatter(x = AirPassengers_ts.index.tolist(), y = AirPassengers_ts.values.tolist(), 
             mode = 'lines+markers', name = "Original")
)
no_plot = fig.add_trace(
  go.Scatter(x = pred.index.tolist(), y = pred.values.tolist(), 
             mode = 'lines+markers', name = "Predicción")
)

no_plot = fig.update_xaxes(rangeslider_visible=True)
fig.show()

#Deep Learning
#Para este caso definiremos la siguiente función la cual utiliza el modelo LSTM, el cual es un tipo de Red Neuronal Recurrente.

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

class LSTM_TSPrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
    self.__scaler = MinMaxScaler(feature_range = (0, 1))
    self.__X = self.__scaler.fit_transform(self.modelo.ts.to_frame())
  
  def __split_sequence(self, sequence, n_steps):
    X, y = [], []
    for i in range(n_steps, len(sequence)):
      X.append(self.__X[i-n_steps:i, 0])
      y.append(self.__X[i, 0])
    return np.array(X), np.array(y)
  
  def forecast(self, steps = 1):
    res = []
    p = self.modelo.p
    for i in range(steps):
      y_pred = [self.__X[-p:].tolist()]
      X, y = self.__split_sequence(self.__X, p)
      X = np.reshape(X, (X.shape[0], X.shape[1], 1))
      self.modelo.m.fit(X, y, epochs = 10, batch_size = 1, verbose = 0)
      pred = self.modelo.m.predict(y_pred)
      res.append(self.__scaler.inverse_transform(pred).tolist()[0][0])
      self.__X = np.append(self.__X, pred.tolist(), axis = 0)
    
    start  = self.modelo.ts.index[-1]
    freq   = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class LSTM_TS(Modelo):
  def __init__(self, ts, p = 1, lstm_units = 50, dense_units = 1, optimizer = 'rmsprop', loss = 'mse'):
    super().__init__(ts)
    self.__p = p
    self.__m = Sequential()
    self.__m.add(LSTM(units = lstm_units, input_shape = (p, 1)))
    self.__m.add(Dense(units = dense_units))
    self.__m.compile(optimizer = optimizer, loss = loss)
  
  @property
  def m(self):
    return self.__m
  
  @property
  def p(self):
    return self.__p
  
  def fit(self):
    res = LSTM_TSPrediccion(self)
    return(res)
#Para este ejemplo utilizaremos nuevamente la tabla de datos “AirPassengers”, la cual trata sobre totales mensuales de pasajeros en aerolíneas internacionales 1949–1960.

AirPassengers = pd.read_csv("../../../datos/AirPassengers.csv", delimiter = ',', decimal = ".", header = 0)
AirPassengers_ts = pd.Series(AirPassengers.Passengers.values, index = pd.DatetimeIndex(AirPassengers.TravelDate, freq = "MS"))


#Partimos en Aprendizaje y Prueba.

AirPassengers_train = AirPassengers_ts.head(len(AirPassengers_ts) - 12)
AirPassengers_test  = AirPassengers_ts.tail(12)


#Generamos el modelo.

modelo = LSTM_TS(AirPassengers_train, 12)
modelo_fit = modelo.fit()

#Generamos la predicción.

pred = modelo_fit.forecast(12)

pred

Gráfico de la predicción

fig = go.Figure()
no_plot = fig.add_trace(
  go.Scatter(x = AirPassengers_train.index.tolist(), y = AirPassengers_train.values.tolist(), 
             mode = 'lines+markers', name = "Entrenamiento")
)
no_plot = fig.add_trace(
  go.Scatter(x = AirPassengers_test.index.tolist(), y = AirPassengers_test.values.tolist(), 
             mode = 'lines+markers', name = "Prueba")
)
no_plot = fig.add_trace(
  go.Scatter(x = pred.index.tolist(), y = pred.values.tolist(), 
             mode = 'lines+markers', name = "Redes Neuronales (LSTM)")
)

no_plot = fig.update_xaxes(rangeslider_visible=True)
fig.show()

Predicción final

modelo = LSTM_TS(AirPassengers_ts, 12)
modelo_fit = modelo.fit()
pred = modelo_fit.forecast(6)

pred

fig = go.Figure()
no_plot = fig.add_trace(
  go.Scatter(x = AirPassengers_ts.index.tolist(), y = AirPassengers_ts.values.tolist(), 
             mode = 'lines+markers', name = "Original")
)
no_plot = fig.add_trace(
  go.Scatter(x = pred.index.tolist(), y = pred.values.tolist(), 
             mode = 'lines+markers', name = "Predicción")
)

no_plot = fig.update_xaxes(rangeslider_visible=True)
fig.show()

#Uso de Reglas
#Vamos a utilizar reglas cuando por criterio experto determinemos que un dato (por ejemplo una fecha) debe recibir un “trato especial”.
#Ejemplo: Regla para el día de la madre en Costa Rica, 15 de agosto 2012.
#Vamos a utlizar los datos de un cajero automático que inician el primero de enero de 1998 y termina el 30 de julio del 2012. Primero estudiamos la historia del retiro de dinero del cajero automático para el día de #la madre en Costa Rica (15 de agosto), esto para poder determinar una regla para este día en años futuros.
#Se sabe que el retiro de dinero en los cajeros varia drásticamente ese día en particular y por lo tanto es poco probable que el modelo pueda hacer una predicción precisa en ese día en particular, por lo que #queremos dar una regla.
#Para determinar esta regla, lo que hacemos es predecir el último 15 de agosto del cual si tenemos conocimiento y comparar la predicción con el dato real para encontrar un Factor de Ajuste (porcentaje de ajuste) que #nos ayude a mejorar la predicción del modelo


#Cargamos los datos

cajero = pd.read_csv("../../../datos/Cajero.csv", delimiter = ',', decimal = ".", header = 0)
fechas = pd.date_range(start = "1998-01-01", end = "2012-07-30", freq = 'D')
cajero_ts = pd.Series(cajero.monto.values, index = fechas)
cajero_ts


Freq: D, Length: 5325, dtype: float64

#Cálculo del Factor de Ajuste en la Regla

cajero_regla = cajero_ts[cajero_ts.index < "2011-08-15"]
cajero_regla.tail()


#Generamos el modelo.

modelo = ExponentialSmoothing(cajero_regla, trend = 'add', seasonal = 'add')
modelo_fit = modelo.fit()

#Predecimos una fecha, es decir, el 15 de Agosto.

pred_15 = modelo_fit.forecast(1)
pred_15

#Tomamos el valor real de la serie completa.

real_15 = cajero_ts[cajero_ts.index == "2011-08-15"]
real_15

#Observar que en este caso el error fue hacia abajo, es decir, la predicción dio por debajo del real.

error = pred_15.values[0] - real_15.values[0]
error

#Calculamos el Factor de Ajuste. Observar que debemos sumar 1 si el error es por debajo del real y restar 1 si el valor esta por encima del real.

if error < 0:
  factor_ajuste = 1 + (abs(error) / pred_15.values[0])
else:
  factor_ajuste = 1 - (abs(error) / pred_15.values[0])

factor_ajuste

#Verficamos que la regla sea correcta.

pred_15.values[0] * factor_ajuste


#Ya establecida la regla, procedemos a realizar el modelo final
#Generamos el modelo y la predicción, pero esta vez con toda la serie de tiempo.

modelo = ExponentialSmoothing(cajero_ts, trend = 'add', seasonal = 'add')
modelo_fit = modelo.fit()

pred = modelo_fit.forecast(30)
pred

#Aplicamos la regla solo al día 15 de agosto.

pred[pred.index == "2012-08-15"] = pred[pred.index == "2012-08-15"] * factor_ajuste
pred[pred.index == "2012-08-15"]


#Como un agregado a estas reglas es conveniente utilizar el máximo o el mínimo si se pasa la predicción de alguno de estos 2 rangos.

if pred[pred.index == "2012-08-15"].values > max(pred): 
  pred[pred.index == "2012-08-15"] = max(pred)

if pred[pred.index == "2012-08-15"].values < min(pred): 
  pred[pred.index == "2012-08-15"] = min(pred)


#Graficamos la serie de tiempo

fig = go.Figure()
no_plot = fig.add_trace(
  go.Scatter(x = cajero_ts.index.tolist(), y = cajero_ts.values.tolist(), 
             mode = 'lines+markers', name = "Original")
)
no_plot = fig.add_trace(
  go.Scatter(x = pred.index.tolist(), y = pred.values.tolist(), 
             mode = 'lines+markers', name = "Predicción")
)

no_plot = fig.add_annotation(x = "2012-08-15", y = pred[15], text = "15 de agosto")

no_plot = fig.update_xaxes(rangeslider_visible=True)
fig.show()


#ejemplos El archivo “beer.csv” tiene las ventas mensuales de la producción de cerveza en Australia desde enero de 1956 hasta agosto de 1995. 


beer = pd.read_csv("../../../datos/beer.csv", delimiter = ';', decimal = ",", header = 0)
beer

#Conversión a Serie de Tiempo

fechas = pd.date_range(start = "1956-01-01", end = "1995-08-01", freq = 'MS')
beer_ts = pd.Series(beer.beer.values, index = fechas)
beer_ts

 #Partimos la serie en train y test

beer_train = beer_ts.head((len(beer_ts) - 10))
beer_test  = beer_ts.tail(10)

import seaborn as sns

promedio, desviacion = stats.norm.fit(np.diff(beer_ts))

r = plt.hist(
  np.diff(beer_ts), bins = 25, alpha = 0.6, color = 'black',
  edgecolor = 'white', density = True)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, promedio, desviacion)
no_print = plt.plot(x, p, 'pink', linewidth = 3, label = 'Normalidad')
no_print = plt.ylabel('Densidad', size = 14)
no_print = plt.legend(framealpha=1, frameon=True);

plt.show()

res = stats.shapiro(np.diff(beer_ts))
res[1]

#Conclusión: El test nos da un valor de p <= 0.05 (p-value <= 0.05) por lo que existe evidencia estadística para rechazar H0. Es decir, podemos afirmar que la “serie” NO sigue una distribución normal. 

 #Descomposición de la serie

from statsmodels.tsa.seasonal import seasonal_decompose

r = seasonal_decompose(beer_ts, model = 'additive')
r.plot()
plt.show()

 #Periodos más importantes

#Calculamos el periodograma con la clase Periodograma.

p_ts = Periodograma(beer_train)

#Tomamos de las frecuencias las mejores posiciones. El valor 1 no se toma en cuenta pues es toda la serie.

p_ts.mejor_freq(best=3)

#Finalmente obtenemos los periodos más importantes.

p_ts.mejor_periodos(best=3)

#plot

p_ts.plot_periodograma(best=3)

fig = p_ts.plotly_periodograma(best=3)
fig.show()

# Holt-Winters

modelo = ExponentialSmoothing(beer_train, trend = 'add', seasonal = 'add')
modelo_fit = modelo.fit()
pred_hw = modelo_fit.forecast(10)
pred_hw


#Holt-Winters Calibrado

modelo_calibrado     = HW_calibrado(beer_train, beer_test)
modelo_calibrado_fit = modelo_calibrado.fit(0.1)
print("Holt-Winters (" + str(modelo_calibrado_fit.alpha) + ", " + 
      str(modelo_calibrado_fit.beta) + ", " + str(modelo_calibrado_fit.gamma) + ")")

pred_hw_calibrado = modelo_calibrado_fit.forecast(10)
pred_hw_calibrado

 #Redes Neuronales (LSTM)

modelo = LSTM_TS(beer_train, 12)
modelo_fit = modelo.fit()
pred_LSTM = modelo_fit.forecast(10)

pred_LSTM

# Graficamos las predicciones

series  = [beer_train, beer_test, pred_hw, pred_hw_calibrado, pred_LSTM]
nombres = ["Entrenamiento", "Prueba", "Holt-Winters", "Holt-Winters Calibrado", "Redes Neuronales (LSTM)"]

fig = go.Figure()
for i in range(len(series)):
  no_plot = fig.add_trace(
    go.Scatter(x = series[i].index.tolist(), y = series[i].values.tolist(),
               mode = 'lines+markers', name = nombres[i])
  )

no_plot = fig.update_xaxes(rangeslider_visible=True)
fig.show()

#mediciòn del errror

errores = ts_error([pred_hw, pred_hw_calibrado, pred_LSTM], beer_test, ["Holt-Winters", "Holt-Winters Calibrado", "Redes Neuronales (LSTM)"])
errores.df_errores()

fig = errores.plotly_errores()
fig.show()

metodo_hw = ExponentialSmoothing(beer_ts, trend = 'add', seasonal = 'add')
pred_hw   = metodo_hw.fit(smoothing_level = 0.8, smoothing_slope = 0, smoothing_seasonal = 0.4).forecast(6)
pred_hw
fig = go.Figure()
no_plot = fig.add_trace(
  go.Scatter(x = beer_ts.index.tolist(), y = beer_ts.values.tolist(), 
             mode = 'lines+markers', name = "Original")
)
no_plot = fig.add_trace(
  go.Scatter(x = pred_hw.index.tolist(), y = pred_hw.values.tolist(), 
             mode = 'lines+markers', name = "Predicción")
)

no_plot = fig.update_xaxes(rangeslider_visible=True)
fig.show()

#ejemplo 2 

#Para este utilizaremos el archivo “AMZN.csv” la cual contiene los movimientos diarios de las acciones de Amazon desde el 01 de Abril del 2016 hasta el 27 de julio del 2018. En la tabla se presenta la información #con la que abrió y cerró la acción en ese día, entre otra información como el mayor y menor valor alcanzado durante ese día. 


amazon = pd.read_csv("../../../datos/AMZN.csv", delimiter = ',', decimal = ".", header = 0)
amazon

 #Corrección de fechas

  #Buscamos cuales son las fechas faltantes.

fechas = pd.DatetimeIndex(amazon.index)
amazon["fechas"] = fechas
fecha_inicio = amazon.fechas.to_list()[0]
fecha_final  = amazon.fechas.to_list()[len(amazon.fechas.to_list()) - 1]
total_fechas = pd.date_range(start = fecha_inicio, end = fecha_final, freq = 'B')
faltan_fechas = [x for x in total_fechas if x not in amazon.fechas.to_list()]
faltan_fechas

amazon = pd.concat([amazon, pd.DataFrame({'fechas': faltan_fechas})], ignore_index = True)
amazon = amazon.sort_values(by = ['fechas'])
amazon.index = amazon['fechas']
amazon

#Realizaremos un suavizado.

amazon_suavizado = amazon.copy()
amazon_suavizado.Open = amazon_suavizado.Open.rolling(5, min_periods = 1, center = True).mean()
amazon_suavizado

#Imputamos el valor del suavizado a la serie original. Vamos a predecir la variable Open, por tanto solo utilizaremos dicha variable.

amazon.loc[faltan_fechas, 'Open'] = amazon_suavizado.loc[faltan_fechas, 'Open']

# Convertimos a serie de tiempo.

#Creamos la serie de tiempo con los datos NO suavizados.

fechas = pd.DatetimeIndex(amazon.index, freq = "B")
amazon_ts = pd.Series(amazon.Open.values, index = fechas)
amazon_ts

 #Partimos en Aprendizaje y Pruebas

amazon_train = amazon_ts.head((len(amazon_ts) - 20))
amazon_test  = amazon_ts.tail(20)

# Normalidad de la serie

import seaborn as sns

promedio, desviacion = stats.norm.fit(np.diff(amazon_ts))

r = plt.hist(
  np.diff(amazon_ts), bins = 25, alpha = 0.6, color = 'black', 
  edgecolor = 'white', density = True)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, promedio, desviacion)
no_print = plt.plot(x, p, 'pink', linewidth = 3, label = 'Normalidad')
no_print = plt.ylabel('Densidad', size = 14)
no_print = plt.legend(framealpha=1, frameon=True);

plt.show()


res = stats.shapiro(np.diff(amazon_ts))
res[1]

 #Descomposición de la serie

from statsmodels.tsa.seasonal import seasonal_decompose

r = seasonal_decompose(amazon_ts, model = 'additive')
r.plot()
plt.show()

 #Periodos más importantes

 #   Calculamos el periodograma con la clase Periodograma.

p_ts = Periodograma(amazon_train)


    #Tomamos de las frecuencias las mejores posiciones. El valor 1 no se toma en cuenta pues es toda la serie.

p_ts.mejor_freq(best=3)


    #Finalmente obtenemos los periodos más importantes.

p_ts.mejor_periodos(best=3)

p_ts.plot_periodograma(best=3)

fig = p_ts.plotly_periodograma(best=3)
fig.show()

modelo = ExponentialSmoothing(amazon_train, trend = 'add', seasonal = 'add')
modelo_fit = modelo.fit()
pred_hw = modelo_fit.forecast(20)
pred_hw

 #Holt-Winters Calibrado

modelo_calibrado     = HW_calibrado(amazon_train, amazon_test)
modelo_calibrado_fit = modelo_calibrado.fit(0.1)
print("Holt-Winters (" + str(modelo_calibrado_fit.alpha) + ", " + 
      str(modelo_calibrado_fit.beta) + ", " + str(modelo_calibrado_fit.gamma) + ")")

pred_hw_calibrado = modelo_calibrado_fit.forecast(20)
pred_hw_calibrado

 #Redes Neuronales (LSTM)

modelo = LSTM_TS(amazon_train, 67)
modelo_fit = modelo.fit()
pred_LSTM = modelo_fit.forecast(20)

pred_LSTM

# Graficamos las predicciones

series  = [amazon_train, amazon_test, pred_hw, pred_hw_calibrado, pred_LSTM]
nombres = ["Entrenamiento", "Prueba", "Holt-Winters", "Holt-Winters Calibrado", "Redes Neuronales (LSTM)"]

fig = go.Figure()
for i in range(len(series)):
  no_plot = fig.add_trace(
    go.Scatter(x = series[i].index.tolist(), y = series[i].values.tolist(),
               mode = 'lines+markers', name = nombres[i])
  )

no_plot = fig.update_xaxes(rangeslider_visible=True)
fig.show()



errores = ts_error([pred_hw, pred_hw_calibrado, pred_LSTM], amazon_test, ["Holt-Winters", "Holt-Winters Calibrado", "Redes Neuronales (LSTM)"])
errores.df_errores()

metodo_hw = ExponentialSmoothing(amazon_ts, trend = 'add', seasonal = 'add')
pred_hw   = metodo_hw.fit(smoothing_level = 0.4, smoothing_slope = 1, smoothing_seasonal = 0.1).forecast(6)
pred_hw

fig = go.Figure()
no_plot = fig.add_trace(
  go.Scatter(x = amazon_ts.index.tolist(), y = amazon_ts.values.tolist(), 
             mode = 'lines+markers', name = "Original")
)
no_plot = fig.add_trace(
  go.Scatter(x = pred_hw.index.tolist(), y = pred_hw.values.tolist(), 
             mode = 'lines+markers', name = "Predicción")
)

no_plot = fig.update_xaxes(rangeslider_visible=True)
fig.show()
fig = errores.plotly_errores()
fig.show()
