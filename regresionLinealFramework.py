#importamos librerías
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp


#cargamos set de datos
data = pd.read_csv("dataset.csv")

#creamos objeto para el modelo lineal
regression = linear_model.LinearRegression()


#creamos variables X y Y
X = data.horas.values.reshape((-1,1))
Y = data.ingreso

#separamos nuestros datos en datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#ajustamos modelo con datos de entrenamiento
modelo = regression.fit(X_train, y_train)

#obtenemos score con datos de entrenamiento
R_sq = modelo.score(X_train,y_train)

#obtenemos score con datos de prueba
R_sq2 = modelo.score(X_test,y_test)
print("Coeficiente de determinación con datos de entrenamiento: ", R_sq)
print("Coeficiente de determinación con datos de prueba: ", R_sq2)



#predecimos valores con datos de prueba
y_pred = modelo.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean squared error datos de prueba: ", mse)

print("\nAhora veremos los mismos datos después de haber regularizado el modelo \n")
ridgeR = linear_model.Ridge(alpha = 1)
ridgeR.fit(X_train, y_train)
y_predR = ridgeR.predict(X_test)

#obtenemos score con datos de entrenamiento
R_sqR = ridgeR.score(X_train,y_train)

#obtenemos score con datos de prueba
R_sq2R = ridgeR.score(X_test,y_test)
print("Coeficiente de determinación con datos de entrenamiento: ", R_sqR)
print("Coeficiente de determinación con datos de prueba: ", R_sq2R)
mseR = mean_squared_error(y_test, y_predR)
print("Mean squared error datos de prueba: ", mseR)

#graficamos predicciones del modelo comparados con valor real
plt.figure(figsize = (10,6))

#gráfica con valores reales de prueba
plt.scatter(X_test, y_test, color = 'green')

#gráfica con valores que el modelo predijo de prueba
plt.plot(X_test, y_pred, color = 'k' , lw = 3)
plt.xlabel('x' , size = 20)
plt.ylabel('y', size = 20)
plt.show()







