import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def cargar_datos(ruta_archivo, posicion_etiqueta=-1, delete_header=False):
    if delete_header:
        df = pd.read_csv(ruta_archivo, header=0)  # La primera fila se usa como encabezado
    else:
        df = pd.read_csv(ruta_archivo, header=None)  # Sin encabezado
    
    # Separar las características (X) y la etiqueta (Y)
    X = df.drop(columns=df.columns[posicion_etiqueta])
    Y = df[df.columns[posicion_etiqueta]].values
    
    return X, Y

def preprocesar_datos(X):
    # Convertir columnas categóricas a números usando codificación One-Hot
    # Identificar columnas no numéricas
    non_numeric_columns = X.select_dtypes(include=['object']).columns
    
    # Aplicar codificación one-hot
    X_encoded = pd.get_dummies(X, columns=non_numeric_columns)
    
    # Escalar características numéricas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    return X_scaled

# Cargar los datos
trainData, trainLabels = cargar_datos('data/music_data.csv', 3, True)

# Preprocesar los datos
trainData = preprocesar_datos(trainData)

# Convertir etiquetas a enteros
encoder = LabelEncoder()
trainLabels_encoded = encoder.fit_transform(trainLabels)

# Convertir etiquetas a numpy array
trainLabels = np.array(trainLabels_encoded).astype('int')


trainData, testVectors, trainLabels, testLabels = train_test_split(
trainData, trainLabels, test_size=0.2, shuffle=True)

# Definir el modelo de red neuronal
model = Sequential()
model.add(Dense(18, activation='relu', input_shape=(trainData.shape[1],)))
model.add(Dense(9, activation='relu'))
model.add(Dense(len(np.unique(trainLabels)), activation='softmax'))  

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(trainData, trainLabels, epochs=10, batch_size=200, validation_split=0.2)
print(trainData)
print(trainLabels)
# Evaluar el modelo
test_loss, test_acc = model.evaluate(testVectors, testLabels)
print('Test accuracy:', test_acc)

"""
# Realizar las predicciones
nuevo_dato = np.expand_dims(trainData[1], axis=0)
predicciones = model.predict(nuevo_dato)

# Las predicciones son probabilidades para cada clase
# Si deseas obtener las etiquetas de las clases, puedes usar np.argmax para obtener la clase con la mayor probabilidad
etiquetas_predichas = np.argmax(predicciones, axis=1)

print("Predicciones (probabilidades):", predicciones)
print("Etiquetas predichas:", etiquetas_predichas)
"""