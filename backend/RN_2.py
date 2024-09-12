import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def cargar_datos(ruta_archivo, posicion_etiqueta=-1, delete_header=False):
    if delete_header:
        df = pd.read_csv(ruta_archivo, header=0)  # La primera fila se usa como encabezado
    else:
        df = pd.read_csv(ruta_archivo, header=None)  # Sin encabezado
    
    # Eliminar columnas no deseadas (filename, length y columnas 11 a 16)
    columns_to_drop = [0, 1, posicion_etiqueta] + list(range(10, 19))
    X = df.drop(columns=df.columns[columns_to_drop])
    
    # Separar las características (X) y la etiqueta (Y)
    Y = df[df.columns[posicion_etiqueta]].values
    num_features = X.shape[1]
    
    return X, Y, num_features

def trainModel():
    # Cargar los datoss
    trainData, trainLabels, num_features = cargar_datos('/app/data/3_sec_real.csv', -1, True)

    #trainData.to_csv('X_datos.csv', index=False)
    #Y_df = pd.DataFrame(trainLabels, columns=['Etiqueta'])
    #Y_df.to_csv('Y_datos.csv', index=False)

    print(num_features)
    # Codificar las etiquetas
    encoder = LabelEncoder()
    y = encoder.fit_transform(trainLabels)

    # Convertir etiquetas a numpy array
    y = np.array(y).astype('int')

    #normalizar:
    scaler = StandardScaler()
    #StandardScaler(), RobustScaler(), MinMaxScaler()
    trainScaler = scaler.fit(trainData)
    trainData = trainScaler.transform(trainData)
    #Guardar todos los datos en un dataframe y exportar a un archivo csv
    trainDF = pd.DataFrame(trainData)
    trainDF['Etiqueta'] = y
    trainDF.to_csv('/app/data/trainData.csv', index=False)
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(trainData, y, test_size=0.2, shuffle=True)

    # Definir el modelo de red neuronal
    model = Sequential()
    model.add(Dense(num_features, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(int(num_features/2), activation='relu'))
    model.add(Dense(len(np.unique(trainLabels)), activation='softmax'))  
    print('modelo')
    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.005),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train,y_train, epochs=20, batch_size=200, validation_split=0.2)
    print(trainData)
    print(trainLabels)
    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(X_test,y_test)
    print('Test accuracy:', test_acc)
    return model, trainScaler, encoder

def predictModel(model,X_test, scaler):
    print("Start predictModel")
    
    # Convertir los datos de prueba a un array de numpy 2D
    X_test = np.array(X_test).reshape(1, -1)

    print("Reshape X_test")
    print(X_test)

    # Normalizar los datos de prueba
    X_test = scaler.transform(X_test)

    print("Normalizar X_test")
    print(X_test)

    # Predecir las etiquetas
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred