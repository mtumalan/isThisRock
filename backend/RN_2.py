import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
"""
def cargar_datos(ruta_archivo, posicion_etiqueta=-1, delete_header=False):
    # Leer el archivo CSV, manejando encabezados y delimitadores
    df = pd.read_csv(
        ruta_archivo, 
        delimiter=',',  # Especifica el delimitador correcto
        header=0 if delete_header else None,  # Manejamos el encabezado si existe
        on_bad_lines='skip',  # Saltar las líneas con errores
        dtype=str  # Leer todo como cadenas para evitar problemas de tipo
    )
    
    # Separar las características (X) y la etiqueta (Y)
    X = df.drop(columns=df.columns[posicion_etiqueta])
#chroma_stft_mean,chroma_stft_var,rms_mean,rms_var,spectral_centroid_mean,spectral_centroid_var,spectral_bandwidth_mean,spectral_bandwidth_var,rolloff_mean,rolloff_var,zero_crossing_rate_mean,zero_crossing_rate_var,harmony_mean,harmony_var,perceptr_mean,perceptr_var,tempo,mfcc1_mean,mfcc1_var,mfcc2_mean,mfcc2_var,mfcc3_mean,mfcc3_var,mfcc4_mean,mfcc4_var,mfcc5_mean,mfcc5_var,mfcc6_mean,mfcc6_var,mfcc7_mean,mfcc7_var,mfcc8_mean,mfcc8_var,mfcc9_mean,mfcc9_var,mfcc10_mean,mfcc10_var,mfcc11_mean,mfcc11_var,mfcc12_mean,mfcc12_var,mfcc13_mean,mfcc13_var,mfcc14_mean,mfcc14_var,mfcc15_mean,mfcc15_var,mfcc16_mean,mfcc16_var,mfcc17_mean,mfcc17_var,mfcc18_mean,mfcc18_var,mfcc19_mean,mfcc19_var,mfcc20_mean,mfcc20_var,label
    #rolloff_var,zero_crossing_rate_mean,zero_crossing_rate_var,harmony_mean,harmony_var,perceptr_mean,perceptr_var,
    X = df.drop(columns=df.columns[0]) #filename
    X = df.drop(columns=df.columns[1]) #length

    for i in range(11,17):
        X = df.drop(columns=df.columns[i])

    Y = df[df.columns[posicion_etiqueta]].values
    num_features = X.shape[1]
    return X, Y, num_features
"""
def cargar_datos(ruta_archivo, posicion_etiqueta=-1, delete_header=False):
    if delete_header:
        df = pd.read_csv(ruta_archivo, header=0)  # La primera fila se usa como encabezado
    else:
        df = pd.read_csv(ruta_archivo, header=None)  # Sin encabezado
    
    # Eliminar columnas no deseadas (filename, length y columnas 11 a 16)
    columns_to_drop = [0, 1, posicion_etiqueta] + list(range(11, 18))
    X = df.drop(columns=df.columns[columns_to_drop])
    
    # Separar las características (X) y la etiqueta (Y)
    Y = df[df.columns[posicion_etiqueta]].values
    num_features = X.shape[1]
    
    return X, Y, num_features

def trainModel():
    # Cargar los datos
    trainData, trainLabels, num_features = cargar_datos('data/3sec_real.csv', -1, True)

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
    trainData = scaler.fit_transform(trainData)
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
    model.fit(X_train,y_train, epochs=15, batch_size=200, validation_split=0.2)
    print(trainData)
    print(trainLabels)
    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(X_test,y_test)
    print('Test accuracy:', test_acc)
    return model

def predictModel(model,X_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred