from django.urls import reverse
import pandas as pd
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model, model_from_json
from keras import backend as K
from apps.Logica import modeloSNN
import pickle
from keras.models import model_from_json
from keras.models import load_model
import json   
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences

class modeloSNN():
    """Clase modelo Preprocesamiento y SNN"""
def cargarPipeline(self,nombreArchivo):
    with open(nombreArchivo+'.pickle', 'rb') as handle:
        pipeline = pickle.load(handle)
    return pipeline

def cargarNN(self,nombreArchivo):
    model = load_model(nombreArchivo+'.h5')
    print("Red Neuronal Cargada desde Archivo") 
    return model

def cargarTokenizer(self,nombreArchivo):
    with open(nombreArchivo+'.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer
def cargarModelo(self):
    #Se carga el Pipeline de Preprocesamiento
    nombreArchivoPreprocesador='Recursos/pipePreprocesadores'
    pipe=self.cargarPipeline(self,nombreArchivoPreprocesador)
    print('Pipeline de Preprocesamiento Cargado')
    cantidadPasos=len(pipe.steps)
    print("Cantidad de pasos: ",cantidadPasos)
    print(pipe.steps)
    #Se carga la Red Neuronal
    modeloOptimizado=self.cargarNN(self,'Recursos/modeloRedNeuronalOptimizada')
    #Se integra la Red Neuronal al final del Pipeline
    pipe.steps.append(['modelNN',modeloOptimizado])
    cantidadPasos=len(pipe.steps)
    print("Cantidad de pasos: ",cantidadPasos)
    print(pipe.steps)
    print('Red Neuronal integrada al Pipeline')
    return pipe
def cargarToken(self):
    #Se carga el Tokenizer
    nombreArchivoTokenizer='Recursos/tokenizerPreprocesadores'
    tokenizer=self.cargarTokenizer(self,nombreArchivoTokenizer)
    print('Tokenizer Cargado')
    return tokenizer

def tokenizar(self,text):
    Tokenizer=self.cargarToken(self)
    text= [text]
    x = Tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(x, maxlen=81, padding='pre')
    x = pad_sequences(padded_sequences)
    x = pd.DataFrame(x)
    nombresVariables= ['0' ,  '1' ,  '2' ,  '3' ,  '4' ,  '5' ,  '6' ,  '7' ,  '8' ,  '9' ,  '10' ,  '11' ,  '12' ,
                    '13' ,  '14' ,  '15' ,  '16' ,  '17' ,  '18' ,  '19' ,  '20' ,  '21' ,  '22' ,  '23' ,  '24' ,
                    '25' ,  '26' ,  '27' ,  '28' ,  '29' ,  '30' ,  '31' ,  '32' ,  '33' ,  '34' ,  '35' ,  '36' ,
                    '37' ,  '38' ,  '39' ,  '40' ,  '41' ,  '42' ,  '43' ,  '44' ,  '45' ,  '46' ,  '47' ,  '48' ,
                    '49' ,  '50' ,  '51' ,  '52' ,  '53' ,  '54' ,  '55' ,  '56' ,  '57' ,  '58' ,  '59' ,  '60' ,
                    '61' ,  '62' ,  '63' ,  '64' ,  '65' ,  '66' ,  '67' ,  '68' ,  '69' ,  '70' ,  '71' ,  '72' , 
                    '73' ,  '74' ,  '75' ,  '76' ,  '77' ,  '78' ,  '79' ,  '80']
    x.columns=(nombresVariables)
    x
    return x

def predecir2(self,texto = 'muy rica y sabrosa la comida'):
    pipe=self.cargarModelo(self)
    Xnew = tokenizar(self ,texto)
    pred = (pipe.predict(Xnew) > 0.5).astype("int32")
    pred = pred.flatten()[0]# de 2D a 1D
    print(pred)
    if(pred == 0):
        prediccion="gracias por su comentario"
    if(pred == 1):
        prediccion=" Lamentamos su experiencia"
    if(pred == 2):
        prediccion="Esperamos mejorar"
    return prediccion
    
  