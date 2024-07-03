from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware

from io import BytesIO 
from PIL import Image 
from typing import Tuple

from keras.models import load_model,Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#Loading models
model=load_model('./models/my_model.keras')
imgfeatures=pickle.load(open('./models/Imagefeatures.pkl','rb'))
tokenizer=pickle.load(open('./models/tokenizer.pkl','rb'))
Vggmodel=VGG16()
Vggmodel=Model(inputs=Vggmodel.inputs,outputs=Vggmodel.layers[-2].output)


def ExtractImg(img):
   feature=[]
#    img=load_img(imgpath,target_size=(224,224))
   img=img_to_array(img)
   img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
   img=preprocess_input(img)

   feature.append(Vggmodel.predict(img,verbose=0))

   return feature

def idx_to_word(integer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def GenarateCaptions(img):
   maxlen=35
   inputtxt='startseq'
   for i in range(maxlen):
      seq=tokenizer.texts_to_sequences([inputtxt])[0]
      seq=pad_sequences([seq],maxlen=maxlen)
      
        
      predCaption=model.predict([img,seq],verbose=0)

      predCaption=np.argmax(predCaption)

      word=idx_to_word(predCaption)

      if word is None:
           break

      inputtxt+=' '+word

      if word == 'endseq':
          break
          
   return inputtxt
   
def CleanResult(caption):
      
    caption=[word for word in caption.split() if word.lower() not in ['startseq','endseq']]
    caption=' '.join(caption)
    
    c=[]
    for w in caption.split():
        if w not in c:
            c.append(w)
    caption=" ".join(c)
    return caption

def Result(img):
    inputFeature=ExtractImg(img)
    caption=GenarateCaptions(inputFeature)
    caption=CleanResult(caption)
    return caption

def read_file_as_image(data)-> Tuple[np.ndarray, Tuple[int, int]]: # A function to read the image file as a numpy array
    img = Image.open(BytesIO(data)).convert('RGB')
    img=img.resize((224,224),Image.Resampling.BICUBIC)
    return img

@app.post('/predict')
async def GenerateCaptions(file: UploadFile=File(...)):
   try:
        img=await file.read()
        img=read_file_as_image(img)
        caption=Result(img)

        print(caption)
        return {"filename":file.filename,'caption':caption}

   except Exception as err:
      raise err     
       

import uvicorn

if __name__ == "__main__": # If the script is run directly
    uvicorn.run(app, host="localhost", port=8002)