from PIL import Image
import pickle
import joblib
import numpy as np


def make_prediction(model , img_path):
    # Load and preprocess the image
    img = Image.open(img_path).convert('L')  
    img = img.resize((28, 28))
    img_array = np.array(img)
    
    img_array = img_array / 255.0              
    img_flat = img_array.flatten()             
    img_flat = img_flat.reshape(1, -1)


    if model == 1:
        file_path = 'bagging_model.pkl'
    elif model == 2:
        file_path = 'boosting_model.pkl'
    else:
        file_path = 'stacking_model.pkl'


    model = load_model(file_path)

    pred = model.predict(img_flat)


    return int(pred[0])  



def load_model(path):

    loaded_model = joblib.load(path)

    return loaded_model