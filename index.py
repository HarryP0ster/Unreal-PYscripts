import os
import numpy as np
import keras.models as mdl
import keras.utils as image_loader

def func(path):
    model = mdl.load_model('data.h5')

    dir_path = path.decode()
    print(dir_path)
    X = []
    
    for i in os.listdir(dir_path):
        img = image_loader.load_img(dir_path+'//'+i,target_size = (720,480))
        
        y = image_loader.img_to_array(img)
        y = np.expand_dims(y,axis = 0)
        X.append(y)

    
    images = np.vstack(X)
    val = model.predict(images)

    if val == 0:
        return 1
    else:
        return 2