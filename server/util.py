import joblib
import cv2
import json
import base64
from wavelet import w2d
import numpy as np
# from  matplotlib import pyplot as plt

__class_name_to_number = {}
__class_number_to_name = {}
__model  = None


# Now in order to make it feasible for the UI to tranfer the image to the backend for the 
# Backend to run the model on it 
# We first need to convert that image into b64 encoding and then reconvert it into a cv2 
# Accesible image for it to be processed upon
# Most of this code is from the model building while data cleaning and training phase


def load_saved_artifacts():
    print("Loading saved artifacts")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k, v in   __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)

    print("loading saved Artifcats Completed...")
            

def get_b64_image():
    with open("b64.txt") as f:
        return f.read()


def class_number_to_name(class_num):
    Result = __class_number_to_name[class_num].split('_')
    Result = str(Result[0]).capitalize() + ' ' + str(Result[1]).capitalize()
    return Result




def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img




def get_cropped_image(image_base64_data, image_path):
    face_cascade = cv2.CascadeClassifier('E:\Python\Image Classification\A\model\opencv\haarcascades\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('E:\Python\Image Classification\A\model\opencv\haarcascades\haarcascade_eye.xml')
    
    if image_path:
                
        img = cv2.imread(image_path)

    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []

    for (x, y, w, h) in faces:
     
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
         
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    
    return cropped_faces
        


def classify_image(image_base64_data, filepath = None):
    imgs = get_cropped_image(image_base64_data, filepath)
    result = []
    for img in imgs:
        scalled_raw_image = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_image = np.vstack((scalled_raw_image.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
        len_image_array = 32*32*3 + 32*32
        final = combined_image.reshape(1, len_image_array).astype(float)

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            
            'class_probability': np.round(__model.predict_proba(final)*100,2).tolist()[0],
          
            'class_dictionary': __class_name_to_number
            })

        

        #We gave the model an image i.e. x data and it will predict the y data 
        # Which it correctly does
    return result
















if __name__ == "__main__":
    load_saved_artifacts()
    
    print("\n\n")


    print(classify_image(get_b64_image(), None)) 



    
    # print("\n")
    # print(classify_image(None, './test_images/dravid1.jpg')) 
    # print("\n")
    # print(classify_image(None, './test_images/messi1.png'))  
    # print("\n")
    # print(classify_image(None, './test_images/seinfeld1.jpg')) 
    # print("\n")
    # print(classify_image(None, './test_images/seinfeld2.jpg')) 

    # print(classify_image(None, './test_images/seinfeld3.webp')) 

    
    

