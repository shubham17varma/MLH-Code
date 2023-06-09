from flask import Flask, render_template, request
import os
import pickle
import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import load_model
import io
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import base64

import mysql.connector


model = load_model('effnet.h5')
UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
 

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'This is your secret key to utilize session in Flask'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=('POST','GET'))
def predict_placement():
    if request.method == 'POST':
        pm = str(request.form.get('pm'))
        dr = str(request.form.get('dr'))
        img = request.files['img']
        # cv2.imwrite(img,'image')

    # prediction
    imga = Image.open(img)
    cv2.imwrite("image.jpg",np.array(imga))
    opencvImage = cv2.cvtColor(np.array(imga), cv2.COLOR_RGB2BGR)
    imgb = cv2.resize(opencvImage,(150,150))
    imgc = imgb.reshape(1,150,150,3)
    p = model.predict(imgc)
    G = p[0][0]*100
    M = p[0][2]*100
    Pu = p[0][3]*100
    # print('Model Confidence: {:0.2f}%'.format(max(p[0])*100))
    p = np.argmax(p,axis=1)[0]
    # result = model.predict(np.array([cgpa, iq, profile_score]).reshape(1, 3))

    if p==0:
        p='Glioma Tumor'
    elif p==1:
        result = 'Result: \nThe model predicts that there is no tumor'
        res = 'The model predicts that there is no tumor'
        result = result.split('\n')

        user = 'root' # your username
        passwd = '' # your password
        host = 'localhost' # your host
        db = 'neuro_care' # database where your table is stored
        table = 'rad_doc' # table you want to save
        image_anu = r'C:\Users\Administrator\Desktop\trial\image.jpg'

        
        con = mysql.connector.connect(user=user, password=passwd, host=host, db=db)
        cursor = con.cursor()
        
        with open(image_anu, 'rb') as file:
            encoded_bytes = base64.b64encode(file.read())
            encoded_string = encoded_bytes.decode('utf-8')

        # remove b' and ' from start and end
        encoded_string = encoded_string[0:-1]


        query = "INSERT INTO  neuro_care.rad_doc (doc_id, patient_id, dor, image, ml_result) VALUES (%s,%s,CURDATE(),%s,%s);" 
        val = (dr, pm, encoded_bytes,res)
        cursor.execute(query,val)
        con.commit() 
        
        return render_template('index.html', result=result)
        
    elif p==2:
        p='Meningioma Tumor'
    else:
        p='Pituitary Tumor'
    
    if p!=1:
        result = """Result:
        The model predicts that there is tumor present.
        The model predicts that it is a {}.
        Glioma Tumor: {:0.2f}%.
        Meningioma Tumor: {:0.2f}%.
        Pituitary Tumor: {:0.2f}%.""".format(p,G,M,Pu)
        res = 'The model predicts that there is tumor present.\nThe model predicts that it is a {}.\nGlioma Tumor: {:0.2f}%.\nMeningioma Tumor: {:0.2f}%.\nPituitary Tumor: {:0.2f}%.'.format(p,G,M,Pu)
        import subprocess
        subprocess.run(["python", r"C:\Users\Administrator\Desktop\trial\yolov5-master\yolov5-master\detect.py"])
    
        result = result.split('\n')

        user = 'root' # your username
        passwd = '' # your password
        host = 'localhost' # your host
        db = 'neuro_care' # database where your table is stored
        table = 'rad_doc' # table you want to save
        image_anu = r'C:\Users\Administrator\Desktop\trial\image.jpg'
        img_sid = r'C:\Users\Administrator\Desktop\trial\yolov5-master\yolov5-master\runs\detect\output\imag.jpg'
        
        con = mysql.connector.connect(user=user, password=passwd, host=host, db=db)
        cursor = con.cursor()
        
        with open(image_anu, 'rb') as file:
            encoded_bytes = base64.b64encode(file.read())
            encoded_string = encoded_bytes.decode('utf-8')

        # remove b' and ' from start and end
        encoded_string = encoded_string[0:-1]

        with open(img_sid, 'rb') as file:
            encoded_bytes_1 = base64.b64encode(file.read())
            encoded_string = encoded_bytes.decode('utf-8')

        query = "INSERT INTO  neuro_care.rad_doc (doc_id, patient_id, dor, image, ml_result,img_result) VALUES (%s,%s,CURDATE(),%s,%s,%s);" 
        val = (dr, pm, encoded_bytes,res,encoded_bytes_1)
        cursor.execute(query,val)
        con.commit() 
        
        return render_template('index.html', result=result)

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
