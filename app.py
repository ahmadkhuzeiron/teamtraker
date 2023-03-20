from flask import Flask, render_template, json, request, session
import pandas as pd
import logging
import json
import csv
import os
from csv import writer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route('/')
def main():
    return render_template('login.html')

@app.route('/home',methods=['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/daftar',methods=['POST','GET'])
def daftar():
    try:
        username = request.form['username']
        password = request.form['password']
        repassword = request.form['repassword']
        nama = request.form['nama']
        
        sukses = "Daftar Gagal"
        
        if password != repassword:
            return render_template('login.html',info="Password dan Repassword Tidak Sama")
            
        List=[nama,username,password,'User']
        
        with open('./data/users.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(List)
    
        return render_template('login.html',info="Pendaftaran Berhasil...")
        
    except Exception as e:
        return render_template('login.html',info="Silahkan Login atau Daftar Terlebih Dahulu")

@app.route('/beranda',methods=['POST','GET'])
def beranda():
    try:
        username = request.form['username']
        password = request.form['password']
        
        ditemukan = "Login Gagal"
        
        if username == "":
            return render_template('login.html',info="Username Tidak Boleh Kosong")
        elif password == "":
            return render_template('login.html',info="Username Tidak Boleh Kosong")
        
        with open('./data/users.csv', 'rt') as f:
         reader = csv.reader(f, delimiter=',')
         for row in reader:
              for field in row:
                  if username in field.lower():
                        ditemukan = "Login Gagal"
                  if password in field.lower():
                        ditemukan = "Login berhasil"
        if ditemukan == "Login berhasil":
            return render_template('index.html')
        else:
            return render_template('login.html',info="Username atau Password Salah")
            
    except Exception as e:
        return render_template('login.html',info="Silahkan Login Terlebih Dahulu")
    
@app.route('/totalData',methods=['POST','GET'])
def totalData():
    total = len(list(csv.reader(open("./data/users.csv","r+"))))
    totaldata = len(list(csv.reader(open("./static/upload/Accelerometer.csv","r+"))))+len(list(csv.reader(open("./static/upload/Gyroscope.csv","r+"))))+len(list(csv.reader(open("./static/upload/Magnetometer.csv","r+"))))
    
    return str(total-1)+" "+str(totaldata)

@app.route('/getUsers',methods=['POST','GET'])
def getUsers():
    data = pd.read_csv('./data/users.csv',nrows=100, skiprows=0)
    df = pd.DataFrame(data, columns= ['nama','username','password','jenis'])
    result = df.to_json(orient="split")
    parsed = json.loads(result)
    dataJson = json.dumps(parsed, indent=4)
    
    #with open('./static/data/acce.json', "w") as outfile:
    #    outfile.write(dataJson)
    
    return render_template('acce.html', dataCsv=dataJson)

@app.route('/getAcce',methods=['POST','GET'])
def getAcce():
    data = pd.read_csv('./static/upload/Accelerometer.csv',nrows=100, skiprows=0)
    df = pd.DataFrame(data, columns= ['timestamp (+0700)','elapsed (s)','subid-gradeid','actid'])
    result = df.to_json(orient="split")
    parsed = json.loads(result)
    dataJson = json.dumps(parsed, indent=4)
    
    #with open('./static/data/acce.json', "w") as outfile:
    #    outfile.write(dataJson)
    
    return render_template('acce.html', dataCsv=dataJson)

@app.route('/getGyro',methods=['POST','GET'])
def getGyro():
    data = pd.read_csv('./static/upload/Gyroscope.csv',nrows=1000, skiprows=0)
    df = pd.DataFrame(data, columns= ['timestamp (+0700)','elapsed (s)','subid-gradeid','actid'])
    result = df.to_json(orient="split")
    parsed = json.loads(result)
    dataJson = json.dumps(parsed, indent=4)
    
    #with open('./static/data/acce.json', "w") as outfile:
    #    outfile.write(dataJson)
    
    return render_template('acce.html', dataCsv=dataJson)

@app.route('/getMagne',methods=['POST','GET'])
def getMagne():
    data = pd.read_csv('./static/upload/Magnetometer.csv',nrows=100, skiprows=0)
    df = pd.DataFrame(data, columns= ['timestamp (+0700)','elapsed (s)','subid-gradeid','actid'])
    result = df.to_json(orient="split")
    parsed = json.loads(result)
    dataJson = json.dumps(parsed, indent=4)
    
    #with open('./static/data/acce.json', "w") as outfile:
    #    outfile.write(dataJson)
    
    return render_template('acce.html', dataCsv=dataJson)

@app.route('/uploadFile',methods=['POST','GET'])
def uploadFile():
    uploaded_df = request.files['file']
    data_filename = secure_filename(uploaded_df.filename)
    uploaded_df.save(os.path.join("./static/upload/", data_filename))
 
    return "Berhasil Diupload"

#Fungsi untuk Mencari getPrediksi
@app.route('/getPrediksi',methods=['POST','GET'])
def getPrediksi():

    tahap = request.form['tahap']
    
    data_raw_acc  = pd.read_csv("./static/upload/Accelerometer.csv")
    data_raw_gyro = pd.read_csv("./static/upload/Gyroscope.csv")
    data_raw_mag  = pd.read_csv("./static/upload/Magnetometer.csv")
    gambar = 'static/grafik/tahap/t'+tahap+'.png'
        
    if int(tahap) == 1:
        data_acc = data_raw_acc[["x-axis (g)","y-axis (g)","z-axis (g)","actid"]]
        plt.figure(figsize=(15,8))
        sns.lineplot(data=data_raw_acc, x='elapsed (s)', y='x-axis (g)')
        sns.lineplot(data=data_raw_acc, x='elapsed (s)', y='y-axis (g)')
        sns.lineplot(data=data_raw_acc, x='elapsed (s)', y='z-axis (g)')
        plt.savefig(gambar)
        
        result = data_acc.to_json(orient="split")
        parsed = json.loads(result)
        dataJson = json.dumps(parsed, indent=4)
        hasil = render_template('acce.html', dataCsv=dataJson)
        
    elif int(tahap) == 2:
        data_gyro = data_raw_gyro[["x-axis (deg/s)","y-axis (deg/s)","z-axis (deg/s)","actid"]]
        plt.figure(figsize=(15,8))
        sns.lineplot(data=data_raw_gyro, x='elapsed (s)', y='x-axis (deg/s)')
        sns.lineplot(data=data_raw_gyro, x='elapsed (s)', y='y-axis (deg/s)')
        sns.lineplot(data=data_raw_gyro, x='elapsed (s)', y='z-axis (deg/s)')
        plt.savefig(gambar)
        
        result = data_gyro.to_json(orient="split")
        parsed = json.loads(result)
        dataJson = json.dumps(parsed, indent=4)
        hasil = render_template('acce.html', dataCsv=dataJson)
        
    elif int(tahap) == 3:
        data_mag = data_raw_mag[["x-axis (T)","y-axis (T)","z-axis (T)","actid"]]
        plt.figure(figsize=(15,8))
        sns.lineplot(data=data_raw_mag, x='elapsed (s)', y='x-axis (T)')
        sns.lineplot(data=data_raw_mag, x='elapsed (s)', y='y-axis (T)')
        sns.lineplot(data=data_raw_mag, x='elapsed (s)', y='z-axis (T)')
        plt.savefig(gambar)
        
        result = data_mag.to_json(orient="split")
        parsed = json.loads(result)
        dataJson = json.dumps(parsed, indent=4)
        hasil = render_template('acce.html', dataCsv=dataJson)
       
    elif int(tahap) == 4:   #Traning Data Accelerometer dari 0 - 300 Masing2 Jenis Tahapan (Preparation (300 Data), Welding (300 Data), dll) = 1.200 Data
        data_acc = data_raw_acc[["x-axis (g)","y-axis (g)","z-axis (g)","actid"]]
        data_gyro = data_raw_gyro[["x-axis (deg/s)","y-axis (deg/s)","z-axis (deg/s)","actid"]]
        data_mag = data_raw_mag[["x-axis (T)","y-axis (T)","z-axis (T)","actid"]]
        
        # Split by Action
        data_acc_A  = data_acc[data_acc['actid'] == "Preparation"]
        data_acc_B  = data_acc[data_acc['actid'] == "Grinding"]
        data_acc_C  = data_acc[data_acc['actid'] == "Welding"]
        data_acc_D  = data_acc[data_acc['actid'] == "Slag Cleaning"]

        data_gyro_A = data_gyro[data_gyro['actid'] == "Preparation"]
        data_gyro_B = data_gyro[data_gyro['actid'] == "Grinding"]
        data_gyro_C = data_gyro[data_gyro['actid'] == "Welding"]
        data_gyro_D = data_gyro[data_gyro['actid'] == "Slag Cleaning"]

        data_mag_A  = data_mag[data_mag['actid'] == "Preparation"]
        data_mag_B  = data_mag[data_mag['actid'] == "Grinding"]
        data_mag_C  = data_mag[data_mag['actid'] == "Welding"]
        data_mag_D  = data_mag[data_mag['actid'] == "Slag Cleaning"]
        
        # Select 10000 data for each Action
        data_acc_A  = data_acc_A[0:10000]
        data_acc_B  = data_acc_B[0:10000]
        data_acc_C  = data_acc_C[0:10000]
        data_acc_D  = data_acc_D[0:10000]

        data_gyro_A = data_gyro_A[0:10000]
        data_gyro_B = data_gyro_B[0:10000]
        data_gyro_C = data_gyro_C[0:10000]
        data_gyro_D = data_gyro_D[0:10000]

        data_mag_A  = data_mag_A[0:10000]
        data_mag_B  = data_mag_B[0:10000]
        data_mag_C  = data_mag_C[0:10000]
        data_mag_D  = data_mag_D[0:10000]
        
        # acc

        data_acc_A_1 = data_acc_A[0:100]
        data_acc_B_1 = data_acc_B[0:100]
        data_acc_C_1 = data_acc_C[0:100]
        data_acc_D_1 = data_acc_D[0:100]

        data_acc_A_2 = data_acc_A[100:200]
        data_acc_B_2 = data_acc_B[100:200]
        data_acc_C_2 = data_acc_C[100:200]
        data_acc_D_2 = data_acc_D[100:200]

        data_acc_A_3 = data_acc_A[200:300]
        data_acc_B_3 = data_acc_B[200:300]
        data_acc_C_3 = data_acc_C[200:300]
        data_acc_D_3 = data_acc_D[200:300]

        data_acc_A_4 = data_acc_A[300:400]
        data_acc_B_4 = data_acc_B[300:400]
        data_acc_C_4 = data_acc_C[300:400]
        data_acc_D_4 = data_acc_D[300:400]

        data_acc_A_5 = data_acc_A[400:500]
        data_acc_B_5 = data_acc_B[400:500]
        data_acc_C_5 = data_acc_C[400:500]
        data_acc_D_5 = data_acc_D[400:500]

        # gyro

        data_gyro_A_1 = data_gyro_A[0:100]
        data_gyro_B_1 = data_gyro_B[0:100]
        data_gyro_C_1 = data_gyro_C[0:100]
        data_gyro_D_1 = data_gyro_D[0:100]

        data_gyro_A_2 = data_gyro_A[100:200]
        data_gyro_B_2 = data_gyro_B[100:200]
        data_gyro_C_2 = data_gyro_C[100:200]
        data_gyro_D_2 = data_gyro_D[100:200]

        data_gyro_A_3 = data_gyro_A[200:300]
        data_gyro_B_3 = data_gyro_B[200:300]
        data_gyro_C_3 = data_gyro_C[200:300]
        data_gyro_D_3 = data_gyro_D[200:300]

        data_gyro_A_4 = data_gyro_A[300:400]
        data_gyro_B_4 = data_gyro_B[300:400]
        data_gyro_C_4 = data_gyro_C[300:400]
        data_gyro_D_4 = data_gyro_D[300:400]

        data_gyro_A_5 = data_gyro_A[400:500]
        data_gyro_B_5 = data_gyro_B[400:500]
        data_gyro_C_5 = data_gyro_C[400:500]
        data_gyro_D_5 = data_gyro_D[400:500]

        # mag

        data_mag_A_1 = data_mag_A[0:100]
        data_mag_B_1 = data_mag_B[0:100]
        data_mag_C_1 = data_mag_C[0:100]
        data_mag_D_1 = data_mag_D[0:100]

        data_mag_A_2 = data_mag_A[100:200]
        data_mag_B_2 = data_mag_B[100:200]
        data_mag_C_2 = data_mag_C[100:200]
        data_mag_D_2 = data_mag_D[100:200]

        data_mag_A_3 = data_mag_A[200:300]
        data_mag_B_3 = data_mag_B[200:300]
        data_mag_C_3 = data_mag_C[200:300]
        data_mag_D_3 = data_mag_D[200:300]

        data_mag_A_4 = data_mag_A[300:400]
        data_mag_B_4 = data_mag_B[300:400]
        data_mag_C_4 = data_mag_C[300:400]
        data_mag_D_4 = data_mag_D[300:400]

        data_mag_A_5 = data_mag_A[400:500]
        data_mag_B_5 = data_mag_B[400:500]
        data_mag_C_5 = data_mag_C[400:500]
        data_mag_D_5 = data_mag_D[400:500]
        
        DataTrainingACC = data_acc_A[0:300].append(data_acc_B[0:300]).append(data_acc_C[0:300]).append(data_acc_D[0:300])
        
        plt.figure(figsize=(15,8))
        data_plot = DataTrainingACC.copy()
        data_plot['index'] = range(0,DataTrainingACC.shape[0])
        sns.lineplot(data=data_plot, x='index', y='x-axis (g)')
        sns.lineplot(data=data_plot, x='index', y='y-axis (g)')
        sns.lineplot(data=data_plot, x='index', y='z-axis (g)')
        plt.savefig(gambar)

        result = DataTrainingACC.to_json(orient="split")
        parsed = json.loads(result)
        dataJson = json.dumps(parsed, indent=4)
        hasil = render_template('acce.html', dataCsv=dataJson)
       
    elif int(tahap) == 5:   #Traning Data Gyroscope dari 0 - 300 Masing2 Jenis Tahapan (Preparation (300 Data), Welding (300 Data), dll) = 1.200 Data
        data_acc = data_raw_acc[["x-axis (g)","y-axis (g)","z-axis (g)","actid"]]
        data_gyro = data_raw_gyro[["x-axis (deg/s)","y-axis (deg/s)","z-axis (deg/s)","actid"]]
        data_mag = data_raw_mag[["x-axis (T)","y-axis (T)","z-axis (T)","actid"]]
        
        # Split by Action
        data_acc_A  = data_acc[data_acc['actid'] == "Preparation"]
        data_acc_B  = data_acc[data_acc['actid'] == "Grinding"]
        data_acc_C  = data_acc[data_acc['actid'] == "Welding"]
        data_acc_D  = data_acc[data_acc['actid'] == "Slag Cleaning"]

        data_gyro_A = data_gyro[data_gyro['actid'] == "Preparation"]
        data_gyro_B = data_gyro[data_gyro['actid'] == "Grinding"]
        data_gyro_C = data_gyro[data_gyro['actid'] == "Welding"]
        data_gyro_D = data_gyro[data_gyro['actid'] == "Slag Cleaning"]

        data_mag_A  = data_mag[data_mag['actid'] == "Preparation"]
        data_mag_B  = data_mag[data_mag['actid'] == "Grinding"]
        data_mag_C  = data_mag[data_mag['actid'] == "Welding"]
        data_mag_D  = data_mag[data_mag['actid'] == "Slag Cleaning"]
        
        # Select 10000 data for each Action
        data_acc_A  = data_acc_A[0:10000]
        data_acc_B  = data_acc_B[0:10000]
        data_acc_C  = data_acc_C[0:10000]
        data_acc_D  = data_acc_D[0:10000]

        data_gyro_A = data_gyro_A[0:10000]
        data_gyro_B = data_gyro_B[0:10000]
        data_gyro_C = data_gyro_C[0:10000]
        data_gyro_D = data_gyro_D[0:10000]

        data_mag_A  = data_mag_A[0:10000]
        data_mag_B  = data_mag_B[0:10000]
        data_mag_C  = data_mag_C[0:10000]
        data_mag_D  = data_mag_D[0:10000]
        
        # acc

        data_acc_A_1 = data_acc_A[0:100]
        data_acc_B_1 = data_acc_B[0:100]
        data_acc_C_1 = data_acc_C[0:100]
        data_acc_D_1 = data_acc_D[0:100]

        data_acc_A_2 = data_acc_A[100:200]
        data_acc_B_2 = data_acc_B[100:200]
        data_acc_C_2 = data_acc_C[100:200]
        data_acc_D_2 = data_acc_D[100:200]

        data_acc_A_3 = data_acc_A[200:300]
        data_acc_B_3 = data_acc_B[200:300]
        data_acc_C_3 = data_acc_C[200:300]
        data_acc_D_3 = data_acc_D[200:300]

        data_acc_A_4 = data_acc_A[300:400]
        data_acc_B_4 = data_acc_B[300:400]
        data_acc_C_4 = data_acc_C[300:400]
        data_acc_D_4 = data_acc_D[300:400]

        data_acc_A_5 = data_acc_A[400:500]
        data_acc_B_5 = data_acc_B[400:500]
        data_acc_C_5 = data_acc_C[400:500]
        data_acc_D_5 = data_acc_D[400:500]

        # gyro

        data_gyro_A_1 = data_gyro_A[0:100]
        data_gyro_B_1 = data_gyro_B[0:100]
        data_gyro_C_1 = data_gyro_C[0:100]
        data_gyro_D_1 = data_gyro_D[0:100]

        data_gyro_A_2 = data_gyro_A[100:200]
        data_gyro_B_2 = data_gyro_B[100:200]
        data_gyro_C_2 = data_gyro_C[100:200]
        data_gyro_D_2 = data_gyro_D[100:200]

        data_gyro_A_3 = data_gyro_A[200:300]
        data_gyro_B_3 = data_gyro_B[200:300]
        data_gyro_C_3 = data_gyro_C[200:300]
        data_gyro_D_3 = data_gyro_D[200:300]

        data_gyro_A_4 = data_gyro_A[300:400]
        data_gyro_B_4 = data_gyro_B[300:400]
        data_gyro_C_4 = data_gyro_C[300:400]
        data_gyro_D_4 = data_gyro_D[300:400]

        data_gyro_A_5 = data_gyro_A[400:500]
        data_gyro_B_5 = data_gyro_B[400:500]
        data_gyro_C_5 = data_gyro_C[400:500]
        data_gyro_D_5 = data_gyro_D[400:500]

        # mag

        data_mag_A_1 = data_mag_A[0:100]
        data_mag_B_1 = data_mag_B[0:100]
        data_mag_C_1 = data_mag_C[0:100]
        data_mag_D_1 = data_mag_D[0:100]

        data_mag_A_2 = data_mag_A[100:200]
        data_mag_B_2 = data_mag_B[100:200]
        data_mag_C_2 = data_mag_C[100:200]
        data_mag_D_2 = data_mag_D[100:200]

        data_mag_A_3 = data_mag_A[200:300]
        data_mag_B_3 = data_mag_B[200:300]
        data_mag_C_3 = data_mag_C[200:300]
        data_mag_D_3 = data_mag_D[200:300]

        data_mag_A_4 = data_mag_A[300:400]
        data_mag_B_4 = data_mag_B[300:400]
        data_mag_C_4 = data_mag_C[300:400]
        data_mag_D_4 = data_mag_D[300:400]

        data_mag_A_5 = data_mag_A[400:500]
        data_mag_B_5 = data_mag_B[400:500]
        data_mag_C_5 = data_mag_C[400:500]
        data_mag_D_5 = data_mag_D[400:500]
        
        DataTrainingACC = data_gyro_A[0:300].append(data_gyro_B[0:300]).append(data_gyro_C[0:300]).append(data_gyro_D[0:300])
        
        plt.figure(figsize=(15,8))
        data_plot = DataTrainingACC.copy()
        data_plot['index'] = range(0,DataTrainingACC.shape[0])
        sns.lineplot(data=data_plot, x='index', y='x-axis (deg/s)')
        sns.lineplot(data=data_plot, x='index', y='y-axis (deg/s)')
        sns.lineplot(data=data_plot, x='index', y='z-axis (deg/s)')
        plt.savefig(gambar)

        result = DataTrainingACC.to_json(orient="split")
        parsed = json.loads(result)
        dataJson = json.dumps(parsed, indent=4)
        hasil = render_template('acce.html', dataCsv=dataJson)
       
    elif int(tahap) == 6:   #Traning Data Magnetometer dari 0 - 300 Masing2 Jenis Tahapan (Preparation (300 Data), Welding (300 Data), dll) = 1.200 Data
        data_acc = data_raw_acc[["x-axis (g)","y-axis (g)","z-axis (g)","actid"]]
        data_gyro = data_raw_gyro[["x-axis (deg/s)","y-axis (deg/s)","z-axis (deg/s)","actid"]]
        data_mag = data_raw_mag[["x-axis (T)","y-axis (T)","z-axis (T)","actid"]]
        
        # Split by Action
        data_acc_A  = data_acc[data_acc['actid'] == "Preparation"]
        data_acc_B  = data_acc[data_acc['actid'] == "Grinding"]
        data_acc_C  = data_acc[data_acc['actid'] == "Welding"]
        data_acc_D  = data_acc[data_acc['actid'] == "Slag Cleaning"]

        data_gyro_A = data_gyro[data_gyro['actid'] == "Preparation"]
        data_gyro_B = data_gyro[data_gyro['actid'] == "Grinding"]
        data_gyro_C = data_gyro[data_gyro['actid'] == "Welding"]
        data_gyro_D = data_gyro[data_gyro['actid'] == "Slag Cleaning"]

        data_mag_A  = data_mag[data_mag['actid'] == "Preparation"]
        data_mag_B  = data_mag[data_mag['actid'] == "Grinding"]
        data_mag_C  = data_mag[data_mag['actid'] == "Welding"]
        data_mag_D  = data_mag[data_mag['actid'] == "Slag Cleaning"]
        
        # Select 10000 data for each Action
        data_acc_A  = data_acc_A[0:10000]
        data_acc_B  = data_acc_B[0:10000]
        data_acc_C  = data_acc_C[0:10000]
        data_acc_D  = data_acc_D[0:10000]

        data_gyro_A = data_gyro_A[0:10000]
        data_gyro_B = data_gyro_B[0:10000]
        data_gyro_C = data_gyro_C[0:10000]
        data_gyro_D = data_gyro_D[0:10000]

        data_mag_A  = data_mag_A[0:10000]
        data_mag_B  = data_mag_B[0:10000]
        data_mag_C  = data_mag_C[0:10000]
        data_mag_D  = data_mag_D[0:10000]
        
        # acc

        data_acc_A_1 = data_acc_A[0:100]
        data_acc_B_1 = data_acc_B[0:100]
        data_acc_C_1 = data_acc_C[0:100]
        data_acc_D_1 = data_acc_D[0:100]

        data_acc_A_2 = data_acc_A[100:200]
        data_acc_B_2 = data_acc_B[100:200]
        data_acc_C_2 = data_acc_C[100:200]
        data_acc_D_2 = data_acc_D[100:200]

        data_acc_A_3 = data_acc_A[200:300]
        data_acc_B_3 = data_acc_B[200:300]
        data_acc_C_3 = data_acc_C[200:300]
        data_acc_D_3 = data_acc_D[200:300]

        data_acc_A_4 = data_acc_A[300:400]
        data_acc_B_4 = data_acc_B[300:400]
        data_acc_C_4 = data_acc_C[300:400]
        data_acc_D_4 = data_acc_D[300:400]

        data_acc_A_5 = data_acc_A[400:500]
        data_acc_B_5 = data_acc_B[400:500]
        data_acc_C_5 = data_acc_C[400:500]
        data_acc_D_5 = data_acc_D[400:500]

        # gyro

        data_gyro_A_1 = data_gyro_A[0:100]
        data_gyro_B_1 = data_gyro_B[0:100]
        data_gyro_C_1 = data_gyro_C[0:100]
        data_gyro_D_1 = data_gyro_D[0:100]

        data_gyro_A_2 = data_gyro_A[100:200]
        data_gyro_B_2 = data_gyro_B[100:200]
        data_gyro_C_2 = data_gyro_C[100:200]
        data_gyro_D_2 = data_gyro_D[100:200]

        data_gyro_A_3 = data_gyro_A[200:300]
        data_gyro_B_3 = data_gyro_B[200:300]
        data_gyro_C_3 = data_gyro_C[200:300]
        data_gyro_D_3 = data_gyro_D[200:300]

        data_gyro_A_4 = data_gyro_A[300:400]
        data_gyro_B_4 = data_gyro_B[300:400]
        data_gyro_C_4 = data_gyro_C[300:400]
        data_gyro_D_4 = data_gyro_D[300:400]

        data_gyro_A_5 = data_gyro_A[400:500]
        data_gyro_B_5 = data_gyro_B[400:500]
        data_gyro_C_5 = data_gyro_C[400:500]
        data_gyro_D_5 = data_gyro_D[400:500]

        # mag

        data_mag_A_1 = data_mag_A[0:100]
        data_mag_B_1 = data_mag_B[0:100]
        data_mag_C_1 = data_mag_C[0:100]
        data_mag_D_1 = data_mag_D[0:100]

        data_mag_A_2 = data_mag_A[100:200]
        data_mag_B_2 = data_mag_B[100:200]
        data_mag_C_2 = data_mag_C[100:200]
        data_mag_D_2 = data_mag_D[100:200]

        data_mag_A_3 = data_mag_A[200:300]
        data_mag_B_3 = data_mag_B[200:300]
        data_mag_C_3 = data_mag_C[200:300]
        data_mag_D_3 = data_mag_D[200:300]

        data_mag_A_4 = data_mag_A[300:400]
        data_mag_B_4 = data_mag_B[300:400]
        data_mag_C_4 = data_mag_C[300:400]
        data_mag_D_4 = data_mag_D[300:400]

        data_mag_A_5 = data_mag_A[400:500]
        data_mag_B_5 = data_mag_B[400:500]
        data_mag_C_5 = data_mag_C[400:500]
        data_mag_D_5 = data_mag_D[400:500]
        
        DataTrainingACC = data_mag_A[0:300].append(data_mag_B[0:300]).append(data_mag_C[0:300]).append(data_mag_D[0:300])
        
        plt.figure(figsize=(15,8))
        data_plot = DataTrainingACC.copy()
        data_plot['index'] = range(0,DataTrainingACC.shape[0])
        sns.lineplot(data=data_plot, x='index', y='x-axis (T)')
        sns.lineplot(data=data_plot, x='index', y='y-axis (T)')
        sns.lineplot(data=data_plot, x='index', y='z-axis (T)')
        plt.savefig(gambar)

        result = DataTrainingACC.to_json(orient="split")
        parsed = json.loads(result)
        dataJson = json.dumps(parsed, indent=4)
        hasil = render_template('acce.html', dataCsv=dataJson)
        
    elif int(tahap) == 7:   #Testing Data Accelerometer dari 300 - 500 Masing2 Jenis Tahapan (Preparation (300 Data), Welding (300 Data), dll) = 800 Data
        data_acc = data_raw_acc[["x-axis (g)","y-axis (g)","z-axis (g)","actid"]]
        data_gyro = data_raw_gyro[["x-axis (deg/s)","y-axis (deg/s)","z-axis (deg/s)","actid"]]
        data_mag = data_raw_mag[["x-axis (T)","y-axis (T)","z-axis (T)","actid"]]
        
        # Split by Action
        data_acc_A  = data_acc[data_acc['actid'] == "Preparation"]
        data_acc_B  = data_acc[data_acc['actid'] == "Grinding"]
        data_acc_C  = data_acc[data_acc['actid'] == "Welding"]
        data_acc_D  = data_acc[data_acc['actid'] == "Slag Cleaning"]

        data_gyro_A = data_gyro[data_gyro['actid'] == "Preparation"]
        data_gyro_B = data_gyro[data_gyro['actid'] == "Grinding"]
        data_gyro_C = data_gyro[data_gyro['actid'] == "Welding"]
        data_gyro_D = data_gyro[data_gyro['actid'] == "Slag Cleaning"]

        data_mag_A  = data_mag[data_mag['actid'] == "Preparation"]
        data_mag_B  = data_mag[data_mag['actid'] == "Grinding"]
        data_mag_C  = data_mag[data_mag['actid'] == "Welding"]
        data_mag_D  = data_mag[data_mag['actid'] == "Slag Cleaning"]
        
        # Select 10000 data for each Action
        data_acc_A  = data_acc_A[0:10000]
        data_acc_B  = data_acc_B[0:10000]
        data_acc_C  = data_acc_C[0:10000]
        data_acc_D  = data_acc_D[0:10000]

        data_gyro_A = data_gyro_A[0:10000]
        data_gyro_B = data_gyro_B[0:10000]
        data_gyro_C = data_gyro_C[0:10000]
        data_gyro_D = data_gyro_D[0:10000]

        data_mag_A  = data_mag_A[0:10000]
        data_mag_B  = data_mag_B[0:10000]
        data_mag_C  = data_mag_C[0:10000]
        data_mag_D  = data_mag_D[0:10000]
        
        # acc

        data_acc_A_1 = data_acc_A[0:100]
        data_acc_B_1 = data_acc_B[0:100]
        data_acc_C_1 = data_acc_C[0:100]
        data_acc_D_1 = data_acc_D[0:100]

        data_acc_A_2 = data_acc_A[100:200]
        data_acc_B_2 = data_acc_B[100:200]
        data_acc_C_2 = data_acc_C[100:200]
        data_acc_D_2 = data_acc_D[100:200]

        data_acc_A_3 = data_acc_A[200:300]
        data_acc_B_3 = data_acc_B[200:300]
        data_acc_C_3 = data_acc_C[200:300]
        data_acc_D_3 = data_acc_D[200:300]

        data_acc_A_4 = data_acc_A[300:400]
        data_acc_B_4 = data_acc_B[300:400]
        data_acc_C_4 = data_acc_C[300:400]
        data_acc_D_4 = data_acc_D[300:400]

        data_acc_A_5 = data_acc_A[400:500]
        data_acc_B_5 = data_acc_B[400:500]
        data_acc_C_5 = data_acc_C[400:500]
        data_acc_D_5 = data_acc_D[400:500]

        # gyro

        data_gyro_A_1 = data_gyro_A[0:100]
        data_gyro_B_1 = data_gyro_B[0:100]
        data_gyro_C_1 = data_gyro_C[0:100]
        data_gyro_D_1 = data_gyro_D[0:100]

        data_gyro_A_2 = data_gyro_A[100:200]
        data_gyro_B_2 = data_gyro_B[100:200]
        data_gyro_C_2 = data_gyro_C[100:200]
        data_gyro_D_2 = data_gyro_D[100:200]

        data_gyro_A_3 = data_gyro_A[200:300]
        data_gyro_B_3 = data_gyro_B[200:300]
        data_gyro_C_3 = data_gyro_C[200:300]
        data_gyro_D_3 = data_gyro_D[200:300]

        data_gyro_A_4 = data_gyro_A[300:400]
        data_gyro_B_4 = data_gyro_B[300:400]
        data_gyro_C_4 = data_gyro_C[300:400]
        data_gyro_D_4 = data_gyro_D[300:400]

        data_gyro_A_5 = data_gyro_A[400:500]
        data_gyro_B_5 = data_gyro_B[400:500]
        data_gyro_C_5 = data_gyro_C[400:500]
        data_gyro_D_5 = data_gyro_D[400:500]

        # mag

        data_mag_A_1 = data_mag_A[0:100]
        data_mag_B_1 = data_mag_B[0:100]
        data_mag_C_1 = data_mag_C[0:100]
        data_mag_D_1 = data_mag_D[0:100]

        data_mag_A_2 = data_mag_A[100:200]
        data_mag_B_2 = data_mag_B[100:200]
        data_mag_C_2 = data_mag_C[100:200]
        data_mag_D_2 = data_mag_D[100:200]

        data_mag_A_3 = data_mag_A[200:300]
        data_mag_B_3 = data_mag_B[200:300]
        data_mag_C_3 = data_mag_C[200:300]
        data_mag_D_3 = data_mag_D[200:300]

        data_mag_A_4 = data_mag_A[300:400]
        data_mag_B_4 = data_mag_B[300:400]
        data_mag_C_4 = data_mag_C[300:400]
        data_mag_D_4 = data_mag_D[300:400]

        data_mag_A_5 = data_mag_A[400:500]
        data_mag_B_5 = data_mag_B[400:500]
        data_mag_C_5 = data_mag_C[400:500]
        data_mag_D_5 = data_mag_D[400:500]
        
        DataTestingACC = data_acc_A[300:500].append(data_acc_B[300:500]).append(data_acc_C[300:500]).append(data_acc_D[300:500])
        
        plt.figure(figsize=(15,8))
        data_plot = DataTestingACC.copy()
        data_plot['index'] = range(0,DataTestingACC.shape[0])
        sns.lineplot(data=data_plot, x='index', y='x-axis (g)')
        sns.lineplot(data=data_plot, x='index', y='y-axis (g)')
        sns.lineplot(data=data_plot, x='index', y='z-axis (g)')
        plt.savefig(gambar)

        result = DataTestingACC.to_json(orient="split")
        parsed = json.loads(result)
        dataJson = json.dumps(parsed, indent=4)
        hasil = render_template('acce.html', dataCsv=dataJson)
        
    elif int(tahap) == 8:   #Testing Data Gyroscope dari 300 - 500 Masing2 Jenis Tahapan (Preparation (300 Data), Welding (300 Data), dll) = 800 Data
        data_acc = data_raw_acc[["x-axis (g)","y-axis (g)","z-axis (g)","actid"]]
        data_gyro = data_raw_gyro[["x-axis (deg/s)","y-axis (deg/s)","z-axis (deg/s)","actid"]]
        data_mag = data_raw_mag[["x-axis (T)","y-axis (T)","z-axis (T)","actid"]]
        
        # Split by Action
        data_acc_A  = data_acc[data_acc['actid'] == "Preparation"]
        data_acc_B  = data_acc[data_acc['actid'] == "Grinding"]
        data_acc_C  = data_acc[data_acc['actid'] == "Welding"]
        data_acc_D  = data_acc[data_acc['actid'] == "Slag Cleaning"]

        data_gyro_A = data_gyro[data_gyro['actid'] == "Preparation"]
        data_gyro_B = data_gyro[data_gyro['actid'] == "Grinding"]
        data_gyro_C = data_gyro[data_gyro['actid'] == "Welding"]
        data_gyro_D = data_gyro[data_gyro['actid'] == "Slag Cleaning"]

        data_mag_A  = data_mag[data_mag['actid'] == "Preparation"]
        data_mag_B  = data_mag[data_mag['actid'] == "Grinding"]
        data_mag_C  = data_mag[data_mag['actid'] == "Welding"]
        data_mag_D  = data_mag[data_mag['actid'] == "Slag Cleaning"]
        
        # Select 10000 data for each Action
        data_acc_A  = data_acc_A[0:10000]
        data_acc_B  = data_acc_B[0:10000]
        data_acc_C  = data_acc_C[0:10000]
        data_acc_D  = data_acc_D[0:10000]

        data_gyro_A = data_gyro_A[0:10000]
        data_gyro_B = data_gyro_B[0:10000]
        data_gyro_C = data_gyro_C[0:10000]
        data_gyro_D = data_gyro_D[0:10000]

        data_mag_A  = data_mag_A[0:10000]
        data_mag_B  = data_mag_B[0:10000]
        data_mag_C  = data_mag_C[0:10000]
        data_mag_D  = data_mag_D[0:10000]
        
        # acc

        data_acc_A_1 = data_acc_A[0:100]
        data_acc_B_1 = data_acc_B[0:100]
        data_acc_C_1 = data_acc_C[0:100]
        data_acc_D_1 = data_acc_D[0:100]

        data_acc_A_2 = data_acc_A[100:200]
        data_acc_B_2 = data_acc_B[100:200]
        data_acc_C_2 = data_acc_C[100:200]
        data_acc_D_2 = data_acc_D[100:200]

        data_acc_A_3 = data_acc_A[200:300]
        data_acc_B_3 = data_acc_B[200:300]
        data_acc_C_3 = data_acc_C[200:300]
        data_acc_D_3 = data_acc_D[200:300]

        data_acc_A_4 = data_acc_A[300:400]
        data_acc_B_4 = data_acc_B[300:400]
        data_acc_C_4 = data_acc_C[300:400]
        data_acc_D_4 = data_acc_D[300:400]

        data_acc_A_5 = data_acc_A[400:500]
        data_acc_B_5 = data_acc_B[400:500]
        data_acc_C_5 = data_acc_C[400:500]
        data_acc_D_5 = data_acc_D[400:500]

        # gyro

        data_gyro_A_1 = data_gyro_A[0:100]
        data_gyro_B_1 = data_gyro_B[0:100]
        data_gyro_C_1 = data_gyro_C[0:100]
        data_gyro_D_1 = data_gyro_D[0:100]

        data_gyro_A_2 = data_gyro_A[100:200]
        data_gyro_B_2 = data_gyro_B[100:200]
        data_gyro_C_2 = data_gyro_C[100:200]
        data_gyro_D_2 = data_gyro_D[100:200]

        data_gyro_A_3 = data_gyro_A[200:300]
        data_gyro_B_3 = data_gyro_B[200:300]
        data_gyro_C_3 = data_gyro_C[200:300]
        data_gyro_D_3 = data_gyro_D[200:300]

        data_gyro_A_4 = data_gyro_A[300:400]
        data_gyro_B_4 = data_gyro_B[300:400]
        data_gyro_C_4 = data_gyro_C[300:400]
        data_gyro_D_4 = data_gyro_D[300:400]

        data_gyro_A_5 = data_gyro_A[400:500]
        data_gyro_B_5 = data_gyro_B[400:500]
        data_gyro_C_5 = data_gyro_C[400:500]
        data_gyro_D_5 = data_gyro_D[400:500]

        # mag

        data_mag_A_1 = data_mag_A[0:100]
        data_mag_B_1 = data_mag_B[0:100]
        data_mag_C_1 = data_mag_C[0:100]
        data_mag_D_1 = data_mag_D[0:100]

        data_mag_A_2 = data_mag_A[100:200]
        data_mag_B_2 = data_mag_B[100:200]
        data_mag_C_2 = data_mag_C[100:200]
        data_mag_D_2 = data_mag_D[100:200]

        data_mag_A_3 = data_mag_A[200:300]
        data_mag_B_3 = data_mag_B[200:300]
        data_mag_C_3 = data_mag_C[200:300]
        data_mag_D_3 = data_mag_D[200:300]

        data_mag_A_4 = data_mag_A[300:400]
        data_mag_B_4 = data_mag_B[300:400]
        data_mag_C_4 = data_mag_C[300:400]
        data_mag_D_4 = data_mag_D[300:400]

        data_mag_A_5 = data_mag_A[400:500]
        data_mag_B_5 = data_mag_B[400:500]
        data_mag_C_5 = data_mag_C[400:500]
        data_mag_D_5 = data_mag_D[400:500]
        
        DataTestingACC = data_gyro_A[300:500].append(data_gyro_B[300:500]).append(data_gyro_C[300:500]).append(data_gyro_D[300:500])
        
        plt.figure(figsize=(15,8))
        data_plot = DataTestingACC.copy()
        data_plot['index'] = range(0,DataTestingACC.shape[0])
        sns.lineplot(data=data_plot, x='index', y='x-axis (deg/s)')
        sns.lineplot(data=data_plot, x='index', y='y-axis (deg/s)')
        sns.lineplot(data=data_plot, x='index', y='z-axis (deg/s)')
        plt.savefig(gambar)

        result = DataTestingACC.to_json(orient="split")
        parsed = json.loads(result)
        dataJson = json.dumps(parsed, indent=4)
        hasil = render_template('acce.html', dataCsv=dataJson)
        
    elif int(tahap) == 9:   #Testing Data Magnetometer dari 300 - 500 Masing2 Jenis Tahapan (Preparation (300 Data), Welding (300 Data), dll) = 800 Data
        data_acc = data_raw_acc[["x-axis (g)","y-axis (g)","z-axis (g)","actid"]]
        data_gyro = data_raw_gyro[["x-axis (deg/s)","y-axis (deg/s)","z-axis (deg/s)","actid"]]
        data_mag = data_raw_mag[["x-axis (T)","y-axis (T)","z-axis (T)","actid"]]
        
        # Split by Action
        data_acc_A  = data_acc[data_acc['actid'] == "Preparation"]
        data_acc_B  = data_acc[data_acc['actid'] == "Grinding"]
        data_acc_C  = data_acc[data_acc['actid'] == "Welding"]
        data_acc_D  = data_acc[data_acc['actid'] == "Slag Cleaning"]

        data_gyro_A = data_gyro[data_gyro['actid'] == "Preparation"]
        data_gyro_B = data_gyro[data_gyro['actid'] == "Grinding"]
        data_gyro_C = data_gyro[data_gyro['actid'] == "Welding"]
        data_gyro_D = data_gyro[data_gyro['actid'] == "Slag Cleaning"]

        data_mag_A  = data_mag[data_mag['actid'] == "Preparation"]
        data_mag_B  = data_mag[data_mag['actid'] == "Grinding"]
        data_mag_C  = data_mag[data_mag['actid'] == "Welding"]
        data_mag_D  = data_mag[data_mag['actid'] == "Slag Cleaning"]
        
        # Select 10000 data for each Action
        data_acc_A  = data_acc_A[0:10000]
        data_acc_B  = data_acc_B[0:10000]
        data_acc_C  = data_acc_C[0:10000]
        data_acc_D  = data_acc_D[0:10000]

        data_gyro_A = data_gyro_A[0:10000]
        data_gyro_B = data_gyro_B[0:10000]
        data_gyro_C = data_gyro_C[0:10000]
        data_gyro_D = data_gyro_D[0:10000]

        data_mag_A  = data_mag_A[0:10000]
        data_mag_B  = data_mag_B[0:10000]
        data_mag_C  = data_mag_C[0:10000]
        data_mag_D  = data_mag_D[0:10000]
        
        # acc

        data_acc_A_1 = data_acc_A[0:100]
        data_acc_B_1 = data_acc_B[0:100]
        data_acc_C_1 = data_acc_C[0:100]
        data_acc_D_1 = data_acc_D[0:100]

        data_acc_A_2 = data_acc_A[100:200]
        data_acc_B_2 = data_acc_B[100:200]
        data_acc_C_2 = data_acc_C[100:200]
        data_acc_D_2 = data_acc_D[100:200]

        data_acc_A_3 = data_acc_A[200:300]
        data_acc_B_3 = data_acc_B[200:300]
        data_acc_C_3 = data_acc_C[200:300]
        data_acc_D_3 = data_acc_D[200:300]

        data_acc_A_4 = data_acc_A[300:400]
        data_acc_B_4 = data_acc_B[300:400]
        data_acc_C_4 = data_acc_C[300:400]
        data_acc_D_4 = data_acc_D[300:400]

        data_acc_A_5 = data_acc_A[400:500]
        data_acc_B_5 = data_acc_B[400:500]
        data_acc_C_5 = data_acc_C[400:500]
        data_acc_D_5 = data_acc_D[400:500]

        # gyro

        data_gyro_A_1 = data_gyro_A[0:100]
        data_gyro_B_1 = data_gyro_B[0:100]
        data_gyro_C_1 = data_gyro_C[0:100]
        data_gyro_D_1 = data_gyro_D[0:100]

        data_gyro_A_2 = data_gyro_A[100:200]
        data_gyro_B_2 = data_gyro_B[100:200]
        data_gyro_C_2 = data_gyro_C[100:200]
        data_gyro_D_2 = data_gyro_D[100:200]

        data_gyro_A_3 = data_gyro_A[200:300]
        data_gyro_B_3 = data_gyro_B[200:300]
        data_gyro_C_3 = data_gyro_C[200:300]
        data_gyro_D_3 = data_gyro_D[200:300]

        data_gyro_A_4 = data_gyro_A[300:400]
        data_gyro_B_4 = data_gyro_B[300:400]
        data_gyro_C_4 = data_gyro_C[300:400]
        data_gyro_D_4 = data_gyro_D[300:400]

        data_gyro_A_5 = data_gyro_A[400:500]
        data_gyro_B_5 = data_gyro_B[400:500]
        data_gyro_C_5 = data_gyro_C[400:500]
        data_gyro_D_5 = data_gyro_D[400:500]

        # mag

        data_mag_A_1 = data_mag_A[0:100]
        data_mag_B_1 = data_mag_B[0:100]
        data_mag_C_1 = data_mag_C[0:100]
        data_mag_D_1 = data_mag_D[0:100]

        data_mag_A_2 = data_mag_A[100:200]
        data_mag_B_2 = data_mag_B[100:200]
        data_mag_C_2 = data_mag_C[100:200]
        data_mag_D_2 = data_mag_D[100:200]

        data_mag_A_3 = data_mag_A[200:300]
        data_mag_B_3 = data_mag_B[200:300]
        data_mag_C_3 = data_mag_C[200:300]
        data_mag_D_3 = data_mag_D[200:300]

        data_mag_A_4 = data_mag_A[300:400]
        data_mag_B_4 = data_mag_B[300:400]
        data_mag_C_4 = data_mag_C[300:400]
        data_mag_D_4 = data_mag_D[300:400]

        data_mag_A_5 = data_mag_A[400:500]
        data_mag_B_5 = data_mag_B[400:500]
        data_mag_C_5 = data_mag_C[400:500]
        data_mag_D_5 = data_mag_D[400:500]
        
        DataTestingACC = data_mag_A[300:500].append(data_mag_B[300:500]).append(data_mag_C[300:500]).append(data_mag_D[300:500])
        
        plt.figure(figsize=(15,8))
        data_plot = DataTestingACC.copy()
        data_plot['index'] = range(0,DataTestingACC.shape[0])
        sns.lineplot(data=data_plot, x='index', y='x-axis (T)')
        sns.lineplot(data=data_plot, x='index', y='y-axis (T)')
        sns.lineplot(data=data_plot, x='index', y='z-axis (T)')
        plt.savefig(gambar)

        result = DataTestingACC.to_json(orient="split")
        parsed = json.loads(result)
        dataJson = json.dumps(parsed, indent=4)
        hasil = render_template('acce.html', dataCsv=dataJson)
        
    else:
        data_acc = data_raw_acc[["x-axis (g)","y-axis (g)","z-axis (g)","actid"]]
        data_gyro = data_raw_gyro[["x-axis (deg/s)","y-axis (deg/s)","z-axis (deg/s)","actid"]]
        data_mag = data_raw_mag[["x-axis (T)","y-axis (T)","z-axis (T)","actid"]]
        
        # Split by Action
        data_acc_A  = data_acc[data_acc['actid'] == "Preparation"]
        data_acc_B  = data_acc[data_acc['actid'] == "Grinding"]
        data_acc_C  = data_acc[data_acc['actid'] == "Welding"]
        data_acc_D  = data_acc[data_acc['actid'] == "Slag Cleaning"]

        data_gyro_A = data_gyro[data_gyro['actid'] == "Preparation"]
        data_gyro_B = data_gyro[data_gyro['actid'] == "Grinding"]
        data_gyro_C = data_gyro[data_gyro['actid'] == "Welding"]
        data_gyro_D = data_gyro[data_gyro['actid'] == "Slag Cleaning"]

        data_mag_A  = data_mag[data_mag['actid'] == "Preparation"]
        data_mag_B  = data_mag[data_mag['actid'] == "Grinding"]
        data_mag_C  = data_mag[data_mag['actid'] == "Welding"]
        data_mag_D  = data_mag[data_mag['actid'] == "Slag Cleaning"]
        
        # Select 10000 data for each Action
        data_acc_A  = data_acc_A[0:10000]
        data_acc_B  = data_acc_B[0:10000]
        data_acc_C  = data_acc_C[0:10000]
        data_acc_D  = data_acc_D[0:10000]

        data_gyro_A = data_gyro_A[0:10000]
        data_gyro_B = data_gyro_B[0:10000]
        data_gyro_C = data_gyro_C[0:10000]
        data_gyro_D = data_gyro_D[0:10000]

        data_mag_A  = data_mag_A[0:10000]
        data_mag_B  = data_mag_B[0:10000]
        data_mag_C  = data_mag_C[0:10000]
        data_mag_D  = data_mag_D[0:10000]
        # acc

        data_acc_A_1 = data_acc_A[0:100]
        data_acc_B_1 = data_acc_B[0:100]
        data_acc_C_1 = data_acc_C[0:100]
        data_acc_D_1 = data_acc_D[0:100]

        data_acc_A_2 = data_acc_A[100:200]
        data_acc_B_2 = data_acc_B[100:200]
        data_acc_C_2 = data_acc_C[100:200]
        data_acc_D_2 = data_acc_D[100:200]

        data_acc_A_3 = data_acc_A[200:300]
        data_acc_B_3 = data_acc_B[200:300]
        data_acc_C_3 = data_acc_C[200:300]
        data_acc_D_3 = data_acc_D[200:300]

        data_acc_A_4 = data_acc_A[300:400]
        data_acc_B_4 = data_acc_B[300:400]
        data_acc_C_4 = data_acc_C[300:400]
        data_acc_D_4 = data_acc_D[300:400]

        data_acc_A_5 = data_acc_A[400:500]
        data_acc_B_5 = data_acc_B[400:500]
        data_acc_C_5 = data_acc_C[400:500]
        data_acc_D_5 = data_acc_D[400:500]

        # gyro

        data_gyro_A_1 = data_gyro_A[0:100]
        data_gyro_B_1 = data_gyro_B[0:100]
        data_gyro_C_1 = data_gyro_C[0:100]
        data_gyro_D_1 = data_gyro_D[0:100]

        data_gyro_A_2 = data_gyro_A[100:200]
        data_gyro_B_2 = data_gyro_B[100:200]
        data_gyro_C_2 = data_gyro_C[100:200]
        data_gyro_D_2 = data_gyro_D[100:200]

        data_gyro_A_3 = data_gyro_A[200:300]
        data_gyro_B_3 = data_gyro_B[200:300]
        data_gyro_C_3 = data_gyro_C[200:300]
        data_gyro_D_3 = data_gyro_D[200:300]

        data_gyro_A_4 = data_gyro_A[300:400]
        data_gyro_B_4 = data_gyro_B[300:400]
        data_gyro_C_4 = data_gyro_C[300:400]
        data_gyro_D_4 = data_gyro_D[300:400]

        data_gyro_A_5 = data_gyro_A[400:500]
        data_gyro_B_5 = data_gyro_B[400:500]
        data_gyro_C_5 = data_gyro_C[400:500]
        data_gyro_D_5 = data_gyro_D[400:500]

        # mag

        data_mag_A_1 = data_mag_A[0:100]
        data_mag_B_1 = data_mag_B[0:100]
        data_mag_C_1 = data_mag_C[0:100]
        data_mag_D_1 = data_mag_D[0:100]

        data_mag_A_2 = data_mag_A[100:200]
        data_mag_B_2 = data_mag_B[100:200]
        data_mag_C_2 = data_mag_C[100:200]
        data_mag_D_2 = data_mag_D[100:200]

        data_mag_A_3 = data_mag_A[200:300]
        data_mag_B_3 = data_mag_B[200:300]
        data_mag_C_3 = data_mag_C[200:300]
        data_mag_D_3 = data_mag_D[200:300]

        data_mag_A_4 = data_mag_A[300:400]
        data_mag_B_4 = data_mag_B[300:400]
        data_mag_C_4 = data_mag_C[300:400]
        data_mag_D_4 = data_mag_D[300:400]

        data_mag_A_5 = data_mag_A[400:500]
        data_mag_B_5 = data_mag_B[400:500]
        data_mag_C_5 = data_mag_C[400:500]
        data_mag_D_5 = data_mag_D[400:500]
        
        df_x = pd.DataFrame({'v1':[data_acc_A_1["x-axis (g)"],data_acc_B_1["x-axis (g)"],data_acc_C_1["x-axis (g)"],data_acc_D_1["x-axis (g)"],
                               data_acc_A_2["x-axis (g)"],data_acc_B_2["x-axis (g)"],data_acc_C_2["x-axis (g)"],data_acc_D_2["x-axis (g)"],
                               data_acc_A_3["x-axis (g)"],data_acc_B_3["x-axis (g)"],data_acc_C_3["x-axis (g)"],data_acc_D_3["x-axis (g)"]],
                         'v2':[data_acc_A_1["y-axis (g)"],data_acc_B_1["y-axis (g)"],data_acc_C_1["y-axis (g)"],data_acc_D_1["y-axis (g)"],
                               data_acc_A_2["y-axis (g)"],data_acc_B_2["y-axis (g)"],data_acc_C_2["y-axis (g)"],data_acc_D_2["y-axis (g)"],
                               data_acc_A_3["y-axis (g)"],data_acc_B_3["y-axis (g)"],data_acc_C_3["y-axis (g)"],data_acc_D_3["y-axis (g)"]],
                         'v3':[data_acc_A_1["z-axis (g)"],data_acc_B_1["z-axis (g)"],data_acc_C_1["z-axis (g)"],data_acc_D_1["z-axis (g)"],
                               data_acc_A_2["z-axis (g)"],data_acc_B_2["z-axis (g)"],data_acc_C_2["z-axis (g)"],data_acc_D_2["z-axis (g)"],
                               data_acc_A_3["z-axis (g)"],data_acc_B_3["z-axis (g)"],data_acc_C_3["z-axis (g)"],data_acc_D_3["z-axis (g)"]],
                         'v4':[data_gyro_A_1["x-axis (deg/s)"],data_gyro_B_1["x-axis (deg/s)"],data_gyro_C_1["x-axis (deg/s)"],data_gyro_D_1["x-axis (deg/s)"],
                               data_gyro_A_2["x-axis (deg/s)"],data_gyro_B_2["x-axis (deg/s)"],data_gyro_C_2["x-axis (deg/s)"],data_gyro_D_2["x-axis (deg/s)"],
                               data_gyro_A_3["x-axis (deg/s)"],data_gyro_B_3["x-axis (deg/s)"],data_gyro_C_3["x-axis (deg/s)"],data_gyro_D_3["x-axis (deg/s)"]],
                         'v5':[data_gyro_A_1["y-axis (deg/s)"],data_gyro_B_1["y-axis (deg/s)"],data_gyro_C_1["y-axis (deg/s)"],data_gyro_D_1["y-axis (deg/s)"],
                               data_gyro_A_2["y-axis (deg/s)"],data_gyro_B_2["y-axis (deg/s)"],data_gyro_C_2["y-axis (deg/s)"],data_gyro_D_2["y-axis (deg/s)"],
                               data_gyro_A_3["y-axis (deg/s)"],data_gyro_B_3["y-axis (deg/s)"],data_gyro_C_3["y-axis (deg/s)"],data_gyro_D_3["y-axis (deg/s)"]],
                         'v6':[data_gyro_A_1["z-axis (deg/s)"],data_gyro_B_1["z-axis (deg/s)"],data_gyro_C_1["z-axis (deg/s)"],data_gyro_D_1["z-axis (deg/s)"],
                               data_gyro_A_2["z-axis (deg/s)"],data_gyro_B_2["z-axis (deg/s)"],data_gyro_C_2["z-axis (deg/s)"],data_gyro_D_2["z-axis (deg/s)"],
                               data_gyro_A_3["z-axis (deg/s)"],data_gyro_B_3["z-axis (deg/s)"],data_gyro_C_3["z-axis (deg/s)"],data_gyro_D_3["z-axis (deg/s)"]],
                         'v7':[data_mag_A_1["x-axis (T)"],data_mag_B_1["x-axis (T)"],data_mag_C_1["x-axis (T)"],data_mag_D_1["x-axis (T)"],
                               data_mag_A_2["x-axis (T)"],data_mag_B_2["x-axis (T)"],data_mag_C_2["x-axis (T)"],data_mag_D_2["x-axis (T)"],
                               data_mag_A_3["x-axis (T)"],data_mag_B_3["x-axis (T)"],data_mag_C_3["x-axis (T)"],data_mag_D_3["x-axis (T)"]],
                         'v8':[data_mag_A_1["y-axis (T)"],data_mag_B_1["y-axis (T)"],data_mag_C_1["y-axis (T)"],data_mag_D_1["y-axis (T)"],
                               data_mag_A_2["y-axis (T)"],data_mag_B_2["y-axis (T)"],data_mag_C_2["y-axis (T)"],data_mag_D_2["y-axis (T)"],
                               data_mag_A_3["y-axis (T)"],data_mag_B_3["y-axis (T)"],data_mag_C_3["y-axis (T)"],data_mag_D_3["y-axis (T)"]],
                         'v9':[data_mag_A_1["z-axis (T)"],data_mag_B_1["z-axis (T)"],data_mag_C_1["z-axis (T)"],data_mag_D_1["z-axis (T)"],
                               data_mag_A_2["z-axis (T)"],data_mag_B_2["z-axis (T)"],data_mag_C_2["z-axis (T)"],data_mag_D_2["z-axis (T)"],
                               data_mag_A_3["z-axis (T)"],data_mag_B_3["z-axis (T)"],data_mag_C_3["z-axis (T)"],data_mag_D_3["z-axis (T)"]]
                         
                        }) 
        df_x_test1 = pd.DataFrame({'v1':[data_acc_A_4["x-axis (g)"],data_acc_B_4["x-axis (g)"],data_acc_C_4["x-axis (g)"],data_acc_D_4["x-axis (g)"]],
                               'v2':[data_acc_A_4["y-axis (g)"],data_acc_B_4["y-axis (g)"],data_acc_C_4["y-axis (g)"],data_acc_D_4["y-axis (g)"]],
                               'v3':[data_acc_A_4["z-axis (g)"],data_acc_B_4["z-axis (g)"],data_acc_C_4["z-axis (g)"],data_acc_D_4["z-axis (g)"]],
                               'v4':[data_gyro_A_4["x-axis (deg/s)"],data_gyro_B_4["x-axis (deg/s)"],data_gyro_C_4["x-axis (deg/s)"],data_gyro_D_4["x-axis (deg/s)"]],
                               'v5':[data_gyro_A_4["y-axis (deg/s)"],data_gyro_B_4["y-axis (deg/s)"],data_gyro_C_4["y-axis (deg/s)"],data_gyro_D_4["y-axis (deg/s)"]],
                               'v6':[data_gyro_A_4["z-axis (deg/s)"],data_gyro_B_4["z-axis (deg/s)"],data_gyro_C_4["z-axis (deg/s)"],data_gyro_D_4["z-axis (deg/s)"]],
                               'v7':[data_mag_A_4["x-axis (T)"],data_mag_B_4["x-axis (T)"],data_mag_C_4["x-axis (T)"],data_mag_D_4["x-axis (T)"]],
                               'v8':[data_mag_A_4["y-axis (T)"],data_mag_B_4["y-axis (T)"],data_mag_C_4["y-axis (T)"],data_mag_D_4["y-axis (T)"]],
                               'v9':[data_mag_A_4["z-axis (T)"],data_mag_B_4["z-axis (T)"],data_mag_C_4["z-axis (T)"],data_mag_D_4["z-axis (T)"]]
                              }) 
        df_x_test2 = pd.DataFrame({'v1':[data_acc_A_5["x-axis (g)"],data_acc_B_5["x-axis (g)"],data_acc_C_5["x-axis (g)"],data_acc_D_5["x-axis (g)"]],
                               'v2':[data_acc_A_5["y-axis (g)"],data_acc_B_5["y-axis (g)"],data_acc_C_5["y-axis (g)"],data_acc_D_5["y-axis (g)"]],
                               'v3':[data_acc_A_5["z-axis (g)"],data_acc_B_5["z-axis (g)"],data_acc_C_5["z-axis (g)"],data_acc_D_5["z-axis (g)"]],
                               'v4':[data_gyro_A_5["x-axis (deg/s)"],data_gyro_B_5["x-axis (deg/s)"],data_gyro_C_5["x-axis (deg/s)"],data_gyro_D_5["x-axis (deg/s)"]],
                               'v5':[data_gyro_A_5["y-axis (deg/s)"],data_gyro_B_5["y-axis (deg/s)"],data_gyro_C_5["y-axis (deg/s)"],data_gyro_D_5["y-axis (deg/s)"]],
                               'v6':[data_gyro_A_5["z-axis (deg/s)"],data_gyro_B_5["z-axis (deg/s)"],data_gyro_C_5["z-axis (deg/s)"],data_gyro_D_5["z-axis (deg/s)"]],
                               'v7':[data_mag_A_5["x-axis (T)"],data_mag_B_5["x-axis (T)"],data_mag_C_5["x-axis (T)"],data_mag_D_5["x-axis (T)"]],
                               'v8':[data_mag_A_5["y-axis (T)"],data_mag_B_5["y-axis (T)"],data_mag_C_5["y-axis (T)"],data_mag_D_5["y-axis (T)"]],
                               'v9':[data_mag_A_5["z-axis (T)"],data_mag_B_5["z-axis (T)"],data_mag_C_5["z-axis (T)"],data_mag_D_5["z-axis (T)"]]
                              }) 
        df_x_test3 = pd.DataFrame({'v1':[data_acc_B_4["x-axis (g)"],data_acc_C_4["x-axis (g)"],data_acc_A_4["x-axis (g)"],data_acc_D_4["x-axis (g)"]],
                               'v2':[data_acc_B_4["y-axis (g)"],data_acc_C_4["y-axis (g)"],data_acc_A_4["y-axis (g)"],data_acc_D_4["y-axis (g)"]],
                               'v3':[data_acc_B_4["z-axis (g)"],data_acc_C_4["z-axis (g)"],data_acc_A_4["z-axis (g)"],data_acc_D_4["z-axis (g)"]],
                               'v4':[data_gyro_B_4["x-axis (deg/s)"],data_gyro_C_4["x-axis (deg/s)"],data_gyro_A_4["x-axis (deg/s)"],data_gyro_D_4["x-axis (deg/s)"]],
                               'v5':[data_gyro_B_4["y-axis (deg/s)"],data_gyro_C_4["y-axis (deg/s)"],data_gyro_A_4["y-axis (deg/s)"],data_gyro_D_4["y-axis (deg/s)"]],
                               'v6':[data_gyro_B_4["z-axis (deg/s)"],data_gyro_C_4["z-axis (deg/s)"],data_gyro_A_4["z-axis (deg/s)"],data_gyro_D_4["z-axis (deg/s)"]],
                               'v7':[data_mag_B_4["x-axis (T)"],data_mag_C_4["x-axis (T)"],data_mag_A_4["x-axis (T)"],data_mag_D_4["x-axis (T)"]],
                               'v8':[data_mag_B_4["y-axis (T)"],data_mag_C_4["y-axis (T)"],data_mag_A_4["y-axis (T)"],data_mag_D_4["y-axis (T)"]],
                               'v9':[data_mag_B_4["z-axis (T)"],data_mag_C_4["z-axis (T)"],data_mag_A_4["z-axis (T)"],data_mag_D_4["z-axis (T)"]]
                              })
        df_x_test4 = pd.DataFrame({'v1':[data_acc_C_5["x-axis (g)"]],
                               'v2':[data_acc_C_5["y-axis (g)"]],
                               'v3':[data_acc_C_5["z-axis (g)"]],
                               'v4':[data_gyro_C_5["x-axis (deg/s)"]],
                               'v5':[data_gyro_C_5["y-axis (deg/s)"]],
                               'v6':[data_gyro_C_5["z-axis (deg/s)"]],
                               'v7':[data_mag_C_5["x-axis (T)"]],
                               'v8':[data_mag_C_5["y-axis (T)"]],
                               'v9':[data_mag_C_5["z-axis (T)"]]
                              })
        df_y = np.array(["Preparation","Grinding","Welding","Slag Cleaning",
                     "Preparation","Grinding","Welding","Slag Cleaning",
                     "Preparation","Grinding","Welding","Slag Cleaning"])
        steps = [
            ("concatenate", ColumnConcatenator()),
            ("classify", TimeSeriesForestClassifier(n_estimators=100)),
        ]
        
        DataTestingPrediksi = data_acc_A[300:500].append(data_acc_B[300:500]).append(data_acc_C[300:500]).append(data_acc_D[300:500])
        
        plt.figure(figsize=(15,8))
        sns.lineplot(data=DataTestingPrediksi, x='actid', y='x-axis (g)')
        sns.lineplot(data=DataTestingPrediksi, x='actid', y='y-axis (g)')
        sns.lineplot(data=DataTestingPrediksi, x='actid', y='z-axis (g)')
        plt.savefig(gambar)
        
        model = Pipeline(steps)
        model.fit(df_x, df_y)
        model.score(df_x, df_y)
        
        model.predict(df_x)
        model.predict(df_x_test1)
        model.predict(df_x_test2)
        model.predict(df_x_test3)
        model.predict(df_x_test4)
        
        hasilPrediksi = model.predict(df_x_test1)
        mystring = "-".join(hasilPrediksi)
        hasil = "selesai-"+mystring;
        
    return hasil
    
    
@app.route('/getPrediksiFinal',methods=['POST','GET'])
def getPrediksiFinal():
    data_raw_acc  = pd.read_csv("./static/upload/Accelerometer.csv")
    data_raw_gyro = pd.read_csv("./static/upload/Gyroscope.csv")
    data_raw_mag  = pd.read_csv("./static/upload/Magnetometer.csv")
    data_raw_acc.info()
    data_raw_gyro.info()
    data_raw_mag.info()
    data_raw_acc.head()
    
    plt.figure(figsize=(15,8))
    sns.lineplot(data=data_raw_acc, x='elapsed (s)', y='x-axis (g)')
    
    data_acc = data_raw_acc[["x-axis (g)","y-axis (g)","z-axis (g)","actid"]]
    data_acc
    data_acc['actid'].value_counts()
    data_gyro = data_raw_gyro[["x-axis (deg/s)","y-axis (deg/s)","z-axis (deg/s)","actid"]]
    data_gyro
    data_gyro['actid'].value_counts()
    data_mag = data_raw_mag[["x-axis (T)","y-axis (T)","z-axis (T)","actid"]]
    data_mag
    data_mag['actid'].value_counts()
    
    # Split by Action
    data_acc_A  = data_acc[data_acc['actid'] == "Preparation"]
    data_acc_B  = data_acc[data_acc['actid'] == "Grinding"]
    data_acc_C  = data_acc[data_acc['actid'] == "Welding"]
    data_acc_D  = data_acc[data_acc['actid'] == "Slag Cleaning"]

    data_gyro_A = data_gyro[data_gyro['actid'] == "Preparation"]
    data_gyro_B = data_gyro[data_gyro['actid'] == "Grinding"]
    data_gyro_C = data_gyro[data_gyro['actid'] == "Welding"]
    data_gyro_D = data_gyro[data_gyro['actid'] == "Slag Cleaning"]

    data_mag_A  = data_mag[data_mag['actid'] == "Preparation"]
    data_mag_B  = data_mag[data_mag['actid'] == "Grinding"]
    data_mag_C  = data_mag[data_mag['actid'] == "Welding"]
    data_mag_D  = data_mag[data_mag['actid'] == "Slag Cleaning"]
    
    # Select 10000 data for each Action
    data_acc_A  = data_acc_A[0:10000]
    data_acc_B  = data_acc_B[0:10000]
    data_acc_C  = data_acc_C[0:10000]
    data_acc_D  = data_acc_D[0:10000]

    data_gyro_A = data_gyro_A[0:10000]
    data_gyro_B = data_gyro_B[0:10000]
    data_gyro_C = data_gyro_C[0:10000]
    data_gyro_D = data_gyro_D[0:10000]

    data_mag_A  = data_mag_A[0:10000]
    data_mag_B  = data_mag_B[0:10000]
    data_mag_C  = data_mag_C[0:10000]
    data_mag_D  = data_mag_D[0:10000]
    
    plt.figure(figsize=(15,8))
    data_plot = data_acc_A.copy()
    data_plot['index'] = range(0,data_acc_A.shape[0])
    sns.lineplot(data=data_plot, x='index', y='x-axis (g)')
    # acc

    data_acc_A_1 = data_acc_A[0:100]
    data_acc_B_1 = data_acc_B[0:100]
    data_acc_C_1 = data_acc_C[0:100]
    data_acc_D_1 = data_acc_D[0:100]

    data_acc_A_2 = data_acc_A[100:200]
    data_acc_B_2 = data_acc_B[100:200]
    data_acc_C_2 = data_acc_C[100:200]
    data_acc_D_2 = data_acc_D[100:200]

    data_acc_A_3 = data_acc_A[200:300]
    data_acc_B_3 = data_acc_B[200:300]
    data_acc_C_3 = data_acc_C[200:300]
    data_acc_D_3 = data_acc_D[200:300]

    data_acc_A_4 = data_acc_A[300:400]
    data_acc_B_4 = data_acc_B[300:400]
    data_acc_C_4 = data_acc_C[300:400]
    data_acc_D_4 = data_acc_D[300:400]

    data_acc_A_5 = data_acc_A[400:500]
    data_acc_B_5 = data_acc_B[400:500]
    data_acc_C_5 = data_acc_C[400:500]
    data_acc_D_5 = data_acc_D[400:500]

    # gyro

    data_gyro_A_1 = data_gyro_A[0:100]
    data_gyro_B_1 = data_gyro_B[0:100]
    data_gyro_C_1 = data_gyro_C[0:100]
    data_gyro_D_1 = data_gyro_D[0:100]

    data_gyro_A_2 = data_gyro_A[100:200]
    data_gyro_B_2 = data_gyro_B[100:200]
    data_gyro_C_2 = data_gyro_C[100:200]
    data_gyro_D_2 = data_gyro_D[100:200]

    data_gyro_A_3 = data_gyro_A[200:300]
    data_gyro_B_3 = data_gyro_B[200:300]
    data_gyro_C_3 = data_gyro_C[200:300]
    data_gyro_D_3 = data_gyro_D[200:300]

    data_gyro_A_4 = data_gyro_A[300:400]
    data_gyro_B_4 = data_gyro_B[300:400]
    data_gyro_C_4 = data_gyro_C[300:400]
    data_gyro_D_4 = data_gyro_D[300:400]

    data_gyro_A_5 = data_gyro_A[400:500]
    data_gyro_B_5 = data_gyro_B[400:500]
    data_gyro_C_5 = data_gyro_C[400:500]
    data_gyro_D_5 = data_gyro_D[400:500]

    # mag

    data_mag_A_1 = data_mag_A[0:100]
    data_mag_B_1 = data_mag_B[0:100]
    data_mag_C_1 = data_mag_C[0:100]
    data_mag_D_1 = data_mag_D[0:100]

    data_mag_A_2 = data_mag_A[100:200]
    data_mag_B_2 = data_mag_B[100:200]
    data_mag_C_2 = data_mag_C[100:200]
    data_mag_D_2 = data_mag_D[100:200]

    data_mag_A_3 = data_mag_A[200:300]
    data_mag_B_3 = data_mag_B[200:300]
    data_mag_C_3 = data_mag_C[200:300]
    data_mag_D_3 = data_mag_D[200:300]

    data_mag_A_4 = data_mag_A[300:400]
    data_mag_B_4 = data_mag_B[300:400]
    data_mag_C_4 = data_mag_C[300:400]
    data_mag_D_4 = data_mag_D[300:400]

    data_mag_A_5 = data_mag_A[400:500]
    data_mag_B_5 = data_mag_B[400:500]
    data_mag_C_5 = data_mag_C[400:500]
    data_mag_D_5 = data_mag_D[400:500]
    
    df_x = pd.DataFrame({'v1':[data_acc_A_1["x-axis (g)"],data_acc_B_1["x-axis (g)"],data_acc_C_1["x-axis (g)"],data_acc_D_1["x-axis (g)"],
                           data_acc_A_2["x-axis (g)"],data_acc_B_2["x-axis (g)"],data_acc_C_2["x-axis (g)"],data_acc_D_2["x-axis (g)"],
                           data_acc_A_3["x-axis (g)"],data_acc_B_3["x-axis (g)"],data_acc_C_3["x-axis (g)"],data_acc_D_3["x-axis (g)"]],
                     'v2':[data_acc_A_1["y-axis (g)"],data_acc_B_1["y-axis (g)"],data_acc_C_1["y-axis (g)"],data_acc_D_1["y-axis (g)"],
                           data_acc_A_2["y-axis (g)"],data_acc_B_2["y-axis (g)"],data_acc_C_2["y-axis (g)"],data_acc_D_2["y-axis (g)"],
                           data_acc_A_3["y-axis (g)"],data_acc_B_3["y-axis (g)"],data_acc_C_3["y-axis (g)"],data_acc_D_3["y-axis (g)"]],
                     'v3':[data_acc_A_1["z-axis (g)"],data_acc_B_1["z-axis (g)"],data_acc_C_1["z-axis (g)"],data_acc_D_1["z-axis (g)"],
                           data_acc_A_2["z-axis (g)"],data_acc_B_2["z-axis (g)"],data_acc_C_2["z-axis (g)"],data_acc_D_2["z-axis (g)"],
                           data_acc_A_3["z-axis (g)"],data_acc_B_3["z-axis (g)"],data_acc_C_3["z-axis (g)"],data_acc_D_3["z-axis (g)"]],
                     'v4':[data_gyro_A_1["x-axis (deg/s)"],data_gyro_B_1["x-axis (deg/s)"],data_gyro_C_1["x-axis (deg/s)"],data_gyro_D_1["x-axis (deg/s)"],
                           data_gyro_A_2["x-axis (deg/s)"],data_gyro_B_2["x-axis (deg/s)"],data_gyro_C_2["x-axis (deg/s)"],data_gyro_D_2["x-axis (deg/s)"],
                           data_gyro_A_3["x-axis (deg/s)"],data_gyro_B_3["x-axis (deg/s)"],data_gyro_C_3["x-axis (deg/s)"],data_gyro_D_3["x-axis (deg/s)"]],
                     'v5':[data_gyro_A_1["y-axis (deg/s)"],data_gyro_B_1["y-axis (deg/s)"],data_gyro_C_1["y-axis (deg/s)"],data_gyro_D_1["y-axis (deg/s)"],
                           data_gyro_A_2["y-axis (deg/s)"],data_gyro_B_2["y-axis (deg/s)"],data_gyro_C_2["y-axis (deg/s)"],data_gyro_D_2["y-axis (deg/s)"],
                           data_gyro_A_3["y-axis (deg/s)"],data_gyro_B_3["y-axis (deg/s)"],data_gyro_C_3["y-axis (deg/s)"],data_gyro_D_3["y-axis (deg/s)"]],
                     'v6':[data_gyro_A_1["z-axis (deg/s)"],data_gyro_B_1["z-axis (deg/s)"],data_gyro_C_1["z-axis (deg/s)"],data_gyro_D_1["z-axis (deg/s)"],
                           data_gyro_A_2["z-axis (deg/s)"],data_gyro_B_2["z-axis (deg/s)"],data_gyro_C_2["z-axis (deg/s)"],data_gyro_D_2["z-axis (deg/s)"],
                           data_gyro_A_3["z-axis (deg/s)"],data_gyro_B_3["z-axis (deg/s)"],data_gyro_C_3["z-axis (deg/s)"],data_gyro_D_3["z-axis (deg/s)"]],
                     'v7':[data_mag_A_1["x-axis (T)"],data_mag_B_1["x-axis (T)"],data_mag_C_1["x-axis (T)"],data_mag_D_1["x-axis (T)"],
                           data_mag_A_2["x-axis (T)"],data_mag_B_2["x-axis (T)"],data_mag_C_2["x-axis (T)"],data_mag_D_2["x-axis (T)"],
                           data_mag_A_3["x-axis (T)"],data_mag_B_3["x-axis (T)"],data_mag_C_3["x-axis (T)"],data_mag_D_3["x-axis (T)"]],
                     'v8':[data_mag_A_1["y-axis (T)"],data_mag_B_1["y-axis (T)"],data_mag_C_1["y-axis (T)"],data_mag_D_1["y-axis (T)"],
                           data_mag_A_2["y-axis (T)"],data_mag_B_2["y-axis (T)"],data_mag_C_2["y-axis (T)"],data_mag_D_2["y-axis (T)"],
                           data_mag_A_3["y-axis (T)"],data_mag_B_3["y-axis (T)"],data_mag_C_3["y-axis (T)"],data_mag_D_3["y-axis (T)"]],
                     'v9':[data_mag_A_1["z-axis (T)"],data_mag_B_1["z-axis (T)"],data_mag_C_1["z-axis (T)"],data_mag_D_1["z-axis (T)"],
                           data_mag_A_2["z-axis (T)"],data_mag_B_2["z-axis (T)"],data_mag_C_2["z-axis (T)"],data_mag_D_2["z-axis (T)"],
                           data_mag_A_3["z-axis (T)"],data_mag_B_3["z-axis (T)"],data_mag_C_3["z-axis (T)"],data_mag_D_3["z-axis (T)"]]
                     
                    }) 
    df_x
    df_x_test1 = pd.DataFrame({'v1':[data_acc_A_4["x-axis (g)"],data_acc_B_4["x-axis (g)"],data_acc_C_4["x-axis (g)"],data_acc_D_4["x-axis (g)"]],
                           'v2':[data_acc_A_4["y-axis (g)"],data_acc_B_4["y-axis (g)"],data_acc_C_4["y-axis (g)"],data_acc_D_4["y-axis (g)"]],
                           'v3':[data_acc_A_4["z-axis (g)"],data_acc_B_4["z-axis (g)"],data_acc_C_4["z-axis (g)"],data_acc_D_4["z-axis (g)"]],
                           'v4':[data_gyro_A_4["x-axis (deg/s)"],data_gyro_B_4["x-axis (deg/s)"],data_gyro_C_4["x-axis (deg/s)"],data_gyro_D_4["x-axis (deg/s)"]],
                           'v5':[data_gyro_A_4["y-axis (deg/s)"],data_gyro_B_4["y-axis (deg/s)"],data_gyro_C_4["y-axis (deg/s)"],data_gyro_D_4["y-axis (deg/s)"]],
                           'v6':[data_gyro_A_4["z-axis (deg/s)"],data_gyro_B_4["z-axis (deg/s)"],data_gyro_C_4["z-axis (deg/s)"],data_gyro_D_4["z-axis (deg/s)"]],
                           'v7':[data_mag_A_4["x-axis (T)"],data_mag_B_4["x-axis (T)"],data_mag_C_4["x-axis (T)"],data_mag_D_4["x-axis (T)"]],
                           'v8':[data_mag_A_4["y-axis (T)"],data_mag_B_4["y-axis (T)"],data_mag_C_4["y-axis (T)"],data_mag_D_4["y-axis (T)"]],
                           'v9':[data_mag_A_4["z-axis (T)"],data_mag_B_4["z-axis (T)"],data_mag_C_4["z-axis (T)"],data_mag_D_4["z-axis (T)"]]
                          }) 
    df_x_test2 = pd.DataFrame({'v1':[data_acc_A_5["x-axis (g)"],data_acc_B_5["x-axis (g)"],data_acc_C_5["x-axis (g)"],data_acc_D_5["x-axis (g)"]],
                           'v2':[data_acc_A_5["y-axis (g)"],data_acc_B_5["y-axis (g)"],data_acc_C_5["y-axis (g)"],data_acc_D_5["y-axis (g)"]],
                           'v3':[data_acc_A_5["z-axis (g)"],data_acc_B_5["z-axis (g)"],data_acc_C_5["z-axis (g)"],data_acc_D_5["z-axis (g)"]],
                           'v4':[data_gyro_A_5["x-axis (deg/s)"],data_gyro_B_5["x-axis (deg/s)"],data_gyro_C_5["x-axis (deg/s)"],data_gyro_D_5["x-axis (deg/s)"]],
                           'v5':[data_gyro_A_5["y-axis (deg/s)"],data_gyro_B_5["y-axis (deg/s)"],data_gyro_C_5["y-axis (deg/s)"],data_gyro_D_5["y-axis (deg/s)"]],
                           'v6':[data_gyro_A_5["z-axis (deg/s)"],data_gyro_B_5["z-axis (deg/s)"],data_gyro_C_5["z-axis (deg/s)"],data_gyro_D_5["z-axis (deg/s)"]],
                           'v7':[data_mag_A_5["x-axis (T)"],data_mag_B_5["x-axis (T)"],data_mag_C_5["x-axis (T)"],data_mag_D_5["x-axis (T)"]],
                           'v8':[data_mag_A_5["y-axis (T)"],data_mag_B_5["y-axis (T)"],data_mag_C_5["y-axis (T)"],data_mag_D_5["y-axis (T)"]],
                           'v9':[data_mag_A_5["z-axis (T)"],data_mag_B_5["z-axis (T)"],data_mag_C_5["z-axis (T)"],data_mag_D_5["z-axis (T)"]]
                          }) 
    df_x_test3 = pd.DataFrame({'v1':[data_acc_B_4["x-axis (g)"],data_acc_C_4["x-axis (g)"],data_acc_A_4["x-axis (g)"],data_acc_D_4["x-axis (g)"]],
                           'v2':[data_acc_B_4["y-axis (g)"],data_acc_C_4["y-axis (g)"],data_acc_A_4["y-axis (g)"],data_acc_D_4["y-axis (g)"]],
                           'v3':[data_acc_B_4["z-axis (g)"],data_acc_C_4["z-axis (g)"],data_acc_A_4["z-axis (g)"],data_acc_D_4["z-axis (g)"]],
                           'v4':[data_gyro_B_4["x-axis (deg/s)"],data_gyro_C_4["x-axis (deg/s)"],data_gyro_A_4["x-axis (deg/s)"],data_gyro_D_4["x-axis (deg/s)"]],
                           'v5':[data_gyro_B_4["y-axis (deg/s)"],data_gyro_C_4["y-axis (deg/s)"],data_gyro_A_4["y-axis (deg/s)"],data_gyro_D_4["y-axis (deg/s)"]],
                           'v6':[data_gyro_B_4["z-axis (deg/s)"],data_gyro_C_4["z-axis (deg/s)"],data_gyro_A_4["z-axis (deg/s)"],data_gyro_D_4["z-axis (deg/s)"]],
                           'v7':[data_mag_B_4["x-axis (T)"],data_mag_C_4["x-axis (T)"],data_mag_A_4["x-axis (T)"],data_mag_D_4["x-axis (T)"]],
                           'v8':[data_mag_B_4["y-axis (T)"],data_mag_C_4["y-axis (T)"],data_mag_A_4["y-axis (T)"],data_mag_D_4["y-axis (T)"]],
                           'v9':[data_mag_B_4["z-axis (T)"],data_mag_C_4["z-axis (T)"],data_mag_A_4["z-axis (T)"],data_mag_D_4["z-axis (T)"]]
                          })
    df_x_test4 = pd.DataFrame({'v1':[data_acc_C_5["x-axis (g)"]],
                           'v2':[data_acc_C_5["y-axis (g)"]],
                           'v3':[data_acc_C_5["z-axis (g)"]],
                           'v4':[data_gyro_C_5["x-axis (deg/s)"]],
                           'v5':[data_gyro_C_5["y-axis (deg/s)"]],
                           'v6':[data_gyro_C_5["z-axis (deg/s)"]],
                           'v7':[data_mag_C_5["x-axis (T)"]],
                           'v8':[data_mag_C_5["y-axis (T)"]],
                           'v9':[data_mag_C_5["z-axis (T)"]]
                          })
    df_y = np.array(["Preparation","Grinding","Welding","Slag Cleaning",
                 "Preparation","Grinding","Welding","Slag Cleaning",
                 "Preparation","Grinding","Welding","Slag Cleaning"])
              
    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(n_estimators=100)),
    ]

    model = Pipeline(steps)
    model.fit(df_x, df_y)
    model.score(df_x, df_y)
    
    model.predict(df_x)
    model.predict(df_x_test1)
    model.predict(df_x_test2)
    model.predict(df_x_test3)
    model.predict(df_x_test4)
    
    hasilPrediksi = model.predict(df_x_test2)
    mystring=' '.join(hasilPrediksi)
    
    return mystring

@app.route('/getGrafik',methods=['POST','GET'])
def getGrafik():

    gambar = 'static/grafik/getGrafik.png'
    dataGrafik = request.form['dataGrafik']
    warna = request.form['warna']
    warna1 = request.form['warna1']
    warna2 = request.form['warna2']
    warna8 = request.form['warna8']
    jenis = request.form['jenis']
    satuan = request.form['satuan']
    
    df=pd.read_csv('./static/upload/'+dataGrafik)
    
    plt.figure(figsize=(15,8))
    
    if jenis == "ex":
        sns.lineplot(data=df.astype(object), x='elapsed (s)', y='x-axis'+satuan, color=warna)
        
    elif jenis == "ey":
        sns.lineplot(data=df.astype(object), x='elapsed (s)', y='y-axis'+satuan, color=warna)
        
    elif jenis == "ez":
        sns.lineplot(data=df.astype(object), x='elapsed (s)', y='z-axis'+satuan, color=warna)
        
    elif jenis == "pez":
        sns.lineplot(data=df[df.actid=='Preparation'], x='elapsed (s)', y='z-axis'+satuan, color=warna)
        
    elif jenis == "gez":
        sns.lineplot(data=df[df.actid=='Grinding'], x='elapsed (s)', y='z-axis'+satuan, color=warna)
        
    elif jenis == "wez":
        sns.lineplot(data=df[df.actid=='Welding'], x='elapsed (s)', y='z-axis'+satuan, color=warna)
        
    elif jenis == "scez":
        sns.lineplot(data=df[df.actid=='Slag Cleaning'], x='elapsed (s)', y='z-axis'+satuan, color=warna)
        
    elif jenis == "exyz":
        sns.lineplot(data=df.astype(object), x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Elapsed Terhadap X')
        sns.lineplot(data=df.astype(object), x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Elapsed Terhadap Y')
        sns.lineplot(data=df.astype(object), x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Elapsed Terhadap Z')
        
    elif jenis == "pxyz":
        sns.lineplot(data=df[df.actid=='Preparation'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Preparation Terhadap X')
        sns.lineplot(data=df[df.actid=='Preparation'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Preparation Terhadap Y')
        sns.lineplot(data=df[df.actid=='Preparation'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Preparation Terhadap Z')
        
    elif jenis == "gxyz":
        sns.lineplot(data=df[df.actid=='Grinding'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Grinding Terhadap X')
        sns.lineplot(data=df[df.actid=='Grinding'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Grinding Terhadap Y')
        sns.lineplot(data=df[df.actid=='Grinding'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Grinding Terhadap Z')
        
    elif jenis == "wxyz":
        sns.lineplot(data=df[df.actid=='Welding'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Welding Terhadap X')
        sns.lineplot(data=df[df.actid=='Welding'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Welding Terhadap Y')
        sns.lineplot(data=df[df.actid=='Welding'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Welding Terhadap Z')
        
    elif jenis == "scxyz":
        sns.lineplot(data=df[df.actid=='Slag Cleaning'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Slag Cleaning Terhadap X')
        sns.lineplot(data=df[df.actid=='Slag Cleaning'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Slag Cleaning Terhadap Y')
        sns.lineplot(data=df[df.actid=='Slag Cleaning'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Slag Cleaning Terhadap Z')
        
    elif jenis == "oxyz":
        sns.lineplot(data=df[df.actid=='Others'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Others Terhadap X')
        sns.lineplot(data=df[df.actid=='Others'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Others Terhadap Y')
        sns.lineplot(data=df[df.actid=='Others'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Others Terhadap Z')
        
    elif jenis == "gexyz":
        dataGrafik1 = request.form['dataGrafik1']
        dataGrafik2 = request.form['dataGrafik2']
        warna3 = request.form['warna3']
        warna4 = request.form['warna4']
        warna5 = request.form['warna5']
        warna6 = request.form['warna6']
        warna7 = request.form['warna7']
        satuan1 = request.form['satuan1']
        satuan2 = request.form['satuan2']
        df1=pd.read_csv('./static/upload/'+dataGrafik1)
        df2=pd.read_csv('./static/upload/'+dataGrafik2)
        
        sns.lineplot(data=df.astype(object), x='elapsed (s)', y='x-axis'+satuan, color=warna, label=' Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df.astype(object), x='elapsed (s)', y='y-axis'+satuan, color=warna1, label=' Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df.astype(object), x='elapsed (s)', y='z-axis'+satuan, color=warna2, label=' Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df1.astype(object), x='elapsed (s)', y='x-axis'+satuan1, color=warna3, label=' Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df1.astype(object), x='elapsed (s)', y='y-axis'+satuan1, color=warna4, label=' Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df1.astype(object), x='elapsed (s)', y='z-axis'+satuan1, color=warna5, label=' Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df2.astype(object), x='elapsed (s)', y='x-axis'+satuan2, color=warna6, label=' Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df2.astype(object), x='elapsed (s)', y='y-axis'+satuan2, color=warna7, label=' Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df2.astype(object), x='elapsed (s)', y='z-axis'+satuan2, color=warna8, label=' Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Z')
        
    elif jenis == "gpxyz":
        dataGrafik1 = request.form['dataGrafik1']
        dataGrafik2 = request.form['dataGrafik2']
        warna3 = request.form['warna3']
        warna4 = request.form['warna4']
        warna5 = request.form['warna5']
        warna6 = request.form['warna6']
        warna7 = request.form['warna7']
        satuan1 = request.form['satuan1']
        satuan2 = request.form['satuan2']
        df1=pd.read_csv('./static/upload/'+dataGrafik1)
        df2=pd.read_csv('./static/upload/'+dataGrafik2)
        
        sns.lineplot(data=df[df.actid=='Preparation'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Preparation Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df[df.actid=='Preparation'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Preparation Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df[df.actid=='Preparation'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Preparation Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df1[df1.actid=='Preparation'], x='elapsed (s)', y='x-axis'+satuan1, color=warna3, label='Preparation Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df1[df1.actid=='Preparation'], x='elapsed (s)', y='y-axis'+satuan1, color=warna4, label='Preparation Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df1[df1.actid=='Preparation'], x='elapsed (s)', y='z-axis'+satuan1, color=warna5, label='Preparation Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df2[df2.actid=='Preparation'], x='elapsed (s)', y='x-axis'+satuan2, color=warna6, label='Preparation Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df2[df2.actid=='Preparation'], x='elapsed (s)', y='y-axis'+satuan2, color=warna7, label='Preparation Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df2[df2.actid=='Preparation'], x='elapsed (s)', y='z-axis'+satuan2, color=warna8, label='Preparation Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Z')
        
    elif jenis == "ggxyz":
        dataGrafik1 = request.form['dataGrafik1']
        dataGrafik2 = request.form['dataGrafik2']
        warna3 = request.form['warna3']
        warna4 = request.form['warna4']
        warna5 = request.form['warna5']
        warna6 = request.form['warna6']
        warna7 = request.form['warna7']
        satuan1 = request.form['satuan1']
        satuan2 = request.form['satuan2']
        df1=pd.read_csv('./static/upload/'+dataGrafik1)
        df2=pd.read_csv('./static/upload/'+dataGrafik2)
        
        sns.lineplot(data=df[df.actid=='Grinding'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Grinding Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df[df.actid=='Grinding'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Grinding Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df[df.actid=='Grinding'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Grinding Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df1[df1.actid=='Grinding'], x='elapsed (s)', y='x-axis'+satuan1, color=warna3, label='Grinding Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df1[df1.actid=='Grinding'], x='elapsed (s)', y='y-axis'+satuan1, color=warna4, label='Grinding Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df1[df1.actid=='Grinding'], x='elapsed (s)', y='z-axis'+satuan1, color=warna5, label='Grinding Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df2[df2.actid=='Grinding'], x='elapsed (s)', y='x-axis'+satuan2, color=warna6, label='Grinding Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df2[df2.actid=='Grinding'], x='elapsed (s)', y='y-axis'+satuan2, color=warna7, label='Grinding Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df2[df2.actid=='Grinding'], x='elapsed (s)', y='z-axis'+satuan2, color=warna8, label='Grinding Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Z')
        
    elif jenis == "gwxyz":
        dataGrafik1 = request.form['dataGrafik1']
        dataGrafik2 = request.form['dataGrafik2']
        warna3 = request.form['warna3']
        warna4 = request.form['warna4']
        warna5 = request.form['warna5']
        warna6 = request.form['warna6']
        warna7 = request.form['warna7']
        satuan1 = request.form['satuan1']
        satuan2 = request.form['satuan2']
        df1=pd.read_csv('./static/upload/'+dataGrafik1)
        df2=pd.read_csv('./static/upload/'+dataGrafik2)
        
        sns.lineplot(data=df[df.actid=='Welding'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Welding Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df[df.actid=='Welding'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Welding Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df[df.actid=='Welding'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Welding Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df1[df1.actid=='Welding'], x='elapsed (s)', y='x-axis'+satuan1, color=warna3, label='Welding Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df1[df1.actid=='Welding'], x='elapsed (s)', y='y-axis'+satuan1, color=warna4, label='Welding Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df1[df1.actid=='Welding'], x='elapsed (s)', y='z-axis'+satuan1, color=warna5, label='Welding Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df2[df2.actid=='Welding'], x='elapsed (s)', y='x-axis'+satuan2, color=warna6, label='Welding Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df2[df2.actid=='Welding'], x='elapsed (s)', y='y-axis'+satuan2, color=warna7, label='Welding Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df2[df2.actid=='Welding'], x='elapsed (s)', y='z-axis'+satuan2, color=warna8, label='Welding Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Z')
        
    elif jenis == "gscxyz":
        dataGrafik1 = request.form['dataGrafik1']
        dataGrafik2 = request.form['dataGrafik2']
        warna3 = request.form['warna3']
        warna4 = request.form['warna4']
        warna5 = request.form['warna5']
        warna6 = request.form['warna6']
        warna7 = request.form['warna7']
        satuan1 = request.form['satuan1']
        satuan2 = request.form['satuan2']
        df1=pd.read_csv('./static/upload/'+dataGrafik1)
        df2=pd.read_csv('./static/upload/'+dataGrafik2)
        
        sns.lineplot(data=df[df.actid=='Slag Cleaning'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Slag Cleaning Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df[df.actid=='Slag Cleaning'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Slag Cleaning Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df[df.actid=='Slag Cleaning'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Slag Cleaning Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df1[df1.actid=='Slag Cleaning'], x='elapsed (s)', y='x-axis'+satuan1, color=warna3, label='Slag Cleaning Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df1[df1.actid=='Slag Cleaning'], x='elapsed (s)', y='y-axis'+satuan1, color=warna4, label='Slag Cleaning Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df1[df1.actid=='Slag Cleaning'], x='elapsed (s)', y='z-axis'+satuan1, color=warna5, label='Slag Cleaning Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df2[df2.actid=='Slag Cleaning'], x='elapsed (s)', y='x-axis'+satuan2, color=warna6, label='Slag Cleaning Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df2[df2.actid=='Slag Cleaning'], x='elapsed (s)', y='y-axis'+satuan2, color=warna7, label='Slag Cleaning Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df2[df2.actid=='Slag Cleaning'], x='elapsed (s)', y='z-axis'+satuan2, color=warna8, label='Slag Cleaning Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Z')
        
    else:
        dataGrafik1 = request.form['dataGrafik1']
        dataGrafik2 = request.form['dataGrafik2']
        warna3 = request.form['warna3']
        warna4 = request.form['warna4']
        warna5 = request.form['warna5']
        warna6 = request.form['warna6']
        warna7 = request.form['warna7']
        satuan1 = request.form['satuan1']
        satuan2 = request.form['satuan2']
        df1=pd.read_csv('./static/upload/'+dataGrafik1)
        df2=pd.read_csv('./static/upload/'+dataGrafik2)
        
        sns.lineplot(data=df[df.actid=='Others'], x='elapsed (s)', y='x-axis'+satuan, color=warna, label='Other Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df[df.actid=='Others'], x='elapsed (s)', y='y-axis'+satuan, color=warna1, label='Other Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df[df.actid=='Others'], x='elapsed (s)', y='z-axis'+satuan, color=warna2, label='Other Elapsed '+dataGrafik.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df1[df1.actid=='Others'], x='elapsed (s)', y='x-axis'+satuan1, color=warna3, label='Other Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df1[df1.actid=='Others'], x='elapsed (s)', y='y-axis'+satuan1, color=warna4, label='Other Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df1[df1.actid=='Others'], x='elapsed (s)', y='z-axis'+satuan1, color=warna5, label='Other Elapsed '+dataGrafik1.replace('.csv', '')+' Terhadap Z')
        
        sns.lineplot(data=df2[df2.actid=='Others'], x='elapsed (s)', y='x-axis'+satuan2, color=warna6, label='Other Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap X')
        sns.lineplot(data=df2[df2.actid=='Others'], x='elapsed (s)', y='y-axis'+satuan2, color=warna7, label='Other Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Y')
        sns.lineplot(data=df2[df2.actid=='Others'], x='elapsed (s)', y='z-axis'+satuan2, color=warna8, label='Other Elapsed '+dataGrafik2.replace('.csv', '')+' Terhadap Z')
    
    
    plt.savefig(gambar)
    
    return gambar


if __name__ == "__main__":
    app.run(port=5002,debug=True, threaded=True)
