import os
import warnings
import logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.models import *
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.preprocessing import sequence
from ucimlrepo import fetch_ucirepo


try:
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
except:
    print("\n**********************************************************\nfailed to connect to database server.\nmake sure you are connected to the internet.\nif you are and you still see this message, run it again\n**********************************************************\n")
    exit()
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets

X=X.values
ytemp=y.values.ravel()

#conversion from 'M' and 'B' to 0 and 1 for LSTM
y=[]
for i in ytemp:
    if(i == 'M'):
        y.append(1)
    else:
        y.append(0)
X = np.array(X)
y= np.array(y)

#calculations for each measurement
def Calculations(matrix, Brier,y_pred_len,avg,auc):
    tp,fp,fn,tn = matrix.ravel()
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    precision = tp /(tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall)/(precision+recall))
    accuracy = (tp + tn)/(tp+tn+fp+fn)
    error = (fp + fn)/(tp+tn+fp+fn)
    bacc = (tpr + tnr)/2
    tss = (tpr - fpr)
    hss = ((2*(tp * tn - fp * fn))/((fp+fn)*(fn+tn)+(tp+fp)*(fp+tn)))
    if(avg==False):
        Brier = Brier/y_pred_len
    return pd.DataFrame([{"TP":tp,"TN":tn,"FP":fp,"FN":fn,"TPR": tpr, "TNR": tnr,
                          "FPR": fpr, "FNR":fnr, "Precision": precision, "Accuracy":accuracy,
                          "Recall":recall ,"F1_measure":f1,"Error_rate":error,"BACC":bacc,
                          "TSS":tss,"HSS":hss,"Brier_Score":Brier, "AUC":auc}])

neighbors = np.arange(1, 9) 
tf.random.set_seed(7)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
knn_metrics_list = pd.DataFrame()
knn_matrix=[]
rf_metrics_list = pd.DataFrame()
rf_matrix=[]
lstm_metrics_list = pd.DataFrame()
lstm_matrix=[]
run=1;
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #KNN calculation
    for i, k in enumerate(neighbors): 
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
    knny_pred = knn.predict(X_test)
    knnBrier=0
    for i in range(len(knny_pred)):
        if(knny_pred[i].round() != y_test[i]):
            knnBrier+=1
    knncm = confusion_matrix(y_test,knny_pred.round())
    knn_matrix.append(knncm)
    knnauc=roc_auc_score(y_test, knny_pred)
    knn_metrics_list=pd.concat([knn_metrics_list,Calculations(knncm,knnBrier,len(knny_pred),False,knnauc)])

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #random forest calculation
    clf = RandomForestClassifier(n_estimators = 569)
    clf.fit(X_train, y_train)
    rfy_pred = clf.predict(X_test)
    rfBrier=0
    for i in range(len(rfy_pred)):
        if(rfy_pred[i].round() != y_test[i]):
            rfBrier+=1
    rfcm = confusion_matrix(y_test,rfy_pred.round())
    rf_matrix.append(rfcm)
    rfauc=roc_auc_score(y_test, rfy_pred)
    rf_metrics_list=pd.concat([rf_metrics_list,Calculations(rfcm,rfBrier,len(rfy_pred),False,rfauc)])
    
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #LSTM calculation
    lstmX_train = sequence.pad_sequences(X_train, maxlen=30, dtype='float32')
    lstmX_test = sequence.pad_sequences(X_test, maxlen=30, dtype='float32')
    embedding_vecor_length = 256
    model = tf.keras.Sequential()
    model.add(Embedding(5000, embedding_vecor_length))
    model.add(LSTM(569))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("-----|Run ",run,"|-----")
    run+=1
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
    lstmy_pred = model.predict(X_test)
    lstmBrier=0
    for i in range(len(lstmy_pred)):
        if(lstmy_pred[i].round() != y_test[i]):
            lstmBrier+=1
    lstmcm = confusion_matrix(y_test,lstmy_pred.round())
    lstm_matrix.append(lstmcm)
    lstmauc=roc_auc_score(y_test, lstmy_pred)
    lstm_metrics_list=pd.concat([lstm_metrics_list,Calculations(lstmcm,lstmBrier,len(lstmy_pred),False,lstmauc)])

#KNN average
mean_of_conf_matrix_arrays = np.mean(knn_matrix, axis=0)
flipped=knn_metrics_list.T
Brier_total = flipped.loc["Brier_Score"]
Brier_avg=0
for val in Brier_total:
    Brier_avg+=val
Brier_avg=Brier_avg/10
auc_total=flipped.loc["AUC"]
auc_avg=0
for val in auc_total:
    auc_avg+=val
auc_avg=auc_avg/10
knn_metrics_list=pd.concat([knn_metrics_list,Calculations(mean_of_conf_matrix_arrays,Brier_avg,0,True,auc_avg)])
knn_metrics_list.index=['fold_1','fold_2','fold_3','fold_4','fold_5','fold_6','fold_7','fold_8','fold_9','fold_10','Mean']
print("\n----K Nearest Neighbor----")
print(knn_metrics_list.T)

#Random Forest average
mean_of_conf_matrix_arrays = np.mean(rf_matrix, axis=0)
flipped=rf_metrics_list.T
Brier_total = flipped.loc["Brier_Score"]
Brier_avg=0
for val in Brier_total:
    Brier_avg+=val
Brier_avg=Brier_avg/10
auc_total=flipped.loc["AUC"]
auc_avg=0
for val in auc_total:
    auc_avg+=val
auc_avg=auc_avg/10
rf_metrics_list=pd.concat([rf_metrics_list,Calculations(mean_of_conf_matrix_arrays,Brier_avg,0,True,auc_avg)])
rf_metrics_list.index=['fold_1','fold_2','fold_3','fold_4','fold_5','fold_6','fold_7','fold_8','fold_9','fold_10','Mean']
print("\n----Random Forest----")
print(rf_metrics_list.T)

#LSTM average
mean_of_conf_matrix_arrays = np.mean(lstm_matrix, axis=0)
flipped=lstm_metrics_list.T
Brier_total = flipped.loc["Brier_Score"]
Brier_avg=0
for val in Brier_total:
    Brier_avg+=val
Brier_avg=Brier_avg/10
auc_total=flipped.loc["AUC"]
auc_avg=0
for val in auc_total:
    auc_avg+=val
auc_avg=auc_avg/10
lstm_metrics_list=pd.concat([lstm_metrics_list,Calculations(mean_of_conf_matrix_arrays,Brier_avg,0,True,auc_avg)])
lstm_metrics_list.index=['fold_1','fold_2','fold_3','fold_4','fold_5','fold_6','fold_7','fold_8','fold_9','fold_10','Mean']
print("\n----Long Short Term Memory----")
print(lstm_metrics_list.T)
mean=pd.DataFrame()

#comparing averages from all 3=
mean = pd.concat([mean, knn_metrics_list.loc["Mean"].rename("KNN_AVG")], axis=1)
mean = pd.concat([mean, rf_metrics_list.loc["Mean"].rename("RF_AVG")], axis=1)
mean = pd.concat([mean, lstm_metrics_list.loc["Mean"].rename("LSTM_AVG")], axis=1)
print(mean.round(decimals=2))
