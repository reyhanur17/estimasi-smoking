import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

smoking_model = pickle.load(open('smoking_knn.sav','rb'))
df = pd.read_csv('smoking.csv')

st.title('Prediksi Perilaku Merokok Berdasarkan Kondisi Kesehatan')
st.caption('Nur Muhammad Reyhan | 211351106')
st.write('\n')

#visualisasi
features = ['hemoglobin','height(cm)','weight(kg)','triglyceride','Gtp','waist(cm)','serum creatinine']
x = df[features].values
y = df['smoking'].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#knn.fit(x_train,y_train)

pca = PCA(n_components=2)
X_train2D = pca.fit_transform(x_train)
X_test2D = pca.fit_transform(x_test)

def knn_plot() :
    knn=KNeighborsClassifier(n_neighbors=41,metric='euclidean',p=2)
    knn.fit(X_test2D, y_test)
    y_pred_test = knn.predict(X_test2D)
    precision_test = accuracy_score(y_pred_test, y_test) * 100
    knn.fit(X_train2D, y_train)
    y_pred_train = knn.predict(X_train2D)
    precision_train = accuracy_score(y_pred_train, y_train) * 100
    print("Akurasi Data Testing: {0:.2f}%".format(precision_test))
    print("Akurasi Data Training: {0:.2f}%".format(precision_train))

    #Plotting decision boundaries
    plot_decision_regions(X_train2D, y_train, clf=knn, legend=1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    st.header("Visualisasi KNN Model Dengan Neighbor 42")
    st.pyplot()
    plt.title('K-NN Visualiasasi KNN Model Dengan Neighbor 42')
    plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
knn_plot()
st.write(" ")
st.write(" ")

#input klasifikasi
st.header("Input Niai Untuk Klasifikasi")
col1, col2=st.columns(2)
with col1 :
    hemoglobin = st.number_input('Input Nilai Hemogoblin :')
with col1 :
    height = st.number_input('Input Tinggi Pasien :')
with col1 :
    weight = st.number_input('Input Berat Pasien :')
with col1 :
    triglyceride = st.number_input('Input Nilai triglyceride :')
with col2 :
    gtp = st.number_input('Input Nilai GTP :')
with col2 :
    waist = st.number_input('Input Ukuran Pinggang Pasien :')
with col2 :
    serum_creatinine = st.number_input('Input Nilai Serum Creatinine :')

smoking_diagnosis = ''

if st.button('Test Perilaku Merokok Pasien') :
    smoking_prediction = smoking_model.predict([[hemoglobin,height,weight,triglyceride,gtp,waist,serum_creatinine]])

    if(smoking_prediction[0] == 0):
        smoking_diagnosis = 'Pasien Bukan Perokok'
    else :
        smoking_diagnosis = 'Pasien adalah Perokok'

    st.success(smoking_diagnosis)
