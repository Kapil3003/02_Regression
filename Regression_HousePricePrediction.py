## pipreqs . (to create requirement file needed for streamshare)
## cd to folder
## streamlit run Regression_HousePricePrediction.py
########################## Initialization #####################

import streamlit as st

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
import pickle
import sklearn
from sklearn.cluster import KMeans



import warnings
import os
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('max_colwidth', 150)
st.set_page_config(layout="wide")



@st.cache_data(hash_funcs={dict: lambda _: None})
def load_data():
	
	Data = pd.read_csv ("Bengaluru_House_Data.csv")  

	ET_model = pickle.load(open('ET.pkl', 'rb'))
	RF_model = pickle.load(open('RF.pkl', 'rb'))
	LR_model = pickle.load(open('LR.pkl', 'rb'))
	scaling_ar = pickle.load(open('scaling.pkl', 'rb'))
	# kmeans =   pickle.load(open('kmeans.pkl', 'rb'))

	final_Data = pd.read_csv ("Final_data.csv") 
	final_Data0 = pd.read_csv ("Final_data0.csv")

	final_Data = final_Data.drop(['Unnamed: 0'],axis='columns')



	return Data,ET_model,RF_model,LR_model,final_Data,final_Data0,scaling_ar

Data,ET_model,RF_model,LR_model,final_Data,final_Data0,scaling_ar  = load_data()


#final_Data

Data_M = final_Data0.copy()
Data_M['price_per_sqft'] = Data_M['price']*100000/Data_M['total_sqft']
Data_M['sqft_per_room'] = Data_M['total_sqft']/Data_M['bhk']
#st.write(Data_M.price_per_sqft.describe())

Data_M = Data_M[~(Data_M.total_sqft/Data_M.bhk<300)]
Data_M = Data_M[~(Data_M.total_sqft/Data_M.bhk>1500)]


@st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def load_matplotlib_figure():

	final_Data0 = pd.read_csv ("Final_data0.csv")  
	Data_M = final_Data0.copy()
	Data_M['price_per_sqft'] = Data_M['price']*100000/Data_M['total_sqft']
	Data_M['sqft_per_room'] = Data_M['total_sqft']/Data_M['bhk']
	#st.write(Data_M.price_per_sqft.describe())

	Data_M = Data_M[~(Data_M.total_sqft/Data_M.bhk<300)]
	Data_M = Data_M[~(Data_M.total_sqft/Data_M.bhk>1500)]

	fig_num = 1
	Hist_pps = plt.figure(fig_num,figsize = (5, 5))
	sns.distplot(Data_M.price_per_sqft)
	fig_num +=1

	Hist_spb = plt.figure(fig_num,figsize = (5 , 5))
	sns.distplot(Data_M.sqft_per_room)
	fig_num +=1

	return Hist_spb,Hist_pps

Hist_spb,Hist_pps = load_matplotlib_figure()


# # ########################## Page UI #####################

# ### Overview of Project
tab1, tab2, tab3 = st.tabs([ "WebApp","Project Overview", "Methodology"])

with tab1:
	"## Forecasting - Banglore House Price Prediction"
	" This web app predicts the house price based on input parameters such as Location, Total area, No of bedrooms and bathrooms. It will show the prediction done by multiple models"
	
	"---"
	col_1, col_2, col_3= st.columns(3)

	with col_1:
		#name = st.text_input("What is your name ?")
		area = st.slider('Total area of the house in sqft ', 0, 5000, 500)
		bed_no = st.slider('No of Bedrooms', 1,10, 1)

	with col_2:
		
		location = st.selectbox( 'Location', final_Data0.sort_values('location').location.unique()
)
	with col_3:
		bath_no = st.slider('Number of bathrooms', 1, 10, 1)
		bal_no = st.slider('Number of Balcony', 0, 10, 0)	


	"---"

	st.write("Your input is:- Area of house",area,", Number of bedrooms",bed_no ,", Number of bathrooms" ,bath_no,", Number of Balconies-", 
		bal_no,", and Locality- ",location)	

	#a = 
	# Data
	# final_Data0

	"---"
	input_data = final_Data.iloc[0:1].copy()
	input_data= input_data.drop(['price'],axis='columns')
	input_data.iloc[:] = 0

	input_data.total_sqft = area
	input_data.bath = bath_no
	input_data.balcony = bal_no
	input_data.bhk = bed_no
	input_data["location_"+location] =1

	'Your Input'
	input_data
	input_data = scaling_ar.transform(input_data)
	'Input after Standardisation'
	input_data

	
	Predicted_Price  = ET_model.predict(input_data.reshape(1,-1))

	with col_2:
		'## Prediction'
		predoction = 'Predicted price of house is ' + str(Predicted_Price[0]) + " Lakhs"
		st.success(predoction)
	





with tab2:

	"##### Forecasting - Banglore House Price Prediction"
	'Building a regression model to predict house prices'


	"This is the second project in the ML for Data science series. The aim of this project is to"

	'- Study and practice in depth all the tools required for EDA and feature engineering'
	'- Explore basic regression algo like linear, losso, ridge, decisiontree, random forest, extratreeregressor and their performance matrices'
	'- practice how to build streamlit app and deploy trained models on heroku server'


	st.info(' Main Aim is to understand how the data science pipeline works and get used to basic tools, and not to build accurate model')

	"---"
	'## Sources'
	"[Kaggle DataSet](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)"
	"[Github](https://github.com/Kapil3003/02_Regression)"

	# "---"
	# "- Checkout the app in WebApp tab"
	# "- Checkout the step by step analysis in Methodology tab"
	"---"

	" ## Projects"
	
	"1. Clustering - Customer Segmentation [Github](https://github.com/Kapil3003/01_Clustering/blob/main/Clustering_CustomerSegmentation.ipynb) [[webApp]](https://kapil3003-01-clustering-clustering-streamlit-app-43fp3b.streamlit.app/) "

	"2. Forecasting - Banglore House Price Prediction [Github](https://github.com/Kapil3003/02_Regression/blob/main/.ipynb_checkpoints/Regression_Project-checkpoint.ipynb) [[webApp]](https://kapil3003-02-regression-regression-housepriceprediction-ifckzh.streamlit.app/)" 

	"3.  Binary Classification - Loan Approval Prediction  [Github](https://github.com/Kapil3003/03_Classification/blob/main/Classification_LoanPrediction.ipynb) [[webApp]](https://kapil3003-03-classification-classification-streamlit-app-el6w2c.streamlit.app/)"

	"4. Hyper parameter Optimisation - Breast Cancer Prediction [Github](https://github.com/Kapil3003/04_Hyperparameter_Optimization/blob/main/Hyperparameter%20Optimization.ipynb) "



with tab3:

	#"# Problem Statement"

	"##### Forecasting - Banglore House Price Prediction"
	'Building a regression model to predict house prices'	

	'In this project we will try to predict banglore house prices based on the features provided using various machine learning regression techniques. We compare different machine learning models for their accuracy and performance. In the end we will create a deployable streamlit app and delpoy it on the streamlit server.'	

	

	"---"
	"#### DataSet"
	st.table(Data.head())
	st.write("Shape of the data" , Data.shape)

	'### Exploratory Data Analysis'

	'##### Data Cleaning'
	"TEST"
	"areatype, availability and society not needed as they dont provide good data or is irrelavanet"
	#'e.g areatype it would have been better if all of them were of one category - but for our analysis it is okay to consider all of them are equal.'
	'- also drop na - drop where there is no data'
	'- convert size into number of bedrooms'	

	data = Data.drop(['area_type','society','availability'],axis='columns')
	data = data.dropna()
	data['bhk'] = data['size'].apply(lambda x: int(x.split(' ')[0]))
	data = data.drop(['size'],axis='columns')
	data = data[["location", "total_sqft", "bath" ,"balcony" ,"bhk","price" ]]
	
	'##### Cleaned Data'
	st.table(data.head())
	st.write("Shape of the data" , data.shape)

	"##### Location Stats"

	col_10, col_20,col_30 = st.columns([1,1,3])
	with col_10:
		"Original_Data_occurance"
		location_stats = data['location'].value_counts(ascending=False)
		st.dataframe(location_stats, height=200)




	with col_30:
		st.write('Location has total', len(data['location'].unique()) ,'unique values. Out of which only',len(location_stats[location_stats>10]), 'have occurance of more than 10')
		'We will replace all those locations with < 10 occurance and group them into \'other\''
		
	with col_20:
		"Modified_Data_Occurance"
		location_stats_less_than_10 = location_stats[location_stats<=10]
		data.location = data.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
		st.dataframe(data['location'].value_counts(ascending=False), height=200)


	'## Feature Engineering'

	col_10, col_20,col_30 = st.columns(3)
	with col_10:
		"Now lets add two new feature in order to analyse data more efficiently"
		"1. Price per sqft- how expensive is the unit property"
		"2. Sqft per room - how big is room"

		## Outlier Removal Using Standard Deviation and Mean
		'In Banglore - Clearly price per sqft should be in 400-40000 range, dataset outside those are either wrong entries or outliers in the data'
		'Also Generally  ~ 300-1500 sqft/[bedroom + balcony] (i.e 300 or 1000 for 1bhk with balcony), outside those are clearly outliers '

		"After cleaning the data has normal distribution"

	with col_20:
		st.pyplot(Hist_spb)

	with col_30:
		st.pyplot(Hist_pps)

	'As we utilized these feature to clean the data now we can drop them'

	Data_M = Data_M.drop(['Unnamed: 0','price_per_sqft','sqft_per_room'],axis='columns')
	Data_M = Data_M.reset_index(drop=True)

	#'##### Correlation between parameters'
	#st.table(Data_M.corr())

	'Hot Encoding - for labelling parametes'
	st.dataframe(pd.get_dummies(Data_M))


	"TRain test split"
	"Prediction models - and their accuracy and residues they are good to display"
	"We used Linear Regreesion,Decision Tree Regression,RandomFOrest Regression and Extra Tree Regression"
	"Extraa Tree Regression had the highest accuracy of 76%."

	"Add graphs and code ---"


