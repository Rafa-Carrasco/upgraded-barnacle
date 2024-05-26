from utils import db_connect
engine = db_connect()

# your code here
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. descargar data

# url = "https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv"
# respuesta = requests.get(url)
# nombre_archivo = "demographic_health_data.csv"
# with open(nombre_archivo, 'wb') as archivo:
#      archivo.write(respuesta.content)

# 2. convertir csv en dataframe

total_data = pd.read_csv("../data/raw/demographic_health_data.csv")


# borrar duplicados
total_data_sin = total_data.drop_duplicates()  
total_data_sin.shape
# no hay duplicados

total_data.head()
ncols = total_data.columns.tolist()
print(ncols)

# eliminar colñumnas irrelevantes. Muchos valores tienen diferencias medidas par el miosmo valor (95% CI, number, etc..)  asi que podemos eliminar algunas referencias 

total_data_cl1 = total_data.drop(['0-9', '19-Oct', '20-29','30-39','40-49','50-59', '60-69' ,'70-79','80+','White-alone pop', 'Black-alone pop','Native American/American Indian-alone pop','Asian-alone pop','Hawaiian/Pacific Islander-alone pop','Two or more races pop',], axis=1) 
total_data_cl2 = total_data_cl1.drop(['TOT_POP','Population Aged 60+','Percent of Population Aged 60+', 'fips', 'CNTY_FIPS', 'STATE_FIPS', 'CI90LBINC_2018', 'CI90UBINC_2018', 'Obesity_Lower 95% CI', 'Obesity_Upper 95% CI', 'Heart disease_Lower 95% CI', 'Heart disease_Upper 95% CI', 'COPD_Lower 95% CI', 'COPD_Upper 95% CI', 'diabetes_Lower 95% CI', 'diabetes_Upper 95% CI', 'CKD_Lower 95% CI', 'CKD_Upper 95% CI', 'anycondition_Lower 95% CI', 'anycondition_Upper 95% CI', 'anycondition_number', 'Obesity_number', 'Heart disease_number', 'COPD_number', 'diabetes_number', 'CKD_number'   ], axis=1)
# total_data_cl2.info()

# el comando info nos dice que no hay null en ninguna fila. Para confirmar

total_nan = total_data_cl2.isna().sum().sum()
print('Total de valores NaN en el DataFrame es igual a', total_nan)

# vamos a agrupar las variables en conjuntos relacionados por dominios: edades, grupos etnico, datos educacion, datos laborales, datos economicos, salud, recursos sanitarios
Cl_cols = total_data_cl2.columns.tolist()
print(Cl_cols)


edad_sub = total_data_cl2[[ '0-9 y/o % of total pop', '10-19 y/o % of total pop', '20-29 y/o % of total pop', '30-39 y/o % of total pop', '40-49 y/o % of total pop', '50-59 y/o % of total pop', '60-69 y/o % of total pop', '70-79 y/o % of total pop', '80+ y/o % of total pop' ]] 
etnic_sub = total_data_cl2[['% White-alone', '% Black-alone', '% NA/AI-alone', '% Asian-alone', '% Hawaiian/PI-alone', '% Two or more races',]]
pop_sub = total_data_cl2[['Total Population', 'POP_ESTIMATE_2018', 'N_POP_CHG_2018', 'GQ_ESTIMATES_2018', 'R_birth_2018', 'R_death_2018', 'R_NATURAL_INC_2018', 'R_INTERNATIONAL_MIG_2018', 'R_DOMESTIC_MIG_2018', 'R_NET_MIG_2018', 'county_pop2018_18 and older' ]]
edu_sub = total_data_cl2[['Less than a high school diploma 2014-18', 'High school diploma only 2014-18', "Some college or associate's degree 2014-18", "Bachelor's degree or higher 2014-18", 'Percent of adults with less than a high school diploma 2014-18', 'Percent of adults with a high school diploma only 2014-18', "Percent of adults completing some college or associate's degree 2014-18", "Percent of adults with a bachelor's degree or higher 2014-18" ]]
work_sub = total_data_cl2[['Civilian_labor_force_2018', 'Employed_2018', 'Unemployed_2018', 'Unemployment_rate_2018']]
econ_sub = total_data_cl2[['POVALL_2018', 'PCTPOVALL_2018', 'PCTPOV017_2018', 'PCTPOV517_2018', 'MEDHHINC_2018','Median_Household_Income_2018','Median_Household_Income_2018', 'Med_HH_Income_Percent_of_State_Total_2018',  ]]
health_sub = total_data_cl2[['anycondition_prevalence', 'Obesity_prevalence', 'Heart disease_prevalence', 'COPD_prevalence', 'diabetes_prevalence', 'CKD_prevalence',]]
sani_sub = total_data_cl2[['Active Physicians per 100000 Population 2018 (AAMC)', 'Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active General Surgeons per 100000 Population 2018 (AAMC)', 'Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)', 'Total nurse practitioners (2019)', 'Total physician assistants (2019)', 'Total Hospitals (2019)', 'Internal Medicine Primary Care (2019)', 'Family Medicine/General Practice Primary Care (2019)', 'Total Specialist Physicians (2019)', 'ICU Beds_x',]]

# dejamos fuera 'COUNTY_NAME', 'STATE_NAME', 'Urban_rural_code']

# graficamos las variables pro grupos

print("Grupos de edades")
print(edad_sub.describe())
sns.pairplot(edad_sub, kind='reg')
plt.show()


# analisis descriptivo por grupos


print("Estadísticas descriptivas básicas de los grupos de edad")
print(edad_sub.describe())

print("Estadísticas descriptivas básicas de los grupos por nivel educativ")
print(edu_sub.describe())

print("Estadísticas descriptivas básicas de los grupos etnicos")
print(etnic_sub.describe())

print("Estadísticas descriptivas básicas de los grupos segun empleo")
print(work_sub.describe())

print("Estadísticas descriptivas básicas de los grupos segun salud")
print(health_sub.describe())

print("Estadísticas descriptivas básicas de los grupos demoigraficos")
print(pop_sub.describe())

print("Estadísticas descriptivas básicas de los grupos segun ingresos")
print(econ_sub.describe())

print("Estadísticas descriptivas básicas de los grupos de recursos sanitarios")
print(sani_sub.describe())


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Seleccionar solo las columnas numéricas
numeric_columns = total_data_cl2.select_dtypes(include=['float64', 'int64']).columns
df_numeric = total_data_cl2[numeric_columns]

# Aplicar el escalamiento solo a las columnas numéricas
scaler = MinMaxScaler()
df_numeric_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=numeric_columns)
print(df_numeric_scaled.head())

from sklearn.model_selection import train_test_split

# Separar características (X) y variable objetivo (y)
X = df_numeric_scaled.drop(columns=["Active Physicians per 100000 Population 2018 (AAMC)"])
y = df_numeric_scaled['Active Physicians per 100000 Population 2018 (AAMC)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Conjunto de entrenamiento (X_train):")
print(X_train)
print("\nConjunto de prueba (X_test):")
print(X_test)
print("\nVariable objetivo de entrenamiento (y_train):")
print(y_train)
print("\nVariable objetivo de prueba (y_test):")
print(y_test)

from sklearn.feature_selection import SelectKBest, f_regression

k = int(len(X_train.columns) * 0.3)
selection_model = SelectKBest(score_func = f_regression, k = k)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()

X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

X_train_sel["Active Physicians per 100000 Population 2018 (AAMC)"] = list(y_train)
X_test_sel["Active Physicians per 100000 Population 2018 (AAMC)"] = list(y_test)

X_train_sel.to_csv("../data/processed/cl_train.csv", index = False)
X_test_sel.to_csv("../data/processed/cl_test.csv", index = False)

train_data = pd.read_csv("../data/processed/cl_train.csv")
test_data = pd.read_csv("../data/processed/cl_test.csv")

train_data.head()

X_train = train_data.drop(["Active Physicians per 100000 Population 2018 (AAMC)"], axis = 1)
y_train = train_data["Active Physicians per 100000 Population 2018 (AAMC)"]
X_test = test_data.drop(["Active Physicians per 100000 Population 2018 (AAMC)"], axis = 1)
y_test = test_data["Active Physicians per 100000 Population 2018 (AAMC)"]

y_test.info()
y_test.head()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercepto (a): {model.intercept_}")
print(f"Coeficientes (b): {model.coef_}")


y_pred = model.predict(X_test)
y_pred


from sklearn.metrics import mean_squared_error, r2_score

print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred)}")
print(f"Coeficiente de determinación: {r2_score(y_test, y_pred)}")

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha = 20, max_iter = 300)
lasso_model.fit(X_train, y_train)
score = lasso_model.score(X_test, y_test)
print("Coefficients:", lasso_model.coef_)
print("R2 score:", score)