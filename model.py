import pandas as pd
import pickle

df = pd.read_csv('data/ispu.csv')

df = df.dropna()

df=df.drop(df[df["no2"]=="---"].index)
df=df.drop(df[df["o3"]=="---"].index)
df=df.drop(df[df["co"]=="---"].index)
df=df.drop(df[df["so2"]=="---"].index)
df=df.drop(df[df["pm25"]=="---"].index)
df=df.drop(df[df["pm10"]=="---"].index)
df = df.astype({"tanggal": object, "stasiun": object,"pm10":int,"pm25":int,"so2":int,"co":int,"o3":int,"no2":int,"max":int,"critical":object,"categori":object})

def detect_outliers(df,x):
    Q1 = df[x].quantile(q=0.25)
    Q3 = df[x].quantile(q=0.75)
    IQR = Q3-Q1
    return df[(df[x] < Q1-1.5*IQR) | (df[x] > Q3+1.5*IQR)]

detect_outliers(df, "pm25").index
df['pm25'].replace(df.loc[116,'pm25'],df['pm25'].mean(),inplace=True)

detect_outliers(df, "so2").index
df['so2'].replace(df.loc[654,'so2'],df['so2'].mean(),inplace=True)

df['so2'].replace(df.loc[654,'so2'],df['so2'].mean(),inplace=True)

df['co'].replace(df.loc[93,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[104,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[123,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[150,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[194,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[262,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[278,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[298,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[312,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[314,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[329,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[343,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[345,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[357,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[360,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[373,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[374,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[375,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[574,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[606,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[715,'co'],df['co'].mean(),inplace=True)
df['co'].replace(df.loc[716,'co'],df['co'].mean(),inplace=True)

detect_outliers(df, "o3").index

df['o3'].replace(df.loc[53,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[329,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[337,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[338,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[340,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[343,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[344,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[345,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[346,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[493,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[503,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[654,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[661,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[668,'o3'],df['o3'].mean(),inplace=True)
df['o3'].replace(df.loc[669,'o3'],df['o3'].mean(),inplace=True)

detect_outliers(df, "no2").index

df['no2'].replace(df.loc[12,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[14,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[15,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[16,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[21,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[22,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[23,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[28,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[29,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[134,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[135,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[273,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[283,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[461,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[462,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[492,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[574,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[578,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[604,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[605,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[606,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[623,'no2'],df['no2'].mean(),inplace=True)
df['no2'].replace(df.loc[746,'no2'],df['no2'].mean(),inplace=True)

detect_outliers(df, "max").index

df['max'].replace(df.loc[116,'max'],df['max'].mean(),inplace=True)
df['max'].replace(df.loc[345,'max'],df['max'].mean(),inplace=True)
df['max'].replace(df.loc[668,'max'],df['max'].mean(),inplace=True)

df.drop(['tanggal'], axis=1, inplace=True)
df.drop(['stasiun'], axis=1, inplace=True)

#one hot encoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')


df = pd.get_dummies(df, columns = ["critical"])

X = df[['pm10',	'pm25',	'so2', 'co', 'o3', 'no2']]
y = df[['categori']]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaler.fit(X)
data_scaled = scaler.transform(X)

column_name = list(X.columns)
df = pd.DataFrame(data=data_scaled, columns= column_name)

#Import library
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X, y)

pickle.dump(scaler, open("minmax.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))