import matplotlib
import pandas as pd

from models.knn import train_knn_model
from models.random_forest import train_random_forest
from preprocessing import preprocessing

# DEBUG: Se remover isso o código quebra kk
matplotlib.use('Agg')

# Comentado pois não precisamos processar novamente os dados, por agora.
# df = preprocessing(data="data/Clean_Dataset.csv")

df = pd.read_csv("data/flights_data_processed.csv")

# Model 1: KNN
train_knn_model(df=df)

# Model 2: Random Forest
train_random_forest(df=df)
