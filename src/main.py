from models.knn import train_knn_model
from preprocessing import preprocessing

df = preprocessing(
    data="data/Clean_Dataset.csv",
)

# Model 1: KNN
train_knn_model(
    df=df
)
