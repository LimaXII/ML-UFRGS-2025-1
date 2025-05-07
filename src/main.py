import matplotlib
import pandas as pd

from models.knn import train_knn_model
from models.random_forest import train_random_forest
from models.gradient_boosting import train_gradient_boosting_model
from models.svr import train_svr_model
from models.decision_tree import train_decision_tree_model
from models.linear_regression import train_linear_regression_model
from preprocessing import preprocessing

# DEBUG: Se remover isso o código quebra kk
matplotlib.use('Agg')

# Comentado pois não precisamos processar novamente os dados, por agora.
# df = preprocessing(data="data/Clean_Dataset.csv")

df = pd.read_csv("data/flights_data_processed.csv")

# Model 1: KNN
#train_knn_model(df=df)

# Model 2: Random Forest
#train_random_forest(df=df)

# Model 3: Gradient Boosting Regressor
#train_gradient_boosting_model(df=df)

# Model 4: SVR
#train_svr_model(df=df)

# Model 5: Decision Tree Regressor
#train_decision_tree_model(df=df)

# Model 6: Linear Regression
train_linear_regression_model(df=df)
