import numpy as np
import pandas as pd
import requests
import joblib  # assuming model is saved using joblib or pickle
from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LinearRegression
# from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


#-------------------------------------- STEP 1: Custom Transformers & Pipelines ----------------------------------#

# Define a function to compute the ratio of the first column to the second
def column_ratio(X):
    if hasattr(X, "iloc"):
        # It's a DataFrame
        return X.iloc[:, [0]].values / X.iloc[:, [1]].values
    else:
        # It's a NumPy array
        return X[:, [0]] / X[:, [1]]

# Define output feature name for the ratio transformation
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

# Create a pipeline for processing ratio features
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

# Custom transformer that computes similarity to k-means clusters using RBF kernel
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X)  # checks that X is an array with finite float values
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _). To be definately provided as validation for production code
        X = check_array(X)
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Pipeline for categorical feature processing
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False))

# Pipeline for numerical features requiring log transformation
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())



#-------------------------------------- STEP 2: User Input Collection ----------------------------------#

# Collect user input for prediction
district = input("District/State_County: ")
housing_median_age = float(input("housing_median_age: "))
total_rooms = int(input("total_rooms: "))
total_bedrooms = int(input("total_bedrooms: "))
population = int(input("population: "))
households = int(input("households: "))
median_income = float(input("median_income: "))
ocean_proximity = input("ocean_proximity: ")




#-------------------------------------- STEP 3: Load & Filter Dataset ----------------------------------#


# Load full housing dataset
housing = pd.read_csv(r"datasets\housing\housing_with_districts.csv")

# Filter dataset to match the user-selected district
candidates_uncleaned = housing[housing["district"] == district]

# Impute missing numeric values in candidate data
imputer = SimpleImputer(strategy="median")
candidates_imputed = imputer.fit_transform(candidates_uncleaned.select_dtypes(include=[np.number]))
candidates = pd.DataFrame(candidates_imputed, columns=candidates_uncleaned.select_dtypes(include=[np.number]).columns)

# print("candidates", candidates)

# # List of numeric feature columns
feature_cols = ["housing_median_age", "total_rooms", "total_bedrooms",
                "population", "households", "median_income"]

# Standardize candidate data for distance calculation
scaler = StandardScaler()
scaled_candidates = scaler.fit_transform(candidates[feature_cols])




#-------------------------------------- STEP 4: Prepare User Input for Prediction ----------------------------------#


# Convert user input to DataFrame for processing
unscaled_input= pd.DataFrame([[
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income
]], columns=feature_cols)

# Standardize user input for fair comparison
scaled_input = scaler.transform(unscaled_input)


#-------------------------------------- STEP 5: Calculate Distance Between Input and Candidates ----------------------------------#



# Compute Euclidean distances between input and all candidates
distances = euclidean_distances(scaled_candidates, scaled_input)
# print("distances", distances)

# # Identify the most similar housing sample (closest match based on lowest distance)
best_idx = distances.argmin()
best_row = candidates.iloc[best_idx]
# print("best_row", best_row)

# Extract location (latitude, longitude) of best match
lat, lon = best_row["latitude"], best_row["longitude"]  



#-------------------------------------- STEP 6: Construct final input row with geographic and other user inputs ----------------------------------#

input_df = pd.DataFrame([{
    "latitude": lat,
    "longitude": lon,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# print("input_df", input_df)



#-------------------------------------- STEP 7: Generate Prediction Using Pretrained Model ----------------------------------#

# Load the pre-trained model
final_model_reloaded = joblib.load("model_assets/housing_model.pkl")

# Prediction
result = final_model_reloaded.predict(input_df)

# # Display the predicted house value
print(f"\n\t\t This House Price should be around: {round(result[0])} $")
