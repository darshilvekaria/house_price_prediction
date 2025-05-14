from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import joblib  
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


#--------------------------------------  Custom Transformers & Pipelines ----------------------------------#

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


app = Flask(__name__)

# Load model and dataset once
model = joblib.load("model_assets/housing_model.pkl")
housing = pd.read_csv(r"datasets/housing/housing_with_districts.csv")

# Define feature columns used in distance calculation
feature_cols = ["housing_median_age", "total_rooms", "total_bedrooms",
                "population", "households", "median_income"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Collect form input
        district = request.form["district"]
        housing_median_age = float(request.form["housing_median_age"])
        total_rooms = int(request.form["total_rooms"])
        total_bedrooms = int(request.form["total_bedrooms"])
        population = int(request.form["population"])
        households = int(request.form["households"])
        median_income = float(request.form["median_income"])
        ocean_proximity = request.form["ocean_proximity"]

        # Filter data for district
        candidates = housing[housing["district"] == district]
        imputer = SimpleImputer(strategy="median")
        candidates_imputed = imputer.fit_transform(candidates.select_dtypes(include=[np.number]))
        candidates = pd.DataFrame(candidates_imputed, columns=candidates.select_dtypes(include=[np.number]).columns)

        feature_cols = ["housing_median_age", "total_rooms", "total_bedrooms",
                        "population", "households", "median_income"]

        scaler = StandardScaler()
        scaled_candidates = scaler.fit_transform(candidates[feature_cols])

        unscaled_input = pd.DataFrame([[housing_median_age, total_rooms, total_bedrooms,
                                        population, households, median_income]],
                                      columns=feature_cols)
        scaled_input = scaler.transform(unscaled_input)
        distances = euclidean_distances(scaled_candidates, scaled_input)
        best_idx = distances.argmin()
        best_row = candidates.iloc[best_idx]
        lat, lon = best_row["latitude"], best_row["longitude"]

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

        prediction = model.predict(input_df)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
