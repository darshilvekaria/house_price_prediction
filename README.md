Machine Learning Workflow

    - Data Cleaning & Preprocessing

        - reverse geocoding for fetching district from lat-long

        - Feature scaling

        - Categorical encoding 

        - creating new features using similarity search (using lat-long and applying kmeans and rbf-kernal)

    - Hyperparameter tuning

        - Grid Search and Randomised search

    - Model Training

        - Linear Regression / Decision Tree / Random Forest

    - Model Evaluation
        - RMSE

    - Model Persistence

        - Saving trained model 

        - Inference using Flask


# Raw data is present in datasets folder

# Install dependencies
pip install -r requirements.txt

# Run the following scripts in the sequence:
    1) initial_preprocessing_data.ipynb
    2) house_price_prediction
    3) main.py for inference through command prompt
    4) app.py for inference using web application with professional UI 


