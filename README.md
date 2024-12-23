# MachineLearning

What the code does:

1.Data Handling:
-Reads data from multiple CSV and Parquet files, including training data, test data, and time-series data.
-Merges the time-series data with the main dataset based on the id column.
-Drops the id column after merging.
-Handles categorical features by filling missing values with "Missing" and converting them into numerical representations using label encoding.
-Separates the target variable sii and handles missing values in the target.

2.Exploratory Data Analysis (EDA):
-Displays the first few rows of the time-series data to provide a quick overview.
-Plots the distribution of the target variable sii.
-Calculates and plots the feature importance using a LightGBM model.

3.Model Training and Evaluation:
-Defines the quadratic_weighted_kappa function to calculate the Quadratic Weighted Kappa (QWK) evaluation metric.
-Implements threshold_Rounder and evaluate_predictions functions to round predictions based on optimized thresholds and evaluate the rounded predictions.
-Defines the TrainML function, which performs the following:
-Splits the data into training and validation sets using StratifiedKFold.
-Trains a given model (passed as model_class) on each fold.
-Predicts on the training and validation sets within each fold.
-Calculates and prints the QWK score for each fold.
-Calculates the mean training and validation QWK across all folds.
-Optimizes the thresholds for rounding predictions using scipy.optimize.minimize to maximize the QWK.
-Applies the optimized thresholds to the out-of-fold (OOF) predictions and the test set predictions.
-Calculates and prints the optimized QWK score.
-Generates a submission file with predictions on the test set.
-Implements TrainMLWithVisualization which is similar but also plots:
-Target distribution.
-QWK scores per fold.
-Predicted vs. actual distribution.

4.Model Definition and Ensemble:
-Defines hyperparameters for LightGBM, XGBoost, and CatBoost models.
-Creates instances of these models using the defined hyperparameters.
-Combines these models into a VotingRegressor to create an ensemble.

5.Execution:
-Calls TrainML and TrainMLWithVisualization with the VotingRegressor to train the ensemble model and generate predictions.
-Saves the predictions to a submission.csv file.

How it has been optimized:

1.Time Series Feature Engineering:
-The code extracts basic descriptive statistics (e.g., mean, standard deviation, min, max) from the time series data using pandas.DataFrame.describe().

2.Categorical Feature Encoding:
-Categorical features are converted to numerical representations using Label Encoding.

3.Hyperparameter Tuning:
-The code uses predefined hyperparameter sets for LightGBM, XGBoost, and CatBoost.
-Threshold Optimization: The code optimizes the thresholds used to round the continuous model predictions into discrete classes (0, 1, 2, 3) to maximize the Quadratic Weighted Kappa (QWK) score. This is done using scipy.optimize.minimize with the Nelder-Mead method.

4.Ensemble Modeling:
-LightGBM, XGBoost, and CatBoost are combined using a VotingRegressor, which averages the predictions of each model.

5.Model Evaluation using K-Fold Cross-Validation:
-StratifiedKFold is used to split the data into training and validation sets, ensuring that the class distribution is maintained in each fold.
