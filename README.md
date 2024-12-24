# Machine Learning

We are KM group of the machine learning class INT3405E 56. 
Our members:
- Hoàng Đức Minh (22028051)
- Trần Trọng Thịnh (22028073)
- Nguyễn Đức Kiên (22028285)
We are thrilled to be part of the Child Mind Institute — Problematic Internet Use competition. This project has given us a unique opportunity to address an increasingly important issue that affects millions of people worldwide – Problematic Internet Use.

# 1. About the contest
In today’s digital age, the internet enriches education, communication, and entertainment. However, excessive use among young people can harm mental health, social interactions, and physical well-being. This competition focuses on predicting the Severe Internet Addiction Index (sii) to assess problematic internet use in children and adolescents.
The Goal: The core objective of this competition is to develop a machine learning model capable of predicting SII. This index is a classification variable that categorizes individuals into four levels of severity – 0, 1, 2, and 3 – reflecting the degree of problematic internet behaviors.

# 2. Models of our project
Our project leverages a combination of advanced machine learning models to predict the Severe Internet Addiction Index (SII). We utilize MLP (Multi-Layer Perceptron), LightGBM, XGBoost, and CatBoost to build a robust ensemble model. Each model contributes unique strengths, enhancing prediction accuracy and providing deeper insights into problematic internet use among children and adolescents.

# 3. Reasons to use the models
a. Multi-Layer Perceptron (MLP)
- Flexibility and Non-Linearity: MLP can model complex patterns and non-linear relationships between features, which is crucial for understanding the diverse factors contributing to problematic internet use.
- Feature Interactions: MLP can automatically learn interactions between different features without manual feature engineering. This is important for datasets with multiple behavioral and demographic indicators.
- Robustness: MLP performs well on large datasets and can generalize across various inputs, helping to improve prediction accuracy for the SII classification task.

![image](https://github.com/user-attachments/assets/8e833da2-64f2-4839-89d2-f871d17c2cf5)

b. LightGBM
- Speed and Efficiency: LightGBM is known for its fast training speed and low memory usage, making it ideal for large datasets. It can handle high-dimensional data efficiently, which is essential for our project involving numerous features.
- Accuracy and Interpretability: LightGBM consistently delivers high accuracy and provides feature importance metrics, allowing us to understand which factors most influence internet addiction levels.
- Handling Imbalanced Data: LightGBM includes techniques for handling imbalanced datasets, improving model performance when certain SII classes have fewer samples.

c. XGBoost
- Boosting Power: XGBoost is a powerful gradient boosting model that reduces bias and variance, improving overall prediction accuracy.
- Handling Missing Data: XGBoost can naturally handle missing values, making it robust for real-world datasets where data completeness might be an issue.
- Consistency and Stability: It often outperforms other models in competitions and real-world applications, making it a reliable choice for predicting SII.

d. CatBoost
- Categorical Data Handling: CatBoost is specifically designed to handle categorical data efficiently without extensive preprocessing, which aligns well with the mixed nature of our dataset (categorical and numerical).
- Fast and Accurate: CatBoost delivers high accuracy while being faster to train compared to other boosting algorithms, making it a practical addition to our ensemble model.
- Reduced Overfitting: CatBoost uses advanced regularization techniques, helping prevent overfitting even with limited data.

e. Why Use All Four Models?
- Ensemble Power: By combining MLP, LightGBM, XGBoost, and CatBoost, we leverage the strengths of each model, increasing overall prediction performance and reducing the likelihood of model bias.
- Comprehensive Insights: Different models excel at capturing different patterns in the data, allowing for a more comprehensive analysis of internet addiction factors.
- Robustness: The ensemble approach provides more stable and reliable predictions, crucial for accurately identifying children and adolescents at risk of problematic internet use.

![image](https://github.com/user-attachments/assets/1a94a312-a0b5-4c3a-bc37-fbe547d07a36)

![image](https://github.com/user-attachments/assets/50a3a10b-ae0d-427c-8997-55d57d935be5)

# 4 How we optimized the models
a. Hyperparameter Tuning
- For models like LightGBM, XGBoost, and CatBoost, we tuned hyperparameters to enhance model performance.
- This fine-tuning helped reduce overfitting and improved the balance between bias and variance.

b. Cross-Validation (K-Fold)
- We implemented Stratified K-Fold Cross-Validation to ensure each fold had a proportional representation of each class.
- This technique maximized the use of data, ensuring the model was trained and validated on different portions, reducing the risk of overfitting to a single subset.

c. Ensemble Learning
- By combining predictions from LightGBM, XGBoost, CatBoost, and MLP models, we created an ensemble model that leveraged the strengths of each algorithm.
- This ensemble approach improved overall accuracy and robustness.

d. Feature Engineering and Selection
- We filtered out unnecessary features and focused on those that had the most predictive power.
- Categorical features were mapped to numerical values, and time-related data was merged to provide more context.

e. Threshold Optimization
- After initial predictions, we applied threshold tuning using techniques like the Nelder-Mead optimization method.
- This step helped adjust classification boundaries to improve metrics like the Quadratic Weighted Kappa (QWK) score.

f. Handling Missing Data
- Records with missing values in the target variable (SII) were removed to ensure data integrity.
- For categorical features, missing values were handled systematically to prevent biases during training.

