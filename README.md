## Machine Learning Pipeline for Predictive Modeling

Developed an end-to-end machine learning pipeline to automate the process of data ingestion, transformation, model training, and evaluation. The project followed best practices for scalability, modularity, and maintainability in machine learning workflows.

* Data Ingestion: Implemented a robust data ingestion pipeline to load, clean, and split datasets into training and testing sets.
* Data Transformation: Applied preprocessing techniques, including feature scaling (StandardScaler, RobustScaler), encoding categorical variables (OneHotEncoder), and handling skewness with Yeo-Johnson transformations.
* Model Training and Tuning: Trained multiple machine learning models (Random Forest, Gradient Boosting, XGBoost, etc.) with hyperparameter tuning using GridSearchCV to optimize performance.
* Model Evaluation: Evaluated models using metrics like R² score on both training and testing datasets to ensure consistency and robustness.
* Automation and Logging: Integrated logging mechanisms and exception handling for seamless debugging and tracking pipeline execution.
* Reusable Components: Packaged pipeline components into reusable modules for scalability and ease of deployment.
* Technologies Used: Python, Scikit-learn, Pandas, NumPy, XGBoost, CatBoost, GridSearchCV, Logging, and Dill for model persistence.
* Outcome: Achieved a best model R² score and automated the workflow for future datasets.
