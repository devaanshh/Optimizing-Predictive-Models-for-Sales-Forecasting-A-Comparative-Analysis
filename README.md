
# Optimizing-Predictive-Models-for-Sales-Forecasting-A-Comparative-Analysis

This project focuses on optimizing predictive models for sales forecasting through a comparative analysis. The goal is to develop a model that accurately predicts sales while avoiding overfitting to the training data. Various regression models, including Linear Regression, Random Forest Regression, and XGBoost Regression, are employed and evaluated based on their Root Mean Squared Error (RMSE) scores. Feature engineering techniques, such as encoding date features as cyclic variables and one-hot encoding categorical variables, are applied to preprocess the data. The Random Forest Regression model is ultimately selected for its balance between accuracy and generalization. The project concludes with forecasting sales for a specific time period using the selected model.


## Installation

- Python: The programming language used for development.
- Pandas: A powerful data manipulation and analysis library.
- NumPy: A fundamental package for scientific computing with Python.
- Matplotlib: A comprehensive library for creating static, animated, and interactive visualizations in Python.
- Seaborn: A Python data visualization library based on Matplotlib.
- Scikit-learn: A machine learning library for Python, which includes tools for data mining and data analysis.
- XGBoost: An optimized distributed gradient boosting library.
- Category Encoders: A library for encoding categorical variables.
- Jupyter Notebook: An open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and narrative text.

These installations can typically be done using Python's package manager, pip, with commands like pip install pandas, pip install numpy, and so on.
## Project Structure

- `EDA`: Exploratory Data Analysis 
- `Modeling`: Model building and evaluation
- `Preprocessing`: Preprocessing steps for both training and test datasets
- `README.md`: Project overview and instructions
- `Training Dataset.csv`: Dataset used for training
- `Test Dataset.csv`: Dataset used for testing


## Model Selection

Based on the Root Mean Squared Error (RMSE) scores, the Random Forest Regression model was selected due to its balance between accuracy and generalization.


## Feature Engineering

- Date features were extracted and encoded as cyclic features to capture seasonal patterns.
- Categorical variables were one-hot encoded and numerical features were scaled using Min-Max scaling.


## Model Evaluation

- Linear Regression, Random Forest Regression, and XGBoost Regression models were evaluated based on their RMSE scores.
- Random Forest Regression outperformed the other models in terms of accuracy and generalization.



## Forecasting

The selected Random Forest Regression model was used to forecast sales for June 2021.


## Tech Stack

The primary components utilized in this project include:

- **Python**: Used as the programming language for data preprocessing, analysis, modeling, and visualization.

- **Jupyter Notebook**: Employed as the integrated development environment (IDE) for executing Python code interactively and documenting the analysis process.

- **Libraries**: Various Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, and Category Encoders are utilized for data manipulation, analysis, visualization, and machine learning modeling.


## Conclusion

This project aimed to optimize predictive models for sales forecasting by analyzing a provided dataset through various stages, including exploratory data analysis (EDA), feature engineering, preprocessing, model selection, and evaluation. EDA revealed insights into the dataset's characteristics, while feature engineering techniques were employed to extract relevant information and enhance model predictive power. Preprocessing steps ensured data readiness for modeling, including handling outliers, scaling numerical features, and encoding categorical variables. Subsequently, several regression models were trained and evaluated using root mean squared error (RMSE), with the Random Forest Regression model emerging as the top performer. Leveraging this model, sales forecasts were generated for future periods, demonstrating its practical applicability. Overall, this project underscores the significance of thorough data analysis, feature engineering, and model selection in optimizing sales forecasting models, offering valuable insights for strategic decision-making.


## Authors

- [@devaanshh](https://github.com/devaanshh)


## 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devansh-singh-61743b23b/)


