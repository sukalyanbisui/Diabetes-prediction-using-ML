# Diabetes Prediction Using Machine Learning
This project focuses on predicting whether a patient has diabetes or not using a Support Vector Machine (SVM), a popular supervised machine learning algorithm. The model is trained on medical data such as blood glucose levels, insulin levels, and other relevant health metrics to predict diabetes occurrence. The final model achieves an accuracy of 78%.

## Table of Contents
- Introduction
- Dataset
- Data Preprocessing
- Model Training
- Evaluation
- Technologies Used
- Conclusion
  
## Introduction
This project leverages machine learning techniques to classify patients as diabetic or non-diabetic based on certain medical features. A Support Vector Machine (SVM) model is implemented and trained on a dataset containing various health indicators. After preprocessing and training, the model can be used to predict diabetes for new data points.

## Dataset
The dataset contains medical information of patients, including:

- Blood glucose levels
- Insulin levels
- Age
- Body mass index (BMI)
- Outcome (whether the patient has diabetes or not)
The dataset is initially split into training and testing sets to evaluate the model’s performance.

## Data Preprocessing
To ensure the accuracy and consistency of the model, the data was cleaned and standardized. Key steps in preprocessing include:

- Handling missing or null values
- Standardizing features to ensure all values are within the same range
- Splitting the dataset into training and test sets for model evaluation

## Model Training
The core of the project involves training a Support Vector Machine (SVM) classifier. The key steps include:

- Splitting the data into training (80%) and testing (20%) sets.
- Training the SVM model on the training set using medical data features.
- Tuning hyperparameters for the best model performance.

## Evaluation
The trained model is evaluated using the test set, yielding an accuracy of 78%. The following metrics are used to evaluate the performance:

- Accuracy: 0.78
- Precision and Recall: (Optional: Add values if you have them)
- Confusion Matrix: To visualize the true positive, true negative, false positive, and false negative rates.

## Technologies Used
Python: Programming language used for implementation.
Scikit-learn: For machine learning model building and evaluation.
Pandas: For data manipulation and preprocessing.
NumPy: For numerical computations.
Matplotlib & Seaborn: For data visualization and exploratory data analysis.

## Conclusion
This project provides a foundational example of using machine learning for medical data classification. The SVM model, trained on diabetes-related medical data, demonstrates an accuracy of 78%, and can be used for future predictions when new medical data is available.

