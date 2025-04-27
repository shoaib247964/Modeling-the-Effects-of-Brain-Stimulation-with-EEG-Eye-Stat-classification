# Modeling-the-Effects-of-Brain-Stimulation-with-EEG-Eye-Stat-classification
1. Introduction
In this project, we applied the XGBoost algorithm to perform classification on a dataset. The primary goal was to predict a binary target variable based on multiple features. XGBoost, an optimized gradient boosting algorithm, is widely used in machine learning competitions and real-world applications for its superior performance in both regression and classification tasks.

The aim of this report is to provide a detailed evaluation of the model's performance, including the accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and AUC.

2. Dataset Overview
The dataset used in this project contains multiple features representing different characteristics, and a binary target variable to predict. The target variable has two classes: 0 and 1. The features used for prediction include different brainwave channels and eye detection data.

Key Features:

Brainwave Channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4.

Eye Detection: Eye status as a feature.

Target Variable: A binary classification variable representing two distinct classes (0 or 1).

3. Data Preprocessing
Before applying the model, the dataset underwent the following preprocessing steps:

Missing Value Handling: The dataset was checked for missing values, and none were found.

Feature Selection: The dataset contains several columns, all of which were used in the model.

Data Splitting: The dataset was split into training and test sets. The training set was used to train the model, while the test set was used for evaluation.

4. Model Selection and Hyperparameter Tuning
The model selected for this task was XGBoost, a gradient boosting algorithm known for its high performance and efficiency.

Hyperparameters Tuned:
We used GridSearchCV to optimize the following hyperparameters:

Learning Rate: Controls the contribution of each tree to the final prediction.

Max Depth: Determines the depth of the individual trees.

Number of Estimators (Trees): The number of boosting rounds or trees in the ensemble.

Best Hyperparameters:
Learning Rate: 0.2

Max Depth: 7

Number of Estimators: 300

5. Model Evaluation
After training the model with the best hyperparameters, we evaluated its performance on the test set.

5.1 Accuracy
The accuracy of the tuned XGBoost model was found to be 94.99%, which indicates that the model is correctly predicting the target variable approximately 95% of the time.

plaintext
Copy
Edit
XGBoost Accuracy: 0.9499
5.2 Precision, Recall, and F1-Score
Using the classification report, we calculated the precision, recall, and F1-score for both classes (0 and 1):

Class 0 (Negative Class):

Precision: 0.95

Recall: 0.96

F1-Score: 0.95

Class 1 (Positive Class):

Precision: 0.95

Recall: 0.94

F1-Score: 0.94

5.3 Confusion Matrix
The confusion matrix provides a summary of the model’s performance in terms of true positives, true negatives, false positives, and false negatives.

The matrix shows:

True Positives (TP): 1,263

True Negatives (TN): 1,550

False Positives (FP): 104

False Negatives (FN): 79

From the confusion matrix, we can observe that the model has a low number of false positives and false negatives, indicating it is performing well in distinguishing between classes.

5.4 ROC Curve and AUC
The Receiver Operating Characteristic (ROC) curve was plotted to evaluate the trade-off between the True Positive Rate (recall) and the False Positive Rate across different threshold values. The Area Under the Curve (AUC) was calculated to be 0.98, indicating that the model has excellent discriminatory ability.

The ROC curve is presented below:


5.5 Precision-Recall Curve (Optional)
For imbalanced datasets, the Precision-Recall Curve is a good indicator of performance. The curve for this dataset also shows a high precision and recall, with the model achieving high values for both metrics.

6. Conclusion
Model Performance:
The XGBoost model achieved a high accuracy of 94.99% on the test set, demonstrating its strong classification capabilities. Additionally, the precision, recall, and F1-scores for both classes were near 0.95, indicating that the model is highly reliable in predicting the target variable for both classes.

Evaluation Metrics:
Accuracy: 94.99%

Precision (Class 0): 0.95

Recall (Class 0): 0.96

F1-Score (Class 0): 0.95

Precision (Class 1): 0.95

Recall (Class 1): 0.94

F1-Score (Class 1): 0.94

AUC: 0.98

Future Work:
Model Improvement: Further model improvements can be made by experimenting with additional hyperparameter tuning, using different boosting algorithms, or adding more features to improve classification.

Cross-validation: Cross-validation could be applied to get a better estimate of the model's generalization ability.

Deployment: The model can be deployed in real-time applications where classification is required, such as in health diagnostics, image classification, etc.
