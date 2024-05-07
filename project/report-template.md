# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Divanshi Jain

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
When I attempted to submit my predictions, I noticed that some of them were negative values, which isn't compatible with Kaggle's guidelines. To rectify this, I had to tweak the predictor's output to ensure all predictions were non-negative before submission. My solution involved replacing any negative prediction values with 0, thus guaranteeing that all predictions met Kaggle's requirements for submission. This adjustment ensured the validity and acceptability of my predictions for submission.

### What was the top ranked model that performed?
The WeightedEnsemble_L2 model emerged as the top performer in terms of validation score. It achieved an impressive validation root mean squared error (RMSE) score of -84.125061, as evaluated using the root_mean_squared_error metric. Notably, this model boasted a prediction time of approximately 0.080544 seconds on the validation set, with a training time of roughly 0.02 seconds. Its marginal prediction time and marginal training time were approximately 0.001560 seconds and 0.007091 seconds, respectively. With a stack level of 2, this model was part of a stacked ensemble, indicating a sophisticated approach to modeling. Moreover, it possessed the capability for making inferences (can_infer = True) and held a fit order of 7 in the training process.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
In the exploratory analysis, several key findings and feature engineering steps were identified:

1. **Feature Extraction from DateTime**: The year, month, day, and hour (dayofweek) features were extracted as independent features from the datetime feature. This allowed for a more granular analysis of temporal patterns within the dataset. After extraction, the original datetime feature was removed from consideration.

2. **Parsing DateTime Feature**: The datetime feature was parsed as a datetime datatype to retrieve hour information from the timestamp. This step enabled the utilization of time-related information in modeling.

3. **Handling Categorical Variables**: The independent attributes season and weather, initially read as integers, were recognized as categorical variables. They were converted to the category data type to properly represent their nature in modeling.

4. **Creation of Day_Type Feature**: A new categorical feature, day_type, was added based on the holiday and workingday features. This feature effectively segregated observations into "weekday", "weekend", and "holiday" categories, providing additional context for modeling.

5. **Handling Casual and Registered Features**: The casual and registered features, present only in the training dataset and absent in the test data, were identified. These features were ignored during model training to prevent data leakage. Removing them resulted in significant improvement in RMSE scores. Additionally, it was observed that these independent features were highly correlated with the target variable.

Overall, these exploratory analyses and feature engineering steps contributed to a better understanding of the dataset's structure and relationships between variables, enhancing the predictive modeling process.

### How much better did your model preform after adding additional features and why do you think that is?
After adding additional features and conducting exploratory data analysis (EDA), the model's performance improved by 75%. Here's why this improvement may have occurred:

1. **Temporal Features Extraction**: By extracting the year, month, day, day of the week, and hour components from the datetime feature and creating separate attributes for each, the model gained a more nuanced understanding of temporal patterns within the data. This allowed it to capture variations in demand based on time of day, day of the week, and seasonal trends, leading to more accurate predictions.

2. **Conversion of Categorical Variables**: Converting certain categorical variables from integer data types to their actual categorical data types likely improved model performance. Categorical variables like season and weather contain discrete categories that are better represented using the categorical data type. This ensured that the model treated these variables appropriately, leading to better predictions.

3. **Exclusion of Inapplicable Features**: By excluding features like "casual" and "registered" from the training set due to their absence in the test dataset, the model avoided data leakage and improved generalization to unseen data. These features, although highly correlated with the target variable, were not applicable for prediction on the test set and could have introduced bias or overfitting if included in the modeling process.

Overall, the combination of feature engineering techniques, including temporal features extraction, categorical variable conversion, and careful feature selection, likely contributed to the significant improvement in model performance observed after EDA.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Hyperparameter tuning significantly improved the model's performance compared to the initial submission. Through experimentation with three different configurations, we observed notable enhancements. However, the models optimized with hyperparameters, while competitive, didn't surpass the performance achieved with exploratory data analysis (EDA) and additional features when evaluated on the Kaggle test dataset.

Key observations include:

1. **Autogluon Configuration**: Initially, we utilized the autogluon package for training with prescribed settings. However, the performance of hyperparameter-tuned models was suboptimal. This was due to limitations in the exploration options available to autogluon, as hyperparameters were tuned with a fixed set of values provided by the user.

2. **Parameters Influence**: Parameters such as 'time_limit' and 'presets' were crucial during hyperparameter optimization using autogluon. The 'time_limit' parameter determined the duration for model construction, while different 'presets' influenced memory usage and computational resources. We experimented with presets like "high_quality" (with auto_stack enabled) but found them demanding in terms of resources. As a result, we explored lighter and faster presets such as 'medium_quality' and 'optimized_for_deployment.'

3. **Preference for Deployment Optimization**: Ultimately, we preferred the "optimized_for_deployment" preset for hyperparameter optimization, as it provided faster results without compromising too much on quality. Other presets failed to create models within the given time limit for the experimental configurations.

4. **Balancing Exploration and Exploitation**: A challenge arose in balancing exploration and exploitation when using autogluon with predefined hyperparameter ranges. Finding the right balance between exploring a wide range of hyperparameters and exploiting promising configurations was crucial for achieving optimal model performance.

In summary, while hyperparameter tuning yielded improvements, it was ultimately the combination of exploratory data analysis, feature engineering, and careful selection of hyperparameters that led to superior model performance on the Kaggle test dataset.

### If you were given more time with this dataset, where do you think you would spend more time?
Given more time with this dataset, I would focus on the following areas to further enhance model performance and gain deeper insights:

1. **Feature Engineering and Selection**: I would continue to explore the dataset for potential features that could improve model performance. This could involve creating new features based on domain knowledge, experimenting with different transformations, and selecting the most informative features using techniques like feature importance analysis, correlation analysis, and dimensionality reduction methods.

2. **Fine-tuning Hyperparameters**: While hyperparameter tuning was performed, there may still be room for improvement by exploring a wider range of hyperparameters and tuning them more exhaustively. This could involve using more advanced techniques such as Bayesian optimization or genetic algorithms to search the hyperparameter space more efficiently.

3. **Model Ensemble and Stacking**: I would experiment with different ensemble methods and stacking techniques to combine the predictions of multiple models. By leveraging the strengths of diverse models, ensemble methods can often improve predictive performance and robustness.

4. **Further Exploratory Data Analysis**: I would delve deeper into the dataset to uncover additional insights and patterns that could be leveraged to improve model performance. This could involve exploring interactions between features, investigating outliers and anomalies, and identifying potential areas for data cleaning and preprocessing.

5. **Model Interpretability**: Understanding the factors driving model predictions is crucial for building trust and gaining insights from the model. I would spend time exploring techniques for model interpretability, such as feature importance analysis, partial dependence plots, and SHAP (SHapley Additive exPlanations) values, to gain insights into how the model makes predictions and identify areas for further improvement.

Overall, with more time, I would aim to iteratively refine the modeling process, experiment with different techniques, and gain a deeper understanding of the dataset to ultimately build a more accurate and robust predictive model.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|prescribed_values|prescribed_values|"presets: 'high quality' (auto_stack=True)"|1.80351|
|add_features|	prescribed_values|prescribed_values	"presets: 'high quality' (auto_stack=True)"|0.4803|
|hpo|Tree-Based Models: (GBM, XT, XGB & RF)|KNN|"presets: 'optimize_for_deployment"|0.51487

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![image](https://github.com/DivanshiJain2005/Predict-Bike-Sharing-Demand/assets/121867251/e6650d70-3344-42d1-9382-abd46621af60)


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![image](https://github.com/DivanshiJain2005/Predict-Bike-Sharing-Demand/assets/121867251/9cc625cd-d4a0-4195-a97c-1b55e4eaf608)


## Summary
The project extensively utilized the AutoGluon AutoML framework for Tabular Data, automating the training and evaluation of machine learning models. AutoGluon facilitated the creation of automated stack ensembles and individually configured regression models, tailoring models for specific tasks and combining predictions for improved performance.

AutoGluon allowed for quick prototyping of baseline models, speeding up the development and testing of initial models. However, the top-performing model benefited significantly from extensive exploratory data analysis (EDA) and feature engineering, highlighting the importance of human-driven insights in enhancing model performance.

Although automatic hyperparameter tuning, model selection, and ensembling capabilities of AutoGluon contributed to exploring the best options, they did not surpass the model with EDA and feature engineering alone. Challenges in hyperparameter tuning were observed due to factors such as time limits, prescribed presets, model families, and hyperparameter ranges.

Overall, AutoGluon proved valuable in facilitating quick prototyping, automatic stack ensembling, and hyperparameter tuning. However, the most significant performance gains came from integrating EDA and feature engineering, emphasizing the importance of human expertise in building effective predictive models.
