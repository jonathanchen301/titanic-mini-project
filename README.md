# Titanic Survival Prediction with Scikit-Learn Pipelines

This project applies scikit-learn pipelines to the classic Kaggle Titanic dataset, demonstrating
how to build an end-to-end ML workflow with proper preprocessing, model training, and evaluation.

## Project Structure

- 'titanic_pipeline.ipynb': main notebook

## Goals

- Practice building preprocessing pipelines with:
    - SimpleImputer (handle missing values)
    - StandardScaler (scale continuous features)
    - OneHotEncoder (encode categorical features)
- Apply ColumnTransformer to handle numeric and categorical features differently
- Train and evaluate models using full pipelines to avoid data leakage
- Compare Logistic Regression and Random Forest Classifiers
- Hyperparameter tuning and cross-validation within Scikit-learn pipelines

## Results

- Model accuracy:
    - 78% test accuracy on Logistic Regression
    - 79% test accuracy on Random Forest
- Random Forest outperformed Logistic Regression slightly

## Next Steps

- Feature engineering (FamilySize, Profession extracted from Name)
- Gradient boosting methods (XGBoost)
- Evaluate addiational metrics like F1 or ROC-AUC