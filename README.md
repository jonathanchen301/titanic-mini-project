# Titanic Survival Prediction

This project applies scikit-learn pipelines to the classic Kaggle Titanic dataset, demonstrating
how to build an end-to-end ML workflow with proper preprocessing, model training, and evaluation.

## Project Structure

- data/
    - train.csv: Training dataset
    - test.csv: Test dataset (unused within this project)
    - gender_submission.csv: Sample submission on Kaggle (unused within this project)
- titanic-pipeline.ipynb: Pipeline implementation
- tiantic-ensemble_cv.ipynb: Ensemble methods comparison
- README.md: Project documentation

## Goals

- Pipeline + ColumnTransformer mastery
- Proper ML workflows: train/test split, CV, hyperparam tuning wo leakage
- Ensembles: bagging vs. boosting approaches and understanding when each excel

## Results

### Pipeline Notebook (`titanic-pipeline.ipynb`)
| Model | Test Accuracy | Best Parameters |
|-------|---------------|-----------------|
| Logistic Regression | ~78% | penalty='l2', C=0.1 |
| Random Forest | ~79% | n_estimators=100, max_depth=5 |

### Ensemble Comparison (`titanic_ensemble_cv.ipynb`)
| Model | F1 Score | ROC AUC | Best Parameters |
|-------|----------|---------|-----------------|
| **Random Forest** | **0.760** | **0.805** | max_depth=None, min_samples_split=5, n_estimators=100 |
| Gradient Boosting | 0.699 | 0.762 | learning_rate=0.1, max_depth=5, n_estimators=100 |

## Key Insights
- RF outperformed GB in this dataset
- RF won because small dataset size, clean features, and simple relationships; favors bagging over boosting
- Hyperparam: improved performance over default params
- Feature preprocessing: crucial for model performance

## Next Steps

- Feature engineering (FamilySize, Profession extracted from Name)
- Model interpretability (SHAP)
- Performance optimization (Bayesian hyperparam opt. (Optuna))
- Deployment
- A/B Testing