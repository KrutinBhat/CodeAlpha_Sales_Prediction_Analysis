# 📊 Sales Prediction & Advertising Impact Analysis

> An end-to-end Data Science and Machine Learning project for predicting sales from advertising expenditure, analyzing marketing-channel effectiveness, and generating data-driven insights for marketing decision-making.

---

## 📌 Project Overview

This project implements an end-to-end **Sales Prediction and Advertising Impact Analysis** workflow using Python, statistical analysis, exploratory data analysis, machine learning, feature engineering, model evaluation, and interactive visualization.

The primary objective is to understand how different advertising channels influence sales and to develop regression-based machine learning models capable of predicting sales from marketing expenditure.

The project goes beyond a basic regression implementation by incorporating:

- Comprehensive data quality analysis
- Exploratory Data Analysis (EDA)
- Statistical relationship analysis
- Feature engineering
- Feature interaction analysis
- Non-linear feature construction
- Multiple regression algorithms
- Regularized regression
- Ensemble machine learning
- Hyperparameter optimization
- Model comparison
- Cross-validation
- Prediction analysis
- Advertising sensitivity analysis
- Marketing budget scenario analysis
- Interactive Plotly visualizations
- Excel-based analytical reporting
- Model serialization using Joblib
- A modular Python source-code structure

The project is designed to demonstrate a complete **data-to-insight-to-prediction workflow** rather than treating machine learning as an isolated modeling exercise.

---

# 🎯 Business Problem

Marketing teams invest budgets across multiple advertising channels, but determining which channels contribute most effectively to sales can be challenging.

The central business questions addressed in this project are:

1. How strongly does advertising expenditure relate to sales?
2. Which advertising channel has the strongest relationship with sales?
3. Can sales be predicted from advertising expenditure?
4. Which regression model provides the best predictive performance?
5. Do interactions between advertising channels provide additional predictive value?
6. How does changing advertising expenditure affect predicted sales?
7. Can machine learning be used to simulate different marketing-budget scenarios?
8. How can the analysis support better marketing allocation decisions?

---

# 🧠 Project Objectives

The project focuses on five major objectives.

### 1. Data Preparation

- Load and inspect the dataset
- Identify data types
- Check missing values
- Detect duplicate observations
- Validate dataset quality
- Standardize column names
- Prepare the data for analysis

### 2. Exploratory Data Analysis

Investigate:

- Distribution of advertising expenditure
- Distribution of sales
- Relationships between advertising channels and sales
- Correlation structure
- Potential outliers
- Channel-level patterns
- Advertising-spend relationships

### 3. Feature Engineering

Construct additional features to capture:

- Total advertising expenditure
- Advertising-channel shares
- Pairwise channel interactions
- Non-linear relationships
- Squared advertising variables

These transformations allow the models to capture relationships that may not be fully represented by simple linear effects.

### 4. Predictive Modeling

Evaluate multiple regression approaches, including:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Random Forest Regression
- Gradient Boosting Regression
- Histogram-based Gradient Boosting Regression

Model performance is evaluated using appropriate regression metrics rather than relying on a single score.

### 5. Marketing Intelligence

Translate model outputs into practical business insights through:

- Advertising sensitivity analysis
- Channel comparison
- Budget scenario simulation
- Predicted sales analysis
- Interactive visualizations
- Excel-based reporting

---

# 📂 Dataset

The project uses an advertising-sales dataset containing observations of advertising expenditure and corresponding sales outcomes.

The primary variables used in the analysis are:

| Feature | Description |
|---|---|
| `TV` | Advertising expenditure through TV |
| `Radio` | Advertising expenditure through radio |
| `Newspaper` | Advertising expenditure through newspaper |
| `Sales` | Target variable representing sales |

The dataset is used to investigate the relationship between advertising investment and sales performance.

> **Dataset attribution:** The dataset used in this project was obtained from the publicly available Kaggle dataset associated with the CodeAlpha Sales Prediction task.

---

# 🗂️ Repository Structure

```text
CodeAlpha_Sales_Prediction_Analysis/
│
├── data/
│   ├── raw/
│   │   └── Advertising.csv
│   │
│   └── processed/
│
├── models/
│   └── sales_prediction_model.joblib
│
├── notebooks/
│   └── Sales_Price_Prediction.ipynb
│
├── reports/
│   ├── excel/
│   │   └── Sales_Prediction_Analysis.xlsx
│   │
│   └── interactive/
│       ├── advertising_sensitivity_analysis.html
│       ├── interactive_3d_advertising_sales.html
│       ├── interactive_tv_sales_analysis.html
│       └── marketing_budget_scenarios.html
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── prediction.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

# 🔬 Methodology

The project follows a structured machine-learning lifecycle.

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Validation
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Feature / Target Separation
     │
     ▼
Train-Test Split
     │
     ▼
Multiple Regression Models
     │
     ▼
Cross-Validation / Hyperparameter Optimization
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
Prediction
     │
     ▼
Advertising Sensitivity Analysis
     │
     ▼
Marketing Budget Scenarios
     │
     ▼
Business Insights
```

---

# 🧹 1. Data Preparation

The dataset is first loaded using Pandas and validated before modeling.

The preprocessing workflow includes:

- Dataset shape inspection
- Column inspection
- Data-type verification
- Missing-value analysis
- Duplicate detection
- Column-name standardization
- Data consistency checks

Reusable data-processing utilities are implemented inside:

```text
src/data_processing.py
```

This separates data-processing logic from exploratory notebook code.

---

# 📊 2. Exploratory Data Analysis

Exploratory Data Analysis is used to understand the structure and statistical characteristics of the dataset before applying machine learning.

The analysis includes:

### Univariate Analysis

Distribution analysis of:

- TV advertising expenditure
- Radio advertising expenditure
- Newspaper advertising expenditure
- Sales

### Bivariate Analysis

Relationships between:

- TV and Sales
- Radio and Sales
- Newspaper and Sales

### Multivariate Analysis

The project also investigates relationships among multiple advertising channels and their combined relationship with sales.

### Correlation Analysis

Correlation analysis is used to identify the strength and direction of relationships between numerical variables.

---

# 🧮 3. Feature Engineering

A key component of the project is feature engineering.

In addition to the original advertising variables, the project creates additional analytical features.

### Total Advertising Spend

```text
Total Ad Spend =
TV + Radio + Newspaper
```

This provides an overall measure of marketing expenditure.

### Channel Share

The proportion of the advertising budget allocated to each channel is calculated.

```text
TV Share
Radio Share
Newspaper Share
```

### Interaction Features

The project investigates potential combined effects using:

```text
TV × Radio
TV × Newspaper
Radio × Newspaper
```

These features allow the model to represent situations where the effect of one advertising channel may depend on another channel.

### Non-Linear Features

Squared terms are also constructed:

```text
TV²
Radio²
Newspaper²
```

These allow regression models to represent potential non-linear relationships.

Feature-engineering utilities are implemented in:

```text
src/feature_engineering.py
```

---

# 🤖 4. Machine Learning Models

Multiple regression approaches are evaluated rather than relying on a single algorithm.

## Linear Regression

Provides a baseline model and interpretable linear relationship between advertising expenditure and sales.

---

## Ridge Regression

Ridge introduces L2 regularization to reduce the impact of potentially correlated predictors and improve model stability.

---

## Lasso Regression

Lasso introduces L1 regularization and can shrink less useful coefficients toward zero.

This can also provide an additional perspective on feature importance.

---

## Elastic Net Regression

Elastic Net combines L1 and L2 regularization.

It provides a compromise between the feature-selection behavior of Lasso and the coefficient stabilization of Ridge.

---

## Random Forest Regression

Random Forest is an ensemble learning approach capable of modeling non-linear relationships and feature interactions.

---

## Gradient Boosting Regression

Gradient Boosting builds an ensemble of weak learners sequentially, allowing the model to capture complex relationships between advertising variables and sales.

---

## Histogram-Based Gradient Boosting

Histogram-based gradient boosting provides another powerful ensemble approach for regression and is included as an additional model benchmark.

---

# ⚙️ 5. Model Training & Evaluation

The dataset is divided into training and testing subsets.

The project evaluates models using multiple regression metrics.

### Mean Absolute Error — MAE

MAE measures the average absolute difference between predicted and actual sales.

Lower values indicate better predictive performance.

---

### Root Mean Squared Error — RMSE

RMSE penalizes larger prediction errors more strongly than MAE.

Lower RMSE indicates better predictive performance.

---

### R² Score

R² measures the proportion of variance in the target variable explained by the model.

Higher values indicate better explanatory performance.

---

# 📈 6. Model Comparison

Rather than selecting a model based on a single metric, the project compares multiple models using a common evaluation framework.

The comparison considers:

- MAE
- RMSE
- R²
- Generalization performance
- Model complexity
- Interpretability
- Non-linear modeling capability

The model-training functionality is implemented in:

```text
src/model_training.py
```

---

# 🔧 7. Hyperparameter Optimization

The project also incorporates hyperparameter tuning for the Gradient Boosting model.

The optimization process evaluates combinations of parameters such as:

- Number of estimators
- Learning rate
- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples per leaf

Cross-validation is used during the search to reduce dependence on a single train-test split.

This provides a more systematic approach to selecting model parameters.

---

# 💾 8. Model Serialization

The trained model is serialized using Joblib.

The saved artifact is:

```text
models/sales_prediction_model.joblib
```

This allows the trained model to be loaded later without retraining it from scratch.

Reusable model persistence and prediction utilities are implemented in:

```text
src/prediction.py
```

---

# 🔮 9. Sales Prediction

The trained model can generate sales predictions based on advertising expenditure.

Conceptually:

```text
TV Advertising
       +
Radio Advertising
       +
Newspaper Advertising
       │
       ▼
Machine Learning Model
       │
       ▼
Predicted Sales
```

The repository includes reusable prediction functions that can load the serialized model and generate predictions.

---

# 📢 10. Advertising Impact Analysis

A major objective of the project is understanding how advertising expenditure affects sales.

The analysis examines:

- Advertising-sales relationships
- Channel-level effects
- Relative advertising contribution
- Combined advertising effects
- Sensitivity of predicted sales to marketing expenditure

This transforms the project from a simple prediction exercise into a marketing analytics problem.

---

# 💰 11. Marketing Budget Scenario Analysis

The project includes scenario-based analysis to investigate how different advertising allocations can influence predicted sales.

Example concept:

```text
Marketing Budget Scenario
          │
          ├── TV allocation
          ├── Radio allocation
          └── Newspaper allocation
                    │
                    ▼
              ML Prediction
                    │
                    ▼
             Expected Sales
```

This allows different hypothetical marketing strategies to be compared using model predictions.

The objective is not to claim that a model can guarantee future sales, but to provide a quantitative framework for evaluating potential advertising scenarios.

---

# 📊 12. Interactive Visual Analytics

Plotly is used to create interactive visualizations for deeper exploration.

The repository includes interactive analytical outputs covering areas such as:

### Advertising Sensitivity Analysis

```text
reports/interactive/
└── advertising_sensitivity_analysis.html
```

### 3D Advertising-Sales Relationship

```text
reports/interactive/
└── interactive_3d_advertising_sales.html
```

### TV Advertising Analysis

```text
reports/interactive/
└── interactive_tv_sales_analysis.html
```

### Marketing Budget Scenarios

```text
reports/interactive/
└── marketing_budget_scenarios.html
```

These HTML reports allow users to interact with the analytical results without requiring a Python environment.

---

# 📑 13. Excel Reporting

An Excel-based analytical report is also included:

```text
reports/excel/
└── Sales_Prediction_Analysis.xlsx
```

Excel provides an additional business-oriented representation of the analysis and makes the results easier to inspect outside the Python environment.

The combination of:

- Python analysis
- Machine learning
- Interactive visualization
- Excel reporting

provides both technical and business-facing outputs.

---

# 📓 14. Jupyter Notebook

The complete analytical workflow is documented in:

```text
notebooks/Sales_Price_Prediction.ipynb
```

The notebook contains the end-to-end analysis, including:

1. Library imports
2. Dataset loading
3. Data inspection
4. Data quality analysis
5. Data cleaning
6. Exploratory data analysis
7. Statistical analysis
8. Feature engineering
9. Feature selection
10. Train-test splitting
11. Regression modeling
12. Model comparison
13. Model evaluation
14. Hyperparameter tuning
15. Prediction analysis
16. Advertising impact analysis
17. Scenario analysis
18. Interactive visualization
19. Business insights

---

# 🧩 15. Modular Python Architecture

The project separates reusable functionality from exploratory analysis.

### `data_processing.py`

Responsible for:

- Loading data
- Column-name standardization
- Dataset validation
- Duplicate handling
- Missing-value handling

### `feature_engineering.py`

Responsible for:

- Advertising feature creation
- Interaction features
- Non-linear features
- Feature-target separation

### `model_training.py`

Responsible for:

- Train-test splitting
- Regression model construction
- Model evaluation
- Model comparison
- Hyperparameter tuning

### `prediction.py`

Responsible for:

- Model serialization
- Model loading
- Prediction generation

This structure makes the repository easier to maintain and extend.

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib | Statistical and analytical visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and model evaluation |
| Plotly | Interactive visualization |
| Joblib | Model serialization |
| OpenPyXL | Excel report generation |
| Jupyter Notebook | Exploratory analysis and documentation |
| VS Code | Development and repository management |
| Git / GitHub | Version control and project sharing |

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone <https://github.com/KrutinBhat/CodeAlpha_Sales_Prediction_Analysis>
```

Navigate into the project:

```bash
cd CodeAlpha_Sales_Prediction_Analysis
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Run the Jupyter Notebook

From the project root:

```bash
jupyter notebook
```

Open:

```text
notebooks/Sales_Price_Prediction.ipynb
```

Run the notebook cells sequentially.

---

# 🧪 Testing the Source Modules

The project source modules can be tested from the project root.

For example:

```bash
python -c "from src.data_processing import load_data; print('Source modules are available.')"
```

The modular source layer is designed to support future expansion into a larger production-oriented application.

---

# 📊 Key Analytical Areas

The project focuses on three major analytical dimensions.

## 1. Predictive Analytics

Use historical advertising expenditure to predict sales.

## 2. Marketing Analytics

Understand relationships between marketing expenditure and sales outcomes.

## 3. Scenario Analysis

Use the trained model to estimate how changes in advertising allocations may influence predicted sales.

Together, these components provide a more complete perspective than a simple regression model.

---

# 💡 Business Value

The analytical workflow can support marketing decision-making by providing a quantitative framework for evaluating advertising expenditure.

Potential applications include:

- Comparing advertising channels
- Evaluating marketing expenditure patterns
- Simulating advertising-budget scenarios
- Estimating expected sales under hypothetical allocations
- Identifying potentially influential advertising variables
- Supporting data-driven marketing planning

The predictions should be interpreted as **model-based estimates rather than guaranteed business outcomes**.

---

# ⚠️ Limitations

Despite the use of multiple advanced modeling techniques, the project has several limitations.

### Dataset Size

The dataset is relatively small compared with production-scale marketing datasets.

### Limited Variables

Sales may depend on many external factors that are not represented in the available dataset.

Examples could include:

- Seasonality
- Pricing
- Competitor activity
- Promotions
- Economic conditions
- Brand awareness
- Distribution
- Customer demographics

### Observational Relationships

Correlation and predictive relationships should not automatically be interpreted as causal relationships.

### Scenario Analysis

Marketing-budget simulations depend on the assumptions and behavior learned by the trained model.

Therefore, scenario results should be treated as analytical estimates rather than guaranteed outcomes.

---

# 🚀 Future Improvements

The project can be extended in several directions.

### Advanced Machine Learning

Potential future models include:

- XGBoost
- LightGBM
- CatBoost
- Explainable Boosting Machines

### Explainable AI

Future versions could incorporate:

- SHAP
- Partial Dependence Plots
- Permutation Importance
- Individual Conditional Expectation

to improve model interpretability.

### Automated ML Pipeline

The workflow could be converted into a fully automated pipeline covering:

```text
Data Ingestion
      ↓
Validation
      ↓
Preprocessing