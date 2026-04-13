# Employee Well-Being & Work-Life Balance Prediction

A People Analytics project that explores the factors affecting employee
well-being and builds a regression model to predict Work-Life Balance scores
from behavioral and demographic data.

## Dataset
The dataset (`employee_wellbeing.csv`) contains records of airline employees
(flight attendants and check-in agents) with the following feature groups:

- **Demographics:** Age, Gender, Marital Status, Employment Role, Salary
- **Lifestyle:** Daily Steps, Sleep Hours, Healthy Diet, Time for Hobby
- **Work Behavior:** Daily Stress, To-Do Completed, Lost Vacation, Flow
- **Social:** Core Circle, Social Network, Supporting Others
- **Target:** `WORK_LIFE_BALANCE_SCORE`

## Project Structure

├── wee_being_analysis.py # Exploratory Data Analysis & visualizations
└── wellBeing_MLPrediction-2.py # Data preprocessing & ML prediction

## Methodology

### 1. Exploratory Data Analysis
- Visualized daily stress distribution across age groups and job roles
- Analyzed time spent on hobbies by gender
- Generated a full correlation heatmap across 20 features to identify
  key relationships with the Work-Life Balance Score

### 2. Preprocessing
- Mode imputation for missing categorical and numerical values
- Label encoding for categorical features (Gender, Status,
  Employment, Salary)

### 3. Model
- **Algorithm:** Linear Regression (scikit-learn)
- **Split:** 70% train / 30% test
- **Evaluation:** R² Score + Actual vs. Predicted scatter plot

## How to Run

```bash
pip install pandas numpy scikit-learn seaborn matplotlib

# Run EDA
python wee_being_analysis.py

# Run ML prediction
python wellBeing_MLPrediction-2.py
```

## Tech Stack
Python · pandas · scikit-learn · seaborn · matplotlib
