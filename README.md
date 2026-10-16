# Texas State Government Salary Analysis & Prediction

Analysis of Texas state government employee salaries, with machine learning models that predict annual salary.

## Objectives
- Detect salary outliers
- Analyze wage disparity by gender and ethnicity
- Explore salary trends over time
- Build and compare models that predict annual salary

## Dataset
Texas state government employee salary records (`salary.csv`), provided as part of my AI/ML specialization training. The file is not included in this repo because of its size.

Key columns include agency, class title, gender, ethnicity, employ date, and monthly and annual salary.

## Approach
1. **Data cleaning:** dropped rows with missing salary, converted employ dates to datetime, handled missing values and duplicates
2. **Exploratory data analysis:** salary distribution, boxplots, correlation heatmap, top agencies and top-paying roles
3. **Outlier detection:** IQR method on annual salary
4. **Pay gap analysis:** annual salary by gender and ethnicity
5. **Modeling:** one-hot encoded agency, class title, gender and ethnicity, then trained on an 80/20 train-test split
6. **Model comparison:** Linear Regression, Random Forest and XGBoost, compared on R² and MSE

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Jupyter Notebook

## How to run
1. Place `BHAVANA_R_TEXAS_PROJECT.ipynb` and your copy of `salary.csv` in the same folder
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`
3. Open the notebook in Jupyter or Google Colab and run all cells
