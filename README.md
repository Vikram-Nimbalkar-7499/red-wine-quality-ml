🍷 Red Wine Quality Analysis & Machine Learning
An end-to-end Machine Learning project for analyzing and predicting Red Wine Quality using Python and Scikit-learn.
The project covers the complete Machine Learning workflow — from data cleaning and exploratory data analysis to statistical analysis, feature engineering, feature selection, dimensionality reduction, model training, evaluation, and deployment.
________________________________________
📌 Project Overview
Wine quality depends on several physicochemical properties such as:
•	Fixed Acidity
•	Volatile Acidity
•	Citric Acid
•	Residual Sugar
•	Chlorides
•	Free Sulfur Dioxide
•	Total Sulfur Dioxide
•	Density
•	pH
•	Sulphates
•	Alcohol
The objective of this project is to understand the relationships between these features and wine quality and build Machine Learning models capable of predicting wine quality.
The project follows an end-to-end ML pipeline:
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Statistical Analysis
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Feature Transformation
     ↓
Feature Scaling
     ↓
Feature Selection
     ↓
PCA
     ↓
Train/Test Split
     ↓
Machine Learning Models
     ↓
Model Evaluation
     ↓
Hyperparameter Tuning
     ↓
Model Saving
     ↓
Deployment
________________________________________
🎯 Objectives
The main objectives of this project are:
1.	Perform complete Exploratory Data Analysis.
2.	Understand the distribution of wine physicochemical properties.
3.	Identify relationships between features.
4.	Analyze correlations and statistical relationships.
5.	Detect and analyze outliers.
6.	Handle skewed features using transformations.
7.	Understand and apply feature scaling.
8.	Perform feature selection.
9.	Apply Principal Component Analysis (PCA).
10.	Train and compare multiple Machine Learning algorithms.
11.	Evaluate models using appropriate classification metrics.
12.	Perform cross-validation and hyperparameter tuning.
13.	Save the final trained model.
14.	Build a deployable Machine Learning application.
________________________________________
📊 Dataset
The dataset contains physicochemical measurements of red wine samples along with their quality score.
Features
Feature	Description
fixed acidity	Fixed/non-volatile acidity of wine
volatile acidity	Volatile acidity concentration
citric acid	Citric acid concentration
residual sugar	Amount of residual sugar
chlorides	Chloride concentration
free sulfur dioxide	Free sulfur dioxide concentration
total sulfur dioxide	Total sulfur dioxide concentration
density	Density of wine
pH	Acidity/basicity level
sulphates	Sulphate concentration
alcohol	Alcohol percentage
quality	Wine quality score
________________________________________
🔎 Exploratory Data Analysis
The project performs both univariate and multivariate analysis.
Univariate Analysis
For individual features:
•	Distribution analysis
•	Histogram
•	KDE
•	Mean
•	Median
•	Mode
•	Variance
•	Standard deviation
•	Skewness
•	Kurtosis
•	Box plots
•	Outlier analysis
Example:
sns.histplot(df["alcohol"], kde=True)
plt.show()
________________________________________
Bivariate Analysis
Relationships between two variables are analyzed using:
•	Scatter plots
•	Box plots
•	Violin plots
•	Grouped analysis
•	Correlation analysis
Example:
sns.scatterplot(
    data=df,
    x="alcohol",
    y="quality"
)

plt.show()
________________________________________
Multivariate Analysis
The project also analyzes interactions between multiple variables using:
•	Correlation heatmaps
•	Pair plots
•	Feature relationships
•	PCA visualization
Example:
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.show()
________________________________________
📈 Statistical Analysis
Statistical techniques are used to understand relationships within the dataset.
Techniques Covered
•	Covariance
•	Pearson Correlation
•	Spearman Correlation
•	ANOVA
•	Chi-Square Test
•	T-Test
These techniques help determine whether observed relationships between variables are statistically meaningful.
________________________________________
⚙️ Feature Engineering
Feature engineering is used to prepare the dataset for Machine Learning.
Categorical Encoding
Techniques studied:
•	Label Encoding
•	One-Hot Encoding
•	Ordinal Encoding
For example:
Bad  → 0
Good → 1
________________________________________
📐 Feature Scaling
Different numerical features can have very different ranges.
The project covers:
StandardScaler
Transforms features approximately to:
Mean = 0
Standard Deviation = 1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
MinMaxScaler
Transforms values to a fixed range, usually:
0 → 1
RobustScaler
Uses the median and IQR and is useful when features contain significant outliers.
________________________________________
🔄 Feature Transformation
Skewed numerical variables can negatively affect some Machine Learning algorithms.
The project covers:
•	Log Transformation
•	log1p()
•	Box-Cox Transformation
•	Yeo-Johnson Transformation
•	PowerTransformer
Example:
import numpy as np

df["chlorides_log"] = np.log1p(
    df["chlorides"]
)
For Power Transformation:
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(
    method="yeo-johnson"
)

X_transformed = pt.fit_transform(X)
________________________________________
🎯 Feature Selection
Feature selection reduces unnecessary or less useful features.
Techniques covered:
Correlation Analysis
Highly correlated features can be investigated for redundancy.
SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

selector = SelectKBest(
    score_func=f_classif,
    k=8
)

X_selected = selector.fit_transform(
    X,
    y
)
Random Forest Feature Importance
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

importance = model.feature_importances_
________________________________________
🧩 PCA — Principal Component Analysis
PCA is used for dimensionality reduction.
Instead of using all original features directly:
11 Original Features
        ↓
       PCA
        ↓
Principal Components
Example:
from sklearn.decomposition import PCA

pca = PCA(
    n_components=0.95
)

X_pca = pca.fit_transform(
    X_scaled
)
n_components=0.95 keeps enough principal components to explain approximately 95% of the variance.
The project also visualizes:
•	Explained variance
•	Cumulative explained variance
•	PC1 vs PC2
________________________________________
🤖 Machine Learning Models
Multiple Machine Learning algorithms can be trained and compared.
Classification Models
Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)
K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(X_train, y_train)
Decision Tree
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)
Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)
Support Vector Machine
from sklearn.svm import SVC

model = SVC(
    kernel="rbf"
)

model.fit(X_train, y_train)
________________________________________
📊 Model Evaluation
Models are evaluated using multiple metrics instead of relying only on accuracy.
Metrics include:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	Confusion Matrix
•	Cross Validation
Example:
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        predictions
    )
)
________________________________________
🔄 Cross Validation
To obtain a more reliable estimate of model performance, K-Fold Cross Validation can be used.
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print(scores)
print(scores.mean())
________________________________________
🔧 Hyperparameter Tuning
Hyperparameters can be optimized using:
•	GridSearchCV
•	RandomizedSearchCV
Example:
from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    param_grid=params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(grid.best_params_)
________________________________________
🛡️ Data Leakage Prevention
The project follows proper Machine Learning preprocessing practices.
Transformers such as:
•	StandardScaler
•	MinMaxScaler
•	RobustScaler
•	PCA
•	Feature Selection
should learn parameters from the training data only.
Correct approach:
Training Data
     ↓
fit()
     ↓
Learn parameters
     ↓
transform Training Data
     ↓
transform Test Data
The test set should not be used during preprocessing model fitting.
________________________________________
📁 Project Structure
red-wine-quality-ml/
│
├── data/
│   └── winequality-red.csv
│
├── notebooks/
│   └── red_wine_quality_analysis.ipynb
│
├── images/
│   ├── correlation_heatmap.png
│   ├── quality_distribution.png
│   ├── feature_distribution.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── models/
│   └── final_model.joblib
│
├── src/
│   └── preprocessing.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
________________________________________
🛠️ Technologies Used
Programming
•	Python
Data Analysis
•	NumPy
•	Pandas
Visualization
•	Matplotlib
•	Seaborn
Machine Learning
•	Scikit-learn
•	SciPy
Deployment
•	Streamlit
Development
•	Jupyter Notebook
•	Git
•	GitHub
________________________________________
🚀 Installation
Clone the repository:
git clone https://github.com/YOUR_USERNAME/red-wine-quality-ml.git
Move into the project:
cd red-wine-quality-ml
Create a virtual environment:
python -m venv venv
Activate it on Windows:
venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
________________________________________
▶️ Run the Notebook
Start Jupyter:
jupyter notebook
Then open:
notebooks/red_wine_quality_analysis.ipynb
________________________________________
🌐 Run the Streamlit Application
After creating app.py:
streamlit run app.py
The application provides an interface where users can enter wine physicochemical properties and receive a predicted wine quality classification.
________________________________________
📦 Requirements
Example requirements.txt:
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
jupyter
joblib
streamlit
________________________________________
📌 Key Learning Outcomes
Through this project, the following Machine Learning concepts are demonstrated:
•	Data preprocessing
•	Exploratory Data Analysis
•	Univariate Analysis
•	Bivariate Analysis
•	Multivariate Analysis
•	Statistical testing
•	Correlation analysis
•	Outlier detection
•	Categorical encoding
•	Feature transformation
•	Feature scaling
•	Feature selection
•	Dimensionality reduction
•	PCA
•	Classification
•	Model comparison
•	Cross-validation
•	Hyperparameter tuning
•	Model evaluation
•	Model persistence
•	Machine Learning deployment
________________________________________
💡 Future Improvements
Potential improvements include:
•	XGBoost
•	LightGBM
•	Hyperparameter optimization with Optuna
•	SHAP-based model explainability
•	MLflow experiment tracking
•	Docker deployment
•	REST API using FastAPI
•	Cloud deployment
•	Automated CI/CD
•	Real-time prediction API
________________________________________
👨‍💻 Author
Vikram Nimbalkar
B.Tech — Artificial Intelligence & Data Science
Interested in Machine Learning, AI Engineering, Backend Development, and building real-world AI systems.
________________________________________
⭐ Project Highlights
This project demonstrates an end-to-end Machine Learning workflow rather than only training a single model.
EDA
 ↓
Statistics
 ↓
Feature Engineering
 ↓
Transformation
 ↓
Scaling
 ↓
Feature Selection
 ↓
PCA
 ↓
ML Models
 ↓
Evaluation
 ↓
Tuning
 ↓
Deployment
If you found this project useful, consider giving the repository a ⭐.

