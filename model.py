import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import joblib

# Load data
df = pd.read_csv("data/Loan_default.csv")

# Drop Loan_ID if exists
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# X and y
X = df.drop("Default", axis=1)
y = df["Default"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
numeric_features = X.select_dtypes(include=['int64','float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# Different classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

results = {}

# Train all models
for name, clf in classifiers.items():
    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('classifier', clf)
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    results[name] = acc
    print(f"{name}: {acc}")

# Select best model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("\nBest Model:", best_model_name, "Accuracy:", best_accuracy)

best_model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', classifiers[best_model_name])
])

best_model.fit(X_train, y_train)

# Save model
joblib.dump(best_model, "models/best_model.pkl")

print("Model Saved Successfully!")

