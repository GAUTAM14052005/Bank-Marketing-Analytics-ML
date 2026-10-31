
# ============================================================
# PROJECT: BANK MARKETING SUBSCRIPTION PREDICTION
# COURSE: INT234 - PREDICTIVE ANALYTICS
# DATA SOURCE: UCI MACHINE LEARNING REPOSITORY
# ============================================================

# ============================================================
# PART 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

print("=" * 65)
print("BANK MARKETING SUBSCRIPTION PREDICTION")
print("=" * 65)

# ============================================================
# PART 2: LOAD DATASET
# ============================================================

# CHANGE THIS PATH IF YOUR CSV IS IN ANOTHER FOLDER

file_path = r"C:\Users\ahuja\Downloads\bank+marketing\bank\bank.csv"

df = pd.read_csv(file_path, sep=";")

print("\nDATASET LOADED SUCCESSFULLY!")

print("\nDataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nLast Five Rows:")
print(df.tail())

print("\nDataset Information:")
df.info()

print("\nStatistical Description:")
print(df.describe(include="all").T)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nUnique Values:")
print(df.nunique())

print("\nTarget Distribution:")
print(df["y"].value_counts())

print("\nTarget Percentage:")
print(df["y"].value_counts(normalize=True) * 100)

# ============================================================
# PART 3: DATA CLEANING
# ============================================================

# Remove exact duplicate records

print("\nDuplicate Records:", df.duplicated().sum())

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Convert unknown values into missing values

feature_columns = [c for c in df.columns if c != "y"]

df[feature_columns] = df[feature_columns].replace(
    ["unknown", "?", ""], np.nan
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sort_values(ascending=False))

print("\nTotal Missing Values:",
      df[feature_columns].isnull().sum().sum())

# Identify numerical and categorical features

numeric_cols = df[feature_columns].select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df[feature_columns].select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumerical Variables:")
print(numeric_cols)

print("\nCategorical Variables:")
print(categorical_cols)

print("\nNumber of Numerical Variables:", len(numeric_cols))
print("Number of Categorical Variables:", len(categorical_cols))

# IMPORTANT:
# Duration is excluded because it is only known after a call.
# This prevents data leakage when predicting before contacting
# a customer.

# ============================================================
# PART 4: EXPLORATORY DATA ANALYSIS
# TEN MEANINGFUL VISUALIZATIONS
# ============================================================

# VISUALIZATION 1: TARGET VARIABLE DISTRIBUTION

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="y")

plt.title("1. Term Deposit Subscription Distribution")
plt.xlabel("Subscription (Yes / No)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# VISUALIZATION 2: AGE DISTRIBUTION

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("2. Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# VISUALIZATION 3: ACCOUNT BALANCE DISTRIBUTION

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="balance",
    bins=50,
    kde=True
)

plt.title("3. Customer Account Balance Distribution")
plt.xlabel("Account Balance")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# VISUALIZATION 4: NUMERICAL FEATURE BOXPLOTS

df[numeric_cols].plot(
    kind="box",
    subplots=True,
    layout=(-1, 3),
    figsize=(15, 12),
    sharex=False,
    sharey=False
)

plt.suptitle("4. Numerical Feature Boxplots", y=1.02)

plt.tight_layout()
plt.show()

# VISUALIZATION 5: MISSING VALUE ANALYSIS

missing_values = df[feature_columns].isnull().sum()
missing_values = missing_values[missing_values > 0]

plt.figure(figsize=(10, 5))

if len(missing_values) > 0:

    sns.barplot(
        x=missing_values.index,
        y=missing_values.values
    )

    plt.xticks(rotation=40, ha="right")

else:

    plt.text(
        0.5, 0.5,
        "No Missing Values Found",
        ha="center",
        va="center"
    )

plt.title("5. Missing Values in Dataset")
plt.xlabel("Features")
plt.ylabel("Missing Value Count")

plt.tight_layout()
plt.show()

# VISUALIZATION 6: CORRELATION HEATMAP

plt.figure(figsize=(11, 8))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("6. Numerical Feature Correlation Heatmap")

plt.tight_layout()
plt.show()

# Helper function for subscription rate charts

def plot_subscription_rate(column, title, order=None):

    rate_data = (
        df.groupby(column, dropna=False)["y"]
        .apply(lambda x: (x == "yes").mean() * 100)
        .reset_index(name="Subscription Rate")
    )

    if order is not None:
        rate_data[column] = pd.Categorical(
            rate_data[column],
            categories=order,
            ordered=True
        )

        rate_data = rate_data.sort_values(column)

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=rate_data,
        x=column,
        y="Subscription Rate"
    )

    plt.title(title)
    plt.xlabel(column.capitalize())
    plt.ylabel("Subscription Rate (%)")

    plt.xticks(rotation=35, ha="right")

    plt.tight_layout()
    plt.show()

# VISUALIZATION 7: SUBSCRIPTION BY JOB

plot_subscription_rate(
    "job",
    "7. Subscription Rate by Job Type"
)

# VISUALIZATION 8: SUBSCRIPTION BY MONTH

month_order = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]

plot_subscription_rate(
    "month",
    "8. Subscription Rate by Month",
    month_order
)

# VISUALIZATION 9: CAMPAIGN CONTACTS VS OUTCOME

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="y",
    y="campaign",
    showfliers=False
)

plt.title("9. Campaign Contacts vs Subscription Outcome")
plt.xlabel("Subscription")
plt.ylabel("Number of Contacts")

plt.tight_layout()
plt.show()

# VISUALIZATION 10: PREVIOUS CAMPAIGN OUTCOME

plot_subscription_rate(
    "poutcome",
    "10. Subscription Rate by Previous Campaign Outcome"
)

# ============================================================
# PART 5: DATA PREPROCESSING
# ============================================================

# Separate independent and dependent variables

X = df.drop(columns=["y", "duration"])

y = df["y"].copy()

print("\nIndependent Variables Shape:", X.shape)
print("Target Variable Shape:", y.shape)
numeric_cols=X.select_dtypes(include=["number"]
                             ).columns.tolist()
categorical_cols=X.select_dtypes(include=["object","category"]
                                 ).columns.tolist()
print("\nNUMERICAL COLUMNS:")
print(numeric_cols)

print("\nCATEGORICAL COLUMNS:")
print(categorical_cols)

# Check that the columns are separated correctly

print("\nChecking Data Types:")

for col in numeric_cols:
    print(col, X[col].dtype)

for col in categorical_cols:
    print(col, X[col].dtype)

# Numerical preprocessing

numeric_transformer = Pipeline(steps=[

    ("imputer", SimpleImputer(strategy="median")),

    ("scaler", StandardScaler())

])

# Categorical preprocessing

categorical_transformer = Pipeline(steps=[

    ("imputer", SimpleImputer(strategy="most_frequent")),

    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))

])

# Combine both preprocessing pipelines

preprocessor = ColumnTransformer(

    transformers=[

        ("numeric", numeric_transformer, numeric_cols),

        ("categorical", categorical_transformer, categorical_cols)

    ],

    remainder="drop"

)

print("\nDATA PREPROCESSING COMPLETED SUCCESSFULLY!")


print("\nPREPROCESSING PIPELINE CREATED SUCCESSFULLY!")

# ============================================================
# PART 6: TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])

print("\nTrain-Test Split Completed!")

# ============================================================
# PART 7: DEFINE FIVE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

    "KNN":

        KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ),

    "Gaussian Naive Bayes":

        GaussianNB(),

    "Decision Tree":

        DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),

    "Linear SVM":

        LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        )

}

# ============================================================
# PART 8: TRAIN AND EVALUATE ALL MODELS
# ============================================================

trained_models = {}
predictions = {}
model_scores = {}

results = []

for name, classifier in models.items():

    print("\n" + "=" * 65)

    print("TRAINING MODEL:", name)

    print("=" * 65)

    # Create complete ML pipeline

    pipeline = Pipeline(steps=[

        ("preprocessor", preprocessor),

        ("classifier", classifier)

    ])

    # Train model

    pipeline.fit(X_train, y_train)

    # Predict output

    y_pred = pipeline.predict(X_test)

    # Obtain prediction scores for ROC-AUC

    if hasattr(pipeline, "predict_proba"):

        y_score = pipeline.predict_proba(X_test)[:, 1]

    else:

        y_score = pipeline.decision_function(X_test)

    # Calculate evaluation metrics

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        pos_label="yes",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        pos_label="yes",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        pos_label="yes",
        zero_division=0
    )

    auc = roc_auc_score(
        (y_test == "yes").astype(int),
        y_score
    )

    avg_precision = average_precision_score(
        (y_test == "yes").astype(int),
        y_score
    )

    # Store results

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": auc,

        "Average Precision": avg_precision

    })

    trained_models[name] = pipeline

    predictions[name] = y_pred

    model_scores[name] = y_score

    print("\nCLASSIFICATION REPORT")

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    print("\nConfusion Matrix:")

    print(confusion_matrix(
        y_test,
        y_pred,
        labels=["no", "yes"]
    ))

# ============================================================
# PART 9: MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df.set_index("Model", inplace=True)

print("\n" + "=" * 75)

print("FINAL COMPARISON OF FIVE MACHINE LEARNING MODELS")

print("=" * 75)

print(results_df.round(4))

# Save comparison results

results_df.to_csv("model_comparison_results.csv")

# Find best model using F1-score

best_model = results_df["F1 Score"].idxmax()

print("\nBEST MODEL BASED ON F1 SCORE:", best_model)

print(
    "BEST F1 SCORE:",
    round(results_df.loc[best_model, "F1 Score"], 4)
)

# ============================================================
# PART 10: MODEL PERFORMANCE VISUALIZATIONS
# ============================================================

# VISUALIZATION 11: MODEL COMPARISON

plt.figure(figsize=(12, 6))

results_df[
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
].plot(kind="bar", figsize=(12, 6))

plt.title("Comparison of Five Machine Learning Models")

plt.ylabel("Performance Score")

plt.xlabel("ML Algorithms")

plt.ylim(0, 1)

plt.xticks(rotation=25, ha="right")

plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# VISUALIZATION 12: CONFUSION MATRICES

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes = axes.flatten()

for i, name in enumerate(models):

    cm = confusion_matrix(
        y_test,
        predictions[name],
        labels=["no", "yes"]
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=["No", "Yes"],

        yticklabels=["No", "Yes"],

        ax=axes[i]

    )

    axes[i].set_title(name)

    axes[i].set_xlabel("Predicted")

    axes[i].set_ylabel("Actual")

axes[-1].axis("off")

plt.suptitle("Confusion Matrix Comparison", fontsize=15)

plt.tight_layout()
plt.show()

# VISUALIZATION 13: ROC CURVES

plt.figure(figsize=(10, 7))

y_test_binary = (y_test == "yes").astype(int)

for name in models:

    fpr, tpr, thresholds = roc_curve(

        y_test_binary,

        model_scores[name]

    )

    auc = results_df.loc[name, "ROC-AUC"]

    plt.plot(

        fpr,

        tpr,

        label=f"{name} (AUC={auc:.3f})"

    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()
plt.show()

# VISUALIZATION 14: PRECISION-RECALL CURVES

plt.figure(figsize=(10, 7))

for name in models:

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_test_binary,
            model_scores[name]
        )
    )

    ap = results_df.loc[name, "Average Precision"]

    plt.plot(

        recall_values,

        precision_values,

        label=f"{name} (AP={ap:.3f})"

    )

plt.title("Precision-Recall Curve Comparison")

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.legend()

plt.tight_layout()
plt.show()

# VISUALIZATION 15: DECISION TREE FEATURE IMPORTANCE

tree_pipeline = trained_models["Decision Tree"]

feature_names = tree_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = tree_pipeline.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(15)

plt.figure(figsize=(10, 7))

sns.barplot(

    data=importance_df,

    x="Importance",

    y="Feature"

)

plt.title("Top 15 Important Features - Decision Tree")

plt.xlabel("Feature Importance")

plt.ylabel("Features")

plt.tight_layout()
plt.show()

# ============================================================
# PART 11: SAVE FINAL RESULTS
# ============================================================

# Save individual model predictions

prediction_df = pd.DataFrame({

    "Actual": y_test.to_numpy()

})

for name in models:

    prediction_df[name] = predictions[name]

prediction_df.to_csv(
    "model_predictions.csv",
    index=False
)

# Save feature importance

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

# ============================================================
# PART 12: FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("PROJECT COMPLETED SUCCESSFULLY!")

print("=" * 70)

print("\nDataset Records:", len(df))

print("Independent Features:", X.shape[1])

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))

print("Machine Learning Models:", len(models))

print("\nFINAL MODEL PERFORMANCE:")

print(results_df.round(4))

print("\nBEST PERFORMING MODEL:", best_model)

print(
    "Best F1 Score:",
    round(results_df.loc[best_model, "F1 Score"], 4)
)

print("\nOUTPUT FILES CREATED:")

print("1. model_comparison_results.csv")

print("2. model_predictions.csv")

print("3. feature_importance.csv")

print("\nALL PROJECT TASKS COMPLETED!")

print("=" * 70)
