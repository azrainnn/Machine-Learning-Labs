# %% [markdown]
# # Lab 2 of 8 — Supervised learning I: regression, classification, and the workflow
#
# *Two regression examples, two classification examples, and a saved model at the end.*
#
# **Goal.** Fit regression and classification pipelines, validate chronologically where needed, tune thresholds and hyperparameters, test once, and save models.
#
# The notebook finds `all_datasets/` by searching its own folder and the folders above it; saved models go to a `models/` folder beside `all_datasets/`.

# %% [markdown]
# ## Setup

# %%
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_data_dir(name="all_datasets", marker="titanic_dataset.csv"):
    """Search the script/working folder and their parents for a data folder holding `marker`."""
    starts = [Path.cwd()]
    try:
        starts.insert(0, Path(__file__).resolve().parent)  # not defined inside a notebook
    except NameError:
        pass
    for start in starts:
        for folder in [start, *start.parents]:
            if (folder / name / marker).is_file():
                return folder / name
    searched = ", ".join(str(s) for s in starts)
    raise FileNotFoundError(f"Could not find {name}/{marker} in or above: {searched}")


# The lab's `Path.cwd().parent` assumes notebooks sit directly in all_experiments/; searching upward
# works from any depth.
DATA = find_data_dir()
PROJECT_ROOT = DATA.parent
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(exist_ok=True)
print("Using data from:", DATA)
print("Saving models to:", MODELS)

# %% [markdown]
# One helper for numerical imputation/scaling and categorical imputation/one-hot encoding, reused throughout.

# %%
def make_preprocess(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

# %% [markdown]
# ## Part A — Regression: ordinary least squares versus ridge

# %% [markdown]
# Predict insurance charges. Compare OLS and ridge on the same features and split.

# %%
insurance = pd.read_csv(DATA / "medical_cost_personal_dataset.csv")
print(insurance.head())

X = insurance.drop(columns="charges")
y = insurance["charges"]
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

# %% [markdown]
# MAE gives average dollar error; RMSE emphasizes extremes; R² compares squared error with the mean-label reference.

# %%
fitted = {}
predictions = {}
for name, estimator in {
    "linear regression": LinearRegression(),
    "ridge": Ridge(alpha=10.0),
}.items():
    model = Pipeline([("preprocess", make_preprocess(num_cols, cat_cols)), ("model", estimator)])
    model.fit(X_train, y_train)
    fitted[name] = model
    predictions[name] = model.predict(X_valid)
    pred = predictions[name]
    print(f"\n{name}")
    print("MAE:", round(mean_absolute_error(y_valid, pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_valid, pred)), 2))
    print("R2:", round(r2_score(y_valid, pred), 3))

# %% [markdown]
# Coefficients give conditional prediction changes. Standardized numerical coefficients represent one-standard-deviation changes.

# %%
linear = fitted["linear regression"]
names = linear.named_steps["preprocess"].get_feature_names_out()
coefs = pd.Series(linear.named_steps["model"].coef_, index=names).sort_values()
plt.figure(figsize=(6, 3.6))
plt.barh(coefs.index, coefs.values)
plt.axvline(0, color="grey", linewidth=0.8)
plt.xlabel("Change in predicted charges (dollars)")
plt.title("What the linear model learned")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Inspect.** The smoker coefficient exceeds $20,000; age and BMI contribute more than region. These are model associations, not causal effects.

# %% [markdown]
# Plot ridge residuals; structure suggests missed patterns.

# %%
ridge_pred = predictions["ridge"]
residual = y_valid.to_numpy() - ridge_pred
is_smoker = (X_valid["smoker"] == "yes").to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
axes[0].scatter(y_valid[~is_smoker], ridge_pred[~is_smoker], alpha=0.5, label="non-smoker")
axes[0].scatter(y_valid[is_smoker], ridge_pred[is_smoker], alpha=0.5, label="smoker")
axes[0].plot([0, y_valid.max()], [0, y_valid.max()], linestyle="--", color="grey")
axes[0].set_xlabel("Actual charges")
axes[0].set_ylabel("Predicted charges")
axes[0].set_title("Ridge: actual versus predicted")
axes[0].legend()
axes[1].scatter(ridge_pred, residual, alpha=0.6)
axes[1].axhline(0, linestyle="--", color="grey")
axes[1].set_xlabel("Predicted charges")
axes[1].set_ylabel("Residual (actual - predicted)")
axes[1].set_title("Ridge: residual plot")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Inspect.** Smoker bands suggest a smoker × BMI interaction. Shrinkage alone cannot add this missing structure.

# %% [markdown]
# ## Part B — Regression on a time series: validate in date order

# %% [markdown]
# Sort two years of daily rentals by date; reserve the last 20% for validation to avoid training on future days.

# %%
bikes = pd.read_csv(DATA / "bike_sharing_dataset_day.csv")
bikes["dteday"] = pd.to_datetime(bikes["dteday"])
bikes = bikes.sort_values("dteday").reset_index(drop=True)

# cnt = casual + registered, so those two columns would leak the answer.
bike_cat = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
bike_num = ["temp", "atemp", "hum", "windspeed"]
X_b = bikes[bike_cat + bike_num]
y_b = bikes["cnt"]

cut = int(len(bikes) * 0.80)
print("train:", bikes["dteday"].iloc[0].date(), "to", bikes["dteday"].iloc[cut - 1].date())
print("valid:", bikes["dteday"].iloc[cut].date(), "to", bikes["dteday"].iloc[-1].date())

# %% [markdown]
# One-hot encode calendar categories; pass numerical weather features directly to the unscaled tree model.

# %%
Xb_train, Xb_valid = X_b.iloc[:cut], X_b.iloc[cut:]
yb_train, yb_valid = y_b.iloc[:cut], y_b.iloc[cut:]
dates_valid = bikes["dteday"].iloc[cut:]

bike_model = Pipeline([
    ("preprocess", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), bike_cat),
        ("num", "passthrough", bike_num),
    ])),
    ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)),
])
bike_model.fit(Xb_train, yb_train)
bike_pred = bike_model.predict(Xb_valid)
print("daily MAE:", round(mean_absolute_error(yb_valid, bike_pred), 1),
      " (mean daily demand in validation:", round(yb_valid.mean(), 1), ")")

plt.figure(figsize=(8, 3.6))
plt.plot(dates_valid, yb_valid.to_numpy(), label="actual")
plt.plot(dates_valid, bike_pred, label="predicted")
plt.xlabel("Date")
plt.ylabel("Daily rentals")
plt.title("Chronological validation: actual versus predicted demand")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Inspect.** The forest tracks seasonality but underpredicts peaks beyond the training-label range; it cannot extrapolate.

# %% [markdown]
# Repeat chronologically on 17,379 hourly rows, adding `hr`; plot one validation week.

# %%
hourly = pd.read_csv(DATA / "bike_sharing_dataset_hour.csv")
hourly["dteday"] = pd.to_datetime(hourly["dteday"])
hourly = hourly.sort_values(["dteday", "hr"]).reset_index(drop=True)
hour_cat = bike_cat + ["hr"]
Xh = hourly[hour_cat + bike_num]
yh = hourly["cnt"]
cut_h = int(len(hourly) * 0.80)

hour_model = Pipeline([
    ("preprocess", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), hour_cat),
        ("num", "passthrough", bike_num),
    ])),
    ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)),
])
hour_model.fit(Xh.iloc[:cut_h], yh.iloc[:cut_h])
hour_pred = hour_model.predict(Xh.iloc[cut_h:])
print("hourly MAE:", round(mean_absolute_error(yh.iloc[cut_h:], hour_pred), 1),
      " (mean hourly demand in validation:", round(yh.iloc[cut_h:].mean(), 1), ")")

week = slice(0, 24 * 7)
plt.figure(figsize=(9, 3.4))
plt.plot(range(24 * 7), yh.iloc[cut_h:].to_numpy()[week], label="actual")
plt.plot(range(24 * 7), hour_pred[week], label="predicted")
plt.xlabel("Hour of the first validation week")
plt.ylabel("Rentals per hour")
plt.title("Hourly demand has a shape that hr captures")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Inspect.** Working days have two peaks; weekends one. Lab 8 predicts this series from 24 previous hours.

# %% [markdown]
# ## Part C — Classification: baselines, probabilities, and the threshold

# %% [markdown]
# Compare logistic regression and a forest using seven passenger features, no identifiers, and stratified splits. Accuracy evaluates thresholded decisions; AUC evaluates ranking.

# %%
titanic = pd.read_csv(DATA / "titanic_dataset.csv")
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
t_num = ["Age", "SibSp", "Parch", "Fare"]
t_cat = ["Pclass", "Sex", "Embarked"]
X_t = titanic[features]
y_t = titanic["Survived"]
Xt_train, Xt_valid, yt_train, yt_valid = train_test_split(
    X_t, y_t, test_size=0.25, random_state=42, stratify=y_t
)
print("majority-class baseline accuracy:", round(1 - yt_valid.mean(), 3))

scores = {}
fitted_t = {}
for name, estimator in {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42),
}.items():
    pipe = Pipeline([("preprocess", make_preprocess(t_num, t_cat)), ("model", estimator)])
    pipe.fit(Xt_train, yt_train)
    fitted_t[name] = pipe
    scores[name] = {"accuracy": accuracy_score(yt_valid, pipe.predict(Xt_valid)),
                    "roc_auc": roc_auc_score(yt_valid, pipe.predict_proba(Xt_valid)[:, 1])}
    print(f"{name:14s} accuracy={scores[name]['accuracy']:.3f} roc_auc={scores[name]['roc_auc']:.3f}")

# The same logistic model, three decision rules (the table in the notes).
prob_t = fitted_t["logistic"].predict_proba(Xt_valid)[:, 1]
for threshold in [0.30, 0.50, 0.70]:
    pred_t = (prob_t >= threshold).astype(int)
    print(f"threshold {threshold:.2f}: precision={precision_score(yt_valid, pred_t):.2f} "
          f"recall={recall_score(yt_valid, pred_t):.2f}")

pd.DataFrame(scores).T.plot.bar(figsize=(5, 3.2), ylim=(0.5, 1.0), rot=0)
plt.axhline(1 - yt_valid.mean(), linestyle="--", color="grey", label="majority baseline")
plt.title("Two models, two metrics, one validation split")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Inspect.** Both beat the baseline, but metric rankings differ. Use Part D cross-validation to assess small differences.

# %% [markdown]
# For 30 tumour measurements, predict malignant (`M`) versus benign (`B`). Plot probabilities by true class; a threshold marks malignant decisions.

# %%
cancer = pd.read_csv(DATA / "breast_cancer_dataset.csv")
X_c = cancer.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore")
y_c = (cancer["diagnosis"] == "M").astype(int)
Xc_train, Xc_valid, yc_train, yc_valid = train_test_split(
    X_c, y_c, test_size=0.25, random_state=42, stratify=y_c
)
cancer_model = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000)),
])
cancer_model.fit(Xc_train, yc_train)
prob = cancer_model.predict_proba(Xc_valid)[:, 1]

plt.figure(figsize=(6, 3.4))
plt.hist(prob[yc_valid == 0], bins=25, alpha=0.6, label="benign (actual)")
plt.hist(prob[yc_valid == 1], bins=25, alpha=0.6, label="malignant (actual)")
plt.axvline(0.50, color="black", linestyle="--", label="threshold 0.50")
plt.axvline(0.30, color="red", linestyle=":", label="threshold 0.30")
plt.xlabel("Predicted probability of malignant")
plt.ylabel("Tumours")
plt.title("A threshold is a vertical line through this picture")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Confusion-matrix rows are truth; columns are decisions. Precision measures correct malignant alerts; recall measures detected malignancies. Lowering the threshold increases alerts and cannot reduce recall.

# %%
for threshold in [0.50, 0.30]:
    pred = (prob >= threshold).astype(int)
    print(f"\nthreshold = {threshold:.2f}")
    print(confusion_matrix(yc_valid, pred))
    print(classification_report(yc_valid, pred, digits=3))
    ConfusionMatrixDisplay.from_predictions(yc_valid, pred, display_labels=["benign", "malignant"])
    plt.title(f"Confusion matrix at threshold {threshold:.2f}")
    plt.tight_layout()
    plt.show()

# %%
thresholds = np.linspace(0.05, 0.95, 19)
precisions = [precision_score(yc_valid, (prob >= t).astype(int), zero_division=0) for t in thresholds]
recalls = [recall_score(yc_valid, (prob >= t).astype(int)) for t in thresholds]

plt.figure(figsize=(6, 3.4))
plt.plot(thresholds, precisions, marker="o", label="precision")
plt.plot(thresholds, recalls, marker="o", label="recall")
plt.xlabel("Decision threshold")
plt.ylabel("Score")
plt.title("The threshold trades recall for precision")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ROC AUC measures ranking across classes; average precision summarizes precision–recall, emphasizing positive retrieval.

# %%
print("ROC AUC:", round(roc_auc_score(yc_valid, prob), 3))
print("PR AUC:", round(average_precision_score(yc_valid, prob), 3))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
RocCurveDisplay.from_predictions(yc_valid, prob, ax=axes[0])
axes[0].set_title("ROC curve")
PrecisionRecallDisplay.from_predictions(yc_valid, prob, ax=axes[1])
axes[1].set_title("Precision-recall curve")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part D — Tune with cross-validation, then test once

# %% [markdown]
# Reserve Titanic test rows. Within training data, use five-fold CV to tune `C` and `class_weight`; refit the winner on all training rows.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.25, random_state=42, stratify=y_t
)
search = GridSearchCV(
    Pipeline([("preprocess", make_preprocess(t_num, t_cat)),
              ("model", LogisticRegression(max_iter=2000))]),
    param_grid={
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__class_weight": [None, "balanced"],
    },
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
)
search.fit(X_train, y_train)

cv_table = pd.DataFrame(search.cv_results_)
plt.figure(figsize=(5.5, 3.4))
for weight, group in cv_table.groupby(cv_table["param_model__class_weight"].astype(str)):
    plt.errorbar(group["param_model__C"].astype(float), group["mean_test_score"],
                 yerr=group["std_test_score"], marker="o", capsize=3, label=f"class_weight={weight}")
plt.xscale("log")
plt.xlabel("C (weaker regularization to the right)")
plt.ylabel("Cross-validated ROC AUC")
plt.title("What the grid search saw")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Inspect.** Fold spread exceeds many differences between settings; avoid overinterpreting small gains.

# %% [markdown]
# Only now is the test set touched, once.

# %%
test_prob = search.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_prob)
print("best parameters:", search.best_params_)
print("cross-validated ROC AUC:", round(search.best_score_, 3))
print("final test ROC AUC:", round(test_auc, 3))

# %% [markdown]
# ## Part E — Save the models and use them

# %% [markdown]
# Save preprocessing and estimator together with `joblib.dump`; add a card documenting inputs and evaluation.

# %%
final_model = search.best_estimator_
model_path = MODELS / "titanic_survival.joblib"
joblib.dump(final_model, model_path)

card = {
    "task": "predict Titanic survival (1 = survived)",
    "features": features,
    "model": "logistic regression pipeline",
    "best_parameters": search.best_params_,
    "training_rows": int(len(X_train)),
    "final_test_roc_auc": round(float(test_auc), 3),
    "decision_threshold": 0.5,
    "scikit_learn_version": sklearn.__version__,
}
(MODELS / "titanic_survival_card.json").write_text(json.dumps(card, indent=2))
size_kb = model_path.stat().st_size / 1024
print(f"saved: {model_path.name} ({size_kb:.1f} KB) and titanic_survival_card.json")

# %% [markdown]
# Load the pipeline and predict new rows with matching columns, without retraining. Repeat for insurance customers differing in one feature.

# %%
loaded = joblib.load(model_path)
new_passengers = pd.DataFrame([
    {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 80.0, "Embarked": "C"},
    {"Pclass": 3, "Sex": "male", "Age": 40, "SibSp": 0, "Parch": 0, "Fare": 7.9, "Embarked": "S"},
])
new_passengers["survival_probability"] = loaded.predict_proba(new_passengers)[:, 1].round(3)
print(new_passengers)

joblib.dump(fitted["linear regression"], MODELS / "insurance_charges.joblib")
pricing = joblib.load(MODELS / "insurance_charges.joblib")
new_customer = pd.DataFrame([
    {"age": 35, "sex": "female", "bmi": 27.5, "children": 2, "smoker": "no", "region": "southeast"},
    {"age": 35, "sex": "female", "bmi": 27.5, "children": 2, "smoker": "yes", "region": "southeast"},
])
new_customer["predicted_charges"] = pricing.predict(new_customer).round(0)
print(new_customer)

# %% [markdown]
# The lab ends by running the saved passenger model from the terminal with `predict_passenger.py`. That script lives beside this notebook; run it from this folder:
#
# ```
# python predict_passenger.py --pclass 2 --sex female --age 30 --fare 20
# ```

# %% [markdown]
# ## Try it yourself

# %% [markdown]
# **Run the Part C classification workflow on `heart_failure_dataset.csv` (label `HeartDisease`).** Column lists come from the data types, with the lab's `make_preprocess`; report accuracy, ROC AUC and the confusion matrix at threshold 0.5.

# %%
heart = pd.read_csv(DATA / "heart_failure_dataset.csv")
X_h = heart.drop(columns="HeartDisease")
y_h = heart["HeartDisease"]
h_num = X_h.select_dtypes(include="number").columns.tolist()
h_cat = X_h.select_dtypes(exclude="number").columns.tolist()
Xh_train, Xh_valid, yh_train, yh_valid = train_test_split(
    X_h, y_h, test_size=0.25, random_state=42, stratify=y_h
)

for name, estimator in {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42),
}.items():
    pipe = Pipeline([("preprocess", make_preprocess(h_num, h_cat)), ("model", estimator)])
    pipe.fit(Xh_train, yh_train)
    p = pipe.predict_proba(Xh_valid)[:, 1]
    pred = (p >= 0.5).astype(int)
    print(f"{name:14s} accuracy={accuracy_score(yh_valid, pred):.3f} roc_auc={roc_auc_score(yh_valid, p):.3f}")
    print(confusion_matrix(yh_valid, pred), "
")

# %% [markdown]
# ## Decision memo

# %% [markdown]
# **Recommendation: use a threshold of about 0.30 for the breast-cancer model.** A missed malignancy costs far more than a false alarm (a follow-up test), so the threshold should sit below 0.50.
#
# Part C validation evidence (143 tumours, 53 malignant): at 0.50 the confusion matrix has 4 missed malignancies (FN) and 1 false alarm (FP), precision 0.980, recall 0.925. At 0.30 it has 2 FN and 1 FP, precision 0.981, recall 0.962. Lowering the threshold converted two missed cancers into caught ones without adding a single false alarm, and lowering a threshold can never reduce recall. The precision–recall plot shows precision only starts to fall at lower thresholds, so 0.30 sits before the price rises.
#
# Caveat: each tumour is about 1.9 percentage points of recall on 53 positives, so this is a small validation sample; the threshold should be re-checked on more data before real use, and moving further down trades more false alarms for fewer misses.

# %% [markdown]
# ## Check your understanding

# %% [markdown]
# **1. Why is logistic regression called a linear classifier even though it outputs a nonlinear sigmoid probability?**
#
# The model computes a linear score $z = w^\top x + b$ and only then squashes it: $p = \sigma(z) = 1/(1+e^{-z})$. The sigmoid is monotonic, so thresholding $p$ at 0.5 is the same as thresholding $z$ at 0, and the set of points where $z = 0$ is a hyperplane. The *decision boundary* is therefore linear in the features (a line in 2-D, a plane in 3-D), even though the probability changes nonlinearly as you move away from it. Equivalently, the log-odds $\log\frac{p}{1-p} = z$ are linear in $x$.
#
# **2. Why should the probability model and the operational classification threshold be treated as separate objects?**
#
# The model's job is to rank cases and estimate how likely each is to be positive; that is learned from data and is judged by AUC, log loss or calibration. The threshold is a *business or clinical decision* that depends on how much a missed positive costs relative to a false alarm, on prevalence, and on capacity (how many alerts someone can act on). Those costs change over time and between deployments while the model need not. Keeping them separate means you can move the threshold without retraining, tune it on validation data, document it in the model card (as `decision_threshold` above), and judge the model on threshold-free metrics. Part C shows the same fixed model giving different error trade-offs as the threshold moves (0.50 versus 0.30 in the breast-cancer confusion matrices).
#
# **3. What changes when binary logistic regression is extended to a multiclass softmax model?**
#
# Instead of one weight vector and one sigmoid there is one weight vector per class, giving K scores $z_k = w_k^\top x + b_k$. The softmax turns them into K probabilities that are positive and sum to one: $p_k = e^{z_k}/\sum_j e^{z_j}$. Binary logistic regression is the K = 2 special case (only the difference of the two scores matters). The loss becomes the categorical cross-entropy over K classes, the decision is the arg-max class rather than a single threshold, and the "threshold" idea generalizes to per-class cost weights or probability cut-offs. The decision boundaries between classes are still linear.
