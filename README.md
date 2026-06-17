# 💊 ML-Driven Pharmacovigilance & Drug Side Effects Analysis

An end-to-end data mining and machine learning exploratory pipeline designed to profile, clean, and analyze a 100k records dataset tracking patient drug side effects, lifestyle risk factors, and clinical severities.

## 📋 Project Architecture & Completed Tasks

This repository details the fundamental phase of our data preprocessing and exploratory data analysis (EDA) pipeline, satisfying the core clinical requirements for robust machine learning model preparation.

### 🔍 Phase 1: Data Understanding & Completeness Audit
- **Feature Typing:** Classified all 16 pipeline features into continuous numerical metrics (`age`, `dosage_mg`, `recovery_days`), nominal categories (`drug_name`, `side_effect`, `country`), and target-driven ordinals (`severity`).
- **Data Quality Assessment:** Audited the dataset for completeness, systematically tracking structural null parameters, absolute duplicate patient records, and logical domain errors (e.g., negative ages).

### 🛠️ Phase 2: Statistical Cleaning & Structural Imputation
- **Data Quality Handling:** Resolved structural noise by eliminating identical duplicates and establishing median baseline values for missing clinical metrics.
- **Anomaly Correction:** Isolated and normalized domain anomalies, repairing extreme unrepresentative value placeholders (e.g., raw value `999` constraints in recovery durations).

### 📊 Phase 3: Comprehensive Comparative Normalization
To prepare the dataset for downstream distance-based classifiers (SVM, k-NN) and prevent feature dominance due to varying scale ranges:
1. **Min-Max Normalization:** Scaled continuous metrics strictly within a bounded range of `[0.0, 1.0]`.
2. **Z-Score Standardization:** Centered distributions around a Mean ($\mu$) of `0.0` with a Standard Deviation ($\sigma$) of `1.0`, ensuring resilient handling of clinical outliers.

### 📈 Phase 4: Statistical Significance & Feature Relevance
- **Chi-Square ($\chi^2$) Feature Selection:** Automated a dynamic data pipeline that bins continuous attributes into statistical quantiles to run global $\chi^2$ testing against the `severity` target, ranking columns by actual predictive impact.
- **Multicollinearity Heatmapping:** Constructed a clean, low-triangle masked feature correlation matrix across all label-encoded variables to explicitly isolate and remove redundant features before model training.

### 🖼️ Phase 5: High-Dimensional Interactive Visualization
- **Target Distributions:** Generated robust proportional visual cross-tabulations assessing `severity` behavior.
- **Interactive Dashboards:** Built dynamic `Plotly` dashboard suites exploring correlations across 8 separate critical dimensions simultaneously:
  $$\text{Severity vs. } \{ \text{Chronic Condition, Smoker, Alcohol Use, Hospitalized, Outcome, Side Effect, Drug Name, Age Group} \}$$

## Dataset:

The domain of the dataset is Pharmacy domain. The name of the dataset is 100k Drug Side Effects Dataset. It is from Kaggle website. The dataset is a high-quality synthetic healthcare dataset containing 100,000 patient drug side-effect reports designed for Machine Learning, Deep Learning, Predictive Analytics, Healthcare AI, Data Visualization Projects. This dataset simulates realistic pharmacovigilance and adverse drug reaction (ADR) reporting patterns using structured healthcare-style data distributions.

### Description of Dataset Features:

patient_id = Unique patient identifier
age = age of Patient
gender = Male / Female / Other
country = Patient's country of origin/residence
drug_name = Prescribed medication
dosage_mg = Drug dosage in mg
side_effect = Reported adverse reaction
severity = Mild / Moderate / Severe
outcome = Recovery status
report_date = Side effect report date
treatment_start_date = Medication start date
chronic_condition = Existing health condition
smoker = Smoking status
alcohol_use = Alcohol consumption pattern
hospitalized = Hospitalization status
recovery_days = Recovery duration

## 🚀 Key Clinical Insights
Time-to-Onset Profile: The vast majority of drug-induced adverse events peak and are actively documented within the initial 14 to 20 days post-treatment start.

Compounding Risk Factors: Geriatric demographics possessing multiple pre-existing chronic_conditions correlate exponentially with prolonged recovery_days and higher rates of hospitalized=Yes.

Algorithmic Selection: Due to the existence of severe, real-world clinical outliers in recovery timelines, Z-Score Normalization was selected as the optimal pipeline transformation over Min-Max scaling.

## 🛠️ Tech Stack & Dependencies
Language: Python 3.10+

Environment: Google Colab / Jupyter Notebooks

Core Packages: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly

## 🔮 Next Implementation Steps
Moving forward into the core machine learning phases, the project is scheduled to execute:

Association Rule Mining: Running Apriori and FP-Growth algorithms to uncover latent combination pairs among lifestyle inputs and adverse reactions.

Supervised Classification Models: Constructing and comparing Naïve Bayes, Decision Tree (Information Gain ordering), and Support Vector Machine (SVM) classifiers to predict patient risk profiles.
