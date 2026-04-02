# H-CAAR: Traffic Accident Severity Prediction Pipeline

This repository contains the source code, data processing scripts, and predictive models for our research paper: **"Asymmetric Risk Thresholding in Cost-Sensitive Random Forests for Proactive Traffic Hazard Warning"**.

## 1. The Core Problem: The Accuracy Paradox
Traffic accident datasets inherently suffer from extreme class imbalance (e.g., Safe outcomes outnumber Fatalities by 90:1). During our baseline testing, we encountered a severe "Accuracy Paradox": standard symmetric algorithms like Support Vector Machine (SVM) achieved the highest global accuracy (82.17%) but completely failed to detect severe accidents (Class 3 Recall = 0.00%). 

By blindly predicting that nearly all accidents are "Safe", standard models achieve high nominal accuracy but become operationally dangerous, systematically suppressing rare but life-threatening hazard signals.

## 2. Our Methodology and Logic (H-CAAR)
To overcome this critical operational blind spot, we developed the **Hybrid Context-Aware and Asymmetric Risk (H-CAAR)** pipeline. The logic is built upon a three-tier architecture:

### Latent Risk Profiling
Before predictive modeling, we establish spatial-temporal priors. Using a curated dataset of 206,410 independent records from Montgomery County, we apply **K-Means clustering** and **Folium spatial mapping** to extract localized risk topologies based on speed limits and temporal dynamics.

### Cost-Sensitive Predictive Engine
Instead of using synthetic oversampling (like SMOTE) which introduces epistemic noise, we employ a **Cost-Sensitive Random Forest**. To force the ensemble to pay attention to the minority class, we modified the standard Gini impurity metric via a custom penalty weight matrix (`1:3:10`). The algorithm is instructed to penalize the misclassification of Severe/Fatal injuries ten times more heavily than nominal safe outcomes.

### Asymmetric Risk Override Mechanism (Decision Theory)
Standard ensembles natively converge toward the conditional mean (0.5 threshold). We mathematically bypassed this using an inference-time override mechanism. 
* We derived the optimal decision threshold empirically via $F_5$-score optimization ($\beta=5.0$), weighting recall 5 times more heavily than precision
* Substituting this utility ratio into our expected cost equation yields an optimal decision boundary of **`τ = 6.0%`**.
* **Result:** If the predicted probability of a fatality exceeds just 6.0%, the system overrides the "Safe" majority vote and triggers a localized hazard warning. This asymmetric logic drastically amplifies severe-event recall to approximately 57.00% (a 9-fold improvement over the baseline).

## 3. System Architecture

<img width="6000" height="3375" alt="pipeline" src="https://github.com/user-attachments/assets/18d3e7e7-ded0-4167-bdc7-8a4e8b9dcae7" />

## 4. Repository Structure
The codebase is structured to ensure full computational reproducibility:

* `cleaning.py`: Curates the raw dataset, handles missingness, and performs deterministic binary encoding for behavioural indicators.
* `sql_analysis.py`: Extracts macroscopic statistical insights and temporal risk patterns.
* `eda_analysis.py`: Generates correlation matrices and K-Means clustering visualizations.
* `traffic_heatmap.py`: Renders interactive thermal density maps of severe incidents.
* threshold_analysis.py: Performs an F-beta sweep across candidate thresholds to empirically identify the optimal decision boundary.
* full_pipeline_eval.py: Provides a comprehensive evaluation and side-by-side confusion matrix comparison between different operational thresholds.
* `model_comparison.py`: Evaluates baseline algorithms and mathematically exposes the Accuracy Paradox.
* `ml_prediction.py`: Trains the core Cost-Sensitive Random Forest model using the customized weight matrix.
* `demo_app.py`: A lightweight Command-Line Interface (CLI) that requires 5 proactive inputs to trigger real-time hazard detection.

## 5. How to Run
To replicate the findings or utilize the predictive engine, execute the pipeline in the following sequence:

```bash
# 1. Clean data and engineer features
python cleaning.py

# 2. Train the Predictive Engine
python ml_prediction.py

# 3. Analysis to find the optimal F-beta threshold
python threshold_analysis.py

# 4. Launch the Early Warning System
python demo_app.py
```

