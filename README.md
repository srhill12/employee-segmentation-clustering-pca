# Employee Segmentation Using K-Means Clustering and PCA

## Overview

This project applies unsupervised machine learning — K-Means clustering combined 
with Principal Component Analysis (PCA) — to segment employees based on performance 
metrics, training history, and development readiness.

It was originally developed as part of my applied AI work in an employee development 
role, where I needed a systematic, data-informed way to identify which employees 
would benefit most from targeted training interventions versus those ready for 
accelerated leadership development.

The core business problem: manual performance reviews were inconsistent and 
time-consuming. This model provided a repeatable, auditable framework for grouping 
employees by development need — reducing bias in training assignment decisions and 
giving managers a clearer picture of team composition.

---

## Business Context

As Director of Employee Development at a multi-location retail organization, I used 
this approach to:

- Identify employees ready for managerial advancement vs. those needing foundational 
  skill development
- Reduce subjective bias in training recommendations by grounding decisions in 
  quantified performance data
- Design targeted development tracks for each cluster rather than one-size-fits-all 
  training programs
- Build a repeatable, auditable process that managers could understand and trust

Cluster analysis revealed meaningful segmentation across three employee profiles:
- **High-readiness employees** (strong performance ratings, high training assessment 
  scores, peer feedback positive) — candidates for leadership development tracks
- **Core performers** (solid attendance and customer satisfaction, moderate training 
  completion) — targeted skill-building interventions
- **Development-priority employees** (lower assessment scores, gaps in training 
  completion) — structured support programs with closer manager touchpoints

This informed a restructured development program that improved training targeting 
and reduced repeat training cycles.

---

## Ethical Considerations

This project was built with data privacy and fairness as explicit design constraints 
— not afterthoughts.

**Data privacy:** All data used in this repository is synthetically generated. No 
real employee records, names, or personally identifiable information are included. 
In the original operational deployment, access to employee performance data was 
restricted to HR leadership and the employee's direct manager, consistent with 
internal data governance policies.

**Bias mitigation:** Performance metrics were selected to reflect observable, 
job-relevant behaviors (training completion, assessment scores, attendance, customer 
satisfaction) rather than subjective manager impressions. Cluster assignments were 
used to inform development conversations — not compensation or termination decisions.

**Human-in-the-loop:** Cluster outputs were reviewed by HR leadership before any 
action was taken. No automated decisions were made based on model output alone. 
Managers were trained on how to interpret cluster assignments and what they did and 
did not indicate.

**Transparency:** Employees were informed that data-informed tools were being used 
to support development planning. The framework and its purpose were explained in 
team meetings.

These practices align with NIST AI RMF principles around transparency, 
accountability, and human oversight in AI-assisted decision-making.

---

## Technical Approach

### Stack
- Python 3.x
- pandas, numpy
- scikit-learn (PCA, KMeans, StandardScaler, LabelEncoder)
- matplotlib

### Pipeline
1. **Data generation** — synthetic employee data with realistic feature distributions 
   across performance, training, and readiness dimensions
2. **Preprocessing** — label encoding for categorical features, StandardScaler 
   normalization to prevent feature dominance
3. **Dimensionality reduction** — PCA to 2 components for visualization while 
   preserving variance structure
4. **Clustering** — K-Means (k=3) applied to PCA-reduced feature space
5. **Interpretation** — PCA component weights analyzed to identify which features 
   drive each principal component

### Why PCA + K-Means?
PCA reduces the feature space to its most explanatory dimensions before clustering, 
which improves K-Means performance on high-dimensional data and makes cluster 
separation more interpretable. The PCA weights reveal which underlying factors 
(e.g., training completion vs. readiness signals) are driving employee groupings.

---

## Files

| File | Description |
|------|-------------|
| `employee_segmentation_analysis.ipynb` | Main notebook — full pipeline with visualizations |
| `README.md` | This file |

---

## Key Outputs

- **PCA scatter plot** — employees plotted in reduced feature space, colored by 
  cluster assignment with name annotations
- **PCA component weights table** — shows which features load most heavily on each 
  principal component, enabling human-interpretable cluster labeling
- **Cluster summary** — development profile for each segment with recommended 
  intervention type

---

## Limitations and Honest Notes

- **Synthetic data** — the model runs on randomly generated data. In a real 
  deployment, cluster stability and feature selection would need validation against 
  actual performance outcomes over time.
- **Small n** — 15 employees is too small for robust clustering in production. 
  This is a proof-of-concept demonstration of the methodology.
- **Static snapshot** — the model reflects a point-in-time assessment. Employee 
  development is dynamic; this would need to be rerun periodically with fresh data 
  in an operational setting.
- **K selection** — k=3 was chosen based on interpretability for this use case. 
  An elbow curve analysis would be appropriate for larger datasets.

---

## Relevance to AI Governance

This project is a small-scale example of a recurring governance challenge: using 
AI-assisted tools to make decisions that affect people, in a context where the 
people affected have limited visibility into how the system works.

The design choices made here — synthetic data for privacy, human-in-the-loop review, 
transparent communication to employees, restriction of use to development (not 
punitive) decisions — reflect the same principles I apply at larger scale when 
conducting AI risk assessments and building governance frameworks.

The hardest part of responsible AI adoption isn't writing the framework. It's 
getting the humans in the loop to trust the process enough to use it well.

---

## Author

**Steven Hill**  
AI Ethics & Policy Professional | Purdue University MSAI Candidate  
[LinkedIn](https://linkedin.com/in/stevenrhill) | [GitHub](https://github.com/srhill12)