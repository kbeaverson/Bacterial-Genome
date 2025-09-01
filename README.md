### Exploring the Bacterial Genome with Data Science
This repository my weekly workbooks and final model submission from a Build Project with the Open Avenues Foundation under the supervision of industry expert Hayden Samsun during the Spring of 2025.

Within you can follow along with my weekly workbooks to see how we went about building predictive models for antibiotic (cefepime) resistance in *Escherichia coli*. This project represents my first foray into the world of predictive modeling and data analysis, preceding any formal learning in these methods/techniques, and my participation was driven by my curiosity and eagerness to learn more and familiarize myself with these topics. 

## Aims and Approach
**Aim**: Predict cefepime resistance/susceptibility in E. coli
**Data Overview**:
- ~1,000 unique DNA samples
- ~1,500 unique genes
**Approach**:
- Data Sources:
  - Bacterial genomic sequences from the [Bacterial and Viral Bioinformatics Resource Center](https://www.bv-brc.org/)
  - Known resistance gene data from the [Comprehensive Antibiotic Resistance Database (CARD)](https://card.mcmaster.ca/)
- Feature Extraction:
  - Kmer frequency (to capture local sequence patterns)
  - Presence/Absence of CARD genes (to leverage known resistance markers)
- Models Tested:
  - Random Forest
  - Histogram Gradient Boosting Classifier
  - CNN (Attempted using raw sequencing data, didn't know enough at the time to fully explore this option)
- Baseline:
  - K-Nearest Neighbors (K=17) achieved 86% accuracy, which provided a strong starting benchmark
- Applied K-Fold cross-validation (K=5) during training for all models to avoid overfitting and obtain reliable accuracy estimates.

## Results
I trained models using each feature set individually and both combined. Combined models generally performed better.
- Best Model: HGB with combined features (89.4% accuracy predicting cefepime resistance)
- Additional tuning (hyperparameter optimization) would likely improve performance, but the project concluded near the semester's end so I had limited availability due to final exams.

## Lessons Learned
- Kmer features appear much more likely to catch nuances within genomic sequences than simple presence/absence of known resistance genes
- Observed multicollinearity between feature sets, which led to feature dominance in combined models, which strikes me as an important lesson in feature selection and regularization

## Concepts Learned
- Featurization methods: one-hot, sequence, binary encoding
- Baseline model building techniques: majority class, KNN
- Simple model building techniques with scikit-learn
- Exposure to pandas, numpy, matplotlib, tensorflow
- Hyperparameter optimization methods: random search, Bayesian optimization
