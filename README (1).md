# Genetic Disorder Prediction using Machine Learning

Predicting a patient's genetic disorder inheritance category - Mitochondrial, Single Gene, or Multifactorial - from routine clinical and family history data. The goal is a triage tool that narrows the focus of genetic testing, not a replacement for testing itself.

**Team:** **Godwin Gyamfi Duah**, Ujjwal Khadka

**Course:** CISC 520-50-B - Data Engineering and Mining

**Institution:** Harrisburg University of Science and Technology 

**Supervisor:** Ki Lee, PhD 

**Date:** August 2026

## Overview

Genetic diseases are a massive health burden worldwide, and delayed diagnosis is a common clinical problem. More than 140 million infants are born every year, and over ten million suffer from a significant birth defect of genetic or partially genetic origin, many of which are detected late (Rahman et al., 2022). Diagnosis is difficult because rare disorders can mimic the symptoms of more common ones, and genetic testing to confirm a diagnosis is costly and not available everywhere.

Machine learning has approached this problem along two lines. One works directly from genome sequencing data. Chen et al. (2025) review this literature, covering variant calling, pathogenicity prediction, and splicing analysis. The other, more accessible line uses routinely collected patient medical history instead of sequence data. Rahman et al. (2022) argue that sequencing-based approaches, though powerful, are constrained by cost and tend to overlook clinical parameters such as abortion history and blood cell counts. Working from patient history alone, they reported 86.6% test accuracy on a binary mitochondrial-versus-multifactorial task using a support vector machine. Ghazal et al. (2022) applied support vector machines and k-nearest neighbors to a smaller multifactorial subset and reported 92.5% test accuracy across three disease subclasses.

Using the Of Genomes and Genetics dataset (Arya, 2021), this project follows that second line of work, predicting which of three inheritance categories a patient's disorder belongs to from 37 clinical, demographic, and family history features.

The aim was mostly deductive, following up on an exploratory study that generated five hypotheses in Deliverable 2. The core hypothesis was that genetic history characteristics would rank highest in Random Forest feature importance, given their direct biological relationship to inheritance. A second hypothesis proposed that Random Forest would outperform Logistic Regression. A supplementary, more open-ended goal was to discover which feature combinations the models actually treated as predictive.

## Dataset

The data comes from the [Of Genomes and Genetics](https://www.kaggle.com/datasets/aryarishabh/of-genomes-and-genetics-hackerearth-ml-challenge) dataset (Arya, 2021), originally released through a HackerEarth machine learning challenge. It contains 22,083 training records across 45 columns, made up of 43 features plus two target variables, along with 9,463 held-out unlabeled test records.

The feature mix includes 6 numeric fields such as age and blood counts, 5 binary symptom flags, 21 categorical fields covering gene presence and test results, and 6 identifier columns that were dropped during preprocessing.

Two challenges stood out early. First, severe class imbalance: Mitochondrial disorders (9,686 records) outnumber Single Gene disorders (7,291) and Multifactorial disorders (1,985), reaching up to a 53 to 1 ratio at the disorder subclass level. Second, substantial missingness: Mother's age and Father's age were each missing over 30% of the time, most clinical columns were missing around 14% of the time, and either target label was missing in 22.3% of records.

## Exploratory Findings (Deliverable 2)

Missingness turned out to be scattered randomly across patients rather than concentrated in any particular subgroup, which supported using imputation rather than dropping large chunks of data outright. Numeric features also showed near-zero correlation with each other, ruling out any multicollinearity concern.

The most important finding, though, was that box plots showed no individual numeric feature meaningfully separates the three disorder categories on its own. That observation directly motivated the choice of an ensemble method that could combine many features rather than lean on any one of them, and it's why Random Forest was proposed as the primary method, with Logistic Regression as a baseline and XGBoost as a stretch goal.

## Methods (Deliverable 3)

**Preprocessing.** Six columns were dropped as identifiers or administrative fields with no clinical meaning: Patient ID, Patient First Name, Family Name, Father's Name, Institute Name, and Location of Institute. Twenty one categorical columns were label encoded, with encoders fitted jointly across training and test data to guarantee consistent mappings. Remaining gaps in the feature matrix were filled by median imputation, fitted on training data and applied to test.

Records missing either target were removed rather than imputed, since a fabricated diagnosis supplies no valid supervision signal. This eliminated 4,923 of 22,083 records, or 22.3%, leaving 17,160: 8,766 mitochondrial, 6,591 single gene, and 1,803 multifactorial. The data were split 80/20 into training and validation partitions, stratified on the target to preserve class proportions, giving 13,728 and 3,432 records, respectively. SMOTE (Chawla et al., 2002) was then applied to the training partition only, synthesizing minority class records by interpolating between each minority instance and its nearest same-class neighbors. This raised every class to 7,013 records, 21,039 in total. The validation partition was deliberately left imbalanced so that measured performance would reflect the real class distribution.

**Algorithm.** Random Forest (Breiman, 2001) works in three steps. First, T bootstrap samples are drawn with replacement from the training set. Second, an unpruned decision tree is grown on each; at every node, a random subset of roughly the square root of p features is considered, and the split that maximizes Gini impurity reduction is chosen. Third, a new record is classified by majority vote across all T trees. Randomizing both records and features decorrelates the trees, so averaging their votes reduces variance without inflating bias. T was set to 100. Out-of-bag scoring exploits the roughly 37% of records excluded from each bootstrap sample, giving an internal validation estimate at no additional cost.

Training cost works out to O(T times the square root of p times n log n), where n is the number of records and p is the number of features, since each of the T trees sorts candidate split values across roughly root p features at each of about log n depth levels. Here T was 100, n was 21,039, and p was 37. Memory cost is O(T times n), since each tree can store up to O(n) nodes, and the whole forest has to be held in memory to make predictions. Prediction cost is O(T log n) per record, one root-to-leaf traversal per tree, and this step parallelizes across trees.

Logistic Regression served as a linear baseline, chosen because it assumes a linear decision boundary and therefore isolates how much of the performance depends on non-linear structure. XGBoost (Chen & Guestrin, 2016) was added as a boosted ensemble comparator. Where Random Forest grows trees independently and averages them, XGBoost grows them sequentially so that each tree corrects the residual errors of the ones before it. Both were trained on the same SMOTE-balanced data. All models were implemented in scikit-learn (Pedregosa et al., 2011), apart from XGBoost, which uses its own library.

**Evaluation.** Deliverable 2 proposed weighted F1 as the primary metric. On closer inspection, this proved insufficient, since weighted F1 weights each class by its support, and a validation set that's 51% mitochondrial produces a score dominated by performance on that one class, the same defect already raised against raw accuracy. Macro F1, balanced accuracy, per-class recall, and one vs rest ROC AUC were reported alongside it instead. Five-fold cross-validation was run on the unbalanced training data for a distribution-independent estimate. Balanced class weights were set for Logistic Regression and Random Forest but had negligible effect, since SMOTE had already equalized the classes.

## Results

**Aggregate performance on the validation set, n equals 3,432:**

| Model | Accuracy | Weighted F1 | Macro F1 | Balanced Acc. | Macro AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.446 | 0.449 | 0.447 | 0.539 | 0.668 |
| Random Forest | 0.594 | 0.570 | 0.510 | 0.502 | 0.727 |
| XGBoost | 0.574 | 0.547 | 0.490 | 0.490 | 0.724 |

Random baseline for balanced accuracy on three classes is 0.333, and for AUC it's 0.500.

Under weighted F1, Random Forest outperformed XGBoost and Logistic Regression, and cross-validation confirmed Random Forest above Logistic Regression across all five folds with low variance: 0.534 plus or minus 0.005 against 0.454 plus or minus 0.009. XGBoost actually achieved the highest cross-validated score of all three, at 0.557 plus or minus 0.006. Random Forest returned an out-of-bag score of 0.691.

Under balanced accuracy, the ranking reverses. Logistic Regression scored 0.539, ahead of Random Forest at 0.502 and XGBoost at 0.490. Per-class recall explains why: Random Forest correctly identified 82.8% of mitochondrial cases but only 31.9% of multifactorial cases and 36.0% of single-gene cases, while Logistic Regression was far more even across classes, at 37.1%, 79.5%, and 44.9% respectively. The gap between weighted and macro F1 quantifies this concentration directly: 0.060 for Random Forest and 0.058 for XGBoost, against just 0.001 for Logistic Regression.

One-vs-rest ROC AUC restores a clearer ordering. Random Forest achieved the highest AUC on all three classes, 0.710 for mitochondrial, 0.852 for multifactorial, and 0.620 for single gene, for a macro AUC of 0.727, ahead of XGBoost at 0.724 and Logistic Regression at 0.668. Multifactorial was the best-separated class in every single model, despite having the lowest recall of the three.

Feature importance directly contradicted the central hypothesis. The three strongest predictors were Symptom 5 (0.082), Symptom 4 (0.066), and Symptom 3 (0.053), followed by blood cell count (0.053) and white blood cell count (0.049). The genetic history features, the ones hypothesized to dominate, ranked 10th, 12th, 18th, and 20th, with importance scores between 0.033 and 0.024. Applied to the 9,463 held-out test records, Random Forest predicted 64.9% mitochondrial, 21.7% single gene, and 13.5% multifactorial.

## Discussion

Three of the five hypotheses failed. Genetic history features did not dominate feature importance; symptom flags did, by a wide margin. Numeric vitals were predicted to show low importance based on the Deliverable 2 box plots showing no separation across classes, yet blood cell count and white blood cell count ranked fourth and fifth overall. That earlier inference was flawed: univariate separation and multivariate importance are different properties, and a feature that carries no signal on its own can still carry considerable signal in combination with others. The SMOTE hypothesis turned out to be untestable given the study's design, since no unbalanced comparison was run at the same split.

The most informative result is the divergence between AUC and recall on the multifactorial class. Random Forest ranks these patients well, with an AUC of 0.852, but recovers only 31.9% of them. AUC measures ranking ability; recall measures what survives the final argmax across three classes. A class making up 10.5% of the validation set can be ranked accurately and still rarely win that final three-way vote. That's a decision threshold failure rather than a learning failure, and it's correctable without retraining anything.

That distinction also resolves the apparent contradiction between the metrics. Random Forest learned more than Logistic Regression, as AUC shows on every class, but converts less of that learning into correct hard predictions on the minority classes. Logistic Regression's better balanced accuracy reflects more evenly distributed errors, not stronger underlying discrimination.

The comparable performance across Logistic Regression, Random Forest, and XGBoost is itself a meaningful result. Boosting brought roughly two points of cross-validated F1 over bagging, against the eight points that separated bagging from the linear baseline. Since XGBoost is generally the strongest available learner for tabular data like this, that pattern suggests the binding constraint here is the data itself, not the choice of algorithm. Single gene disorders are the clearest case of this, with AUC sitting near 0.62 across all three models regardless of approach.

## Limitations

Several limitations qualify these conclusions. Listwise deletion of the 22.3% of records missing a target assumes those diagnoses were missing at random, which can't actually be verified, and if unconfirmed diagnoses tend to concentrate in harder cases, these estimates are likely optimistic. Median imputation was applied uniformly, including to parental ages that were missing in over 30% of records, and both parental age features ranked in the top eight overall, so the imputation choice materially shapes the result. Label encoding also imposes an artificial ordering on categories that have no natural order, something tree-based methods tolerate but linear models don't. Because all models were trained on SMOTE balanced data, their predicted probabilities are calibrated to an artificial 1 to 1 to 1 class prior and shouldn't be read as real clinical likelihoods. Finally, the dataset itself is synthetic, so these results don't establish real-world clinical validity on their own. Chen et al. (2025) note that data quality, bias, and interpretability remain the principal obstacles to deploying models like this clinically, and all three apply here.

## A Note on AI Tool Use

AI tools were used to assist with code generation and initial drafts of the presentation slides, with all outputs reviewed, verified, and edited by the team. Literature review and analytical write-up were done independently.

## Conclusion

This project applied Random Forest, with Logistic Regression and XGBoost as comparators, to predict genetic disorder inheritance category from 17,160 patient records. Random Forest achieved a cross-validated weighted F1 of 0.534 and a macro one-vs.-rest AUC of 0.727, exceeding the linear baseline on every class.

Two findings go beyond what the exploratory phase established. First, symptom presentation and blood cell counts outrank family genetic history as predictors in this dataset, reversing the original hypothesis and suggesting that clinical presentation carries more classifiable signal than recorded inheritance indicators do. Second, the models discriminate minority classes considerably better than their recall numbers suggest, and the shortfall comes from decision thresholds rather than from what the models actually learned.

The evaluation strategy was also revised mid-project. Weighted F1, proposed back in Deliverable 2, proved to inflate toward majority-class performance for the same reason raw accuracy had already been rejected, and reporting it alone would have concealed the fact that Random Forest misses roughly two-thirds of multifactorial patients.

## Repository Structure

The repo includes `01_data_exploration.py`, covering the Deliverable 2 statistical and visual exploratory analysis, `02_modeling_and_evaluation.py`, covering the Deliverable 3 preprocessing, modeling, and evaluation pipeline, a `results` folder with the class distribution charts, ROC curves, and feature importance plots, and this README.

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), Matplotlib, Google Colab

## Future Work

Threshold optimization or cost-sensitive decision rules should recover much of the lost minority recall, and this is probably the highest-value next step. Extending the model to the nine disorder subclasses would add finer clinical resolution, though the rarest subclasses contain under 100 records, which limits how far that can go. Probability calibration would be necessary before any predicted score could be interpreted clinically. It's also worth comparing listwise deletion against target imputation directly to test whether dropping 22.3% of records introduced meaningful bias, and validating this approach against real, non-synthetic clinical records remains a precondition for any claim of practical usefulness.

## References

Arya, R. (2021). *Of Genomes and Genetics* [HackerEarth ML Challenge data set]. Kaggle. https://www.kaggle.com/datasets/aryarishabh/of-genomes-and-genetics-hackerearth-ml-challenge

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Chen, Y.-M., Hsiao, T.-H., Lin, C.-H., & Fann, Y. C. (2025). Unlocking precision medicine: Clinical applications of integrating health records, genetics, and immunology through artificial intelligence. *Journal of Biomedical Science, 32*(1), Article 16. https://doi.org/10.1186/s12929-024-01110-w

Duah, G. G., & Khadka, U. (2026). *Genetic disorder prediction: Data mining notebook* [Unpublished Colab notebook]. Harrisburg University of Science and Technology.

Ghazal, T. M., Al Hamadi, H., Nasir, M. U., Rahman, A., Gollapalli, M., Zubair, M., Khan, M. A., & Yeun, C. Y. (2022). Supervised machine learning empowered multifactorial genetic inheritance disorder prediction. *Computational Intelligence and Neuroscience, 2022*, Article 1051388. https://doi.org/10.1155/2022/1051388

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Rahman, A., Nasir, M. U., Gollapalli, M., Alsaif, S. A., Almadhor, A. S., Mehmood, S., Khan, M. A., & Mosavi, A. (2022). IoMT-based mitochondrial and multifactorial genetic inheritance disorder prediction using machine learning. *Computational Intelligence and Neuroscience, 2022*, Article 2650742. https://doi.org/10.1155/2022/2650742
