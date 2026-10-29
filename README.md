# Week 3: Unsupervised Learning and Clustering Analysis

This project applies K-Means clustering to the public Iris dataset using Python and scikit-learn. It covers preprocessing, choosing the number of clusters, PCA visualization, cluster profiling, and post-hoc comparison with known species labels.

## Files
- `Week3_Clustering_Report.docx` — detailed report
- `clustering_analysis.py` — reproducible analysis script
- `iris_clustering_dataset.csv` — feature data plus reference species label
- `iris_clustering_results.csv` — cluster assignments from the final model
- `visualizations/` — figures used in the report
- `requirements.txt` — Python dependencies

## Run
```bash
pip install -r requirements.txt
python clustering_analysis.py
```
The script writes generated files to an `outputs/` directory. The species reference label is not used to fit the clustering model; it is used only for optional evaluation after clustering.

## Dataset
Iris dataset bundled with scikit-learn: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
