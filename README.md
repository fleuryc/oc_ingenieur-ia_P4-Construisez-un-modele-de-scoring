[![Codacy Badge](https://app.codacy.com/project/badge/Grade/dd4a397e7bfe43a5bcf16ff7e6a2056a)](https://www.codacy.com/gh/fleuryc/oc_ingenieur-ia_P4-Construisez-un-modele-de-scoring/dashboard)

# Prêt à dépenser : Create a scoring model

Repository of OpenClassrooms' [AI Engineer path](https://openclassrooms.com/fr/paths/188-ingenieur-ia), project #4

Goal : use Jupyter Notebook and Scikit-Learn to create, assess and improve a scoring model based on the insurance's clients credit history.

You can see the results here :

- [Presentation]()
- [HTML page with interactive plots]()

## Requirements

- Conda

```bash
# conda install -c conda-forge jupyterlab ipywidgets numpy pandas matplotlib seaborn plotly statsmodels imbalanced-learn scikit-learn scikit-learn-intelex auto-sklearn xgboost lightgbm graphviz python-graphviz lime shap
# or :
conda env update -f environment.yml
```

- Pip

```bash
# pip install jupyterlab ipywidgets numpy pandas matplotlib seaborn plotly statsmodels imbalanced-learn scikit-learn scikit-learn-intelex auto-sklearn xgboost lightgbm graphviz python-graphviz lime shap
# or :
pip install -r requirements.txt
```

## Troubleshooting

- Fix Plotly issues with JupyterLab

cf. [Plotly troubleshooting](https://plotly.com/python/troubleshooting/#jupyterlab-problems)

```bash
jupyter labextension install jupyterlab-plotly
```

- If using Jupyter Notebook instead of JupyterLab, uncomment the following lines in the notebook

```python
import plotly.io as pio
pio.renderers.default='notebook'
```
