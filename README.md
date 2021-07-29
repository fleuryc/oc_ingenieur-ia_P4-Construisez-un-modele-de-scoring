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
# conda install jupyterlab ipywidgets numpy pandas matplotlib seaborn plotly statsmodels scikit-learn scikit-learn-intelex
# conda install -c conda-forge voila
# or :
conda env update -f environment.yml
```

- Pip

```bash
# pip install jupyterlab ipywidgets numpy pandas matplotlib seaborn plotly statsmodels scikit-learn scikit-learn-intelex voila
# or :
pip install -r requirements.txt
```

## Showcase with Voilà

You can display the live Notebook with only the Markdown cells and code results (without the Python source code cells) with [Voilà](https://github.com/voila-dashboards/voila) :

- Launch JupyterLab

```bash
jupyter-lab
```

- Execute the whole Notebook in JupyterLab (http://localhost:8888/lab/tree/notebook.ipynb)
- Render the Notebook with Voilà : (http://localhost:8888/voila/render/notebook.ipynb)

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
