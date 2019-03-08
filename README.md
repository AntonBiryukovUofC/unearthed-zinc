unearthed-zinc
==============================

project repo for zinc


How & what to run:
=================================

Start with placing the original dataset in the form it was distributed (a zip file) under `data/raw`.
 Then run in the root directory:
`make data` to create the dataset with a little bit of simple preprocessing (dropping NA, filling NA in the inputs), as well as creating additional time aggregated features
using `tsfresh` library. The features are created by looking back at at most `N` points for each row and then aggregating the columns using functions passed as a dictionary.

The details can be found in the `src/data/make_dataset.py`

If make_dataset causes an error in the process - that could be due to one of the aggregate summaries i custom-added into tsfresh. Unfortunately,
the package does not have a **nice** way to add custom aggregates other than going into the package sources and monkey patching. Feel free to 
comment them out (`max_min_diff, max_slope, min_slope`), since i added them fairly recently, and they did not affect the submission on Feb 23 in any way.

If anything, the source code for those is:

    
    @set_property("fctype", "simple")
    def max_min_diff(x):
        # Calculation of feature as float, int or bool
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        f = np.max(x) - np.min(x)
        return f

    
    @set_property("fctype", "simple")
    def max_slope(x):
        # Calculation of feature as float, int or bool
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        try:
            f = np.max(np.diff(x))
        except ValueError:
            f = 0
        return f
    
    @set_property("fctype", "simple")
    def min_slope(x):
        # Calculation of feature as float, int or bool
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        try:
            f = np.min(np.diff(x))
        except ValueError:
            f = 0
        return f


2. Proceed by splitting the data into train and test, as well as augmenting the datasets further by calculating the encodings of select features:
that is, predict them from the rest of the input features, and append to the features matrix.

3. Run `train_stack_with_knn_enet.py` to train a dictionary of models for both targets, with a simpler meta-estimator on top of them.
4. Further in the competition, as i kept progressing on the public LB and as my main pipelines became more sophisticated, i started blending models 
in a semi-automatic way, using the following method:
    - run `importance_lgb.py` script to get the importances of variables using mean absolute SHAP value approach,
        saving the sorted lists of features in the `notebooks` folder
    - run `models/hyperopt_model_cv2018_rougher.py` or `models/hyperopt_model_cv2018_final.py` to find a set of optimal parameters for `LGBMRegressor()`
    for each of the targets through CV on all 2018 data
    - run `train_stack.py` (for `rougher.output.recovery`) or `train_stack_final.py` (for `final.output.recovery`)
    
5. I have also included a few more features in the dataset that calculate lag differences and higher order derivatives with a **lookback**, that you 
are free to examine in the `make_dataset.py` file. They allowed me to achieve the 4.68 on the public leaderboard, but did not seem to help much on private.
However, that does not mean they will be useless in the production case scenario, where the crossvalidation / holdout scheme as well as training methodology 
(online vs static batch as we did in the competition is used).  

6. Moreover, all the submissions can be checked in the `results` folder, as well as `subs.xlsx` where I kept track of all submissions since you guys wanted to 
pick the best out of all.
7. There's also a plentiful of notebooks with some experimental stacking codes, that were also used to create submissions midway in the competition, that are 
now basically deprecated and can be fully replaced with the methodology in p.3. You can find them under `notebooks`



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
