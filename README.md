# Quantifying Context Effects in a Survey with Hierarchical Modeling

We use a Bayesian Hierarchical Model for modeling discrete choice in a similarity setting.


### Modules

- db
- models
    - metric_models
    - freq_models
    - bayes_models
- utils
    data_types
- synthetic_data




*In our context, the synthetic data is used for model validation, not to improve training.*


## installation

We use SQLite for managing the datasets of this project. It was tested on version 3.49.1.

If you are using OSX, you can install `sqlite3` through Homebrew, by running

```
brew install sqlite
``` 

Due to PyMC, we recommend using conda and conda environments for installing the required Python libraries.

Per the [PyMC documentation](https://www.pymc.io/projects/docs/en/latest/installation.html), start with

```
conda create -c conda-forge -n pymc_env "pymc>=5"
conda activate pymc_env
```

Finally, to install the other required Python libraries, perform

```
pip install -r requirements.txt
```

Currently, the `db/similarity.db` file contains the populated SQLite database. If you want to recreate it from scratch, delete the file and run `python -m main` from the `src` directory. You need to request the `raw_data` folder to [@hugosc](https://github.com/hugosc) to be able to run it.
