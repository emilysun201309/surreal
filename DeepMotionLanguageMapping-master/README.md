# Installation

For the most part, installing the necessary dependencies can be done using pip:

```
pip install -r requirements.txt
```

However, there are some additional dependencies that need some extra work.

## pyMMM
`pyMMM` is used to convert from and to the MMM representation.
Note that this step is only necessary to visualize motion and for some advanced features of this release.
Most of functionality should also be available without it.

pyMMM can be installed as follows:

```
vendor/install_pymmm.sh
```

pyMMM does require that you have [simox](https://gitlab.com/Simox/simox), [MMMCore](https://gitlab.com/mastermotormap/mmmcore/tree/master) and [MMMTools](https://gitlab.com/mastermotormap/mmmtools/tree/master) installed.
All come with extensive installation documentations ([Simox installation instructions](), [MMMCore and MMMTools installation instructions](https://gitlab.com/Simox/simox/wikis/Installation)).

# Usage

- `data_import.py` takes a release of the [KIT Motion-Language Dataset](https://motion-annotation.humanoids.kit.edu/dataset/) and parses the necessary files.
- `data_process.py` takes the data processed by `data_import.py` and produces a dataset that is ready to be used for training models.
- `model_language2motion.py` trains the language to motion model using the dataset produced by `data_process.py`. It also contains code to evaluate a trained model.
- `model_motion2language.py` trains the motion to language model using the dataset produced by `data_process.py`. It also contains code to evaluate a trained model.
- A bunch of additional helper scripts exist that can be used to visualize certain aspects of the models (`visualize_*.py`).

For all scripts, you can see all available arguments by passing in `--help` as an option, e.g. `python data_import.py --help`.
The argument should are clearly named and should hopefully be self-explanatory.

Python 2.7 on Ubuntu 16.04 was used for all experiments and the source code is compatible with that setup.
