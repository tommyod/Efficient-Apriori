# New version to Pypi

First update the version number, then do (change `0.X.Y`).

```bash
$ python3 setup.py sdist bdist_wheel
$ twine upload --repository-url https://upload.pypi.org/legacy/ dist/efficient_apriori-0.X.Y*
```

Must have
`conda install "twine>=1.11.0"`,
`conda install "setuptools>=38.6.0"` and
`conda install wheel==0.31.0`
for markdown to be rendered correctly.

# Pushing tags to GitHub


```bash
$ git tag X.Y
$ git push origin X.Y
```

# Setting up conda environment for testing

```bash
$ conda create -n apriori python=3.6
$ source activate apriori
$ pip install efficient_apriori
$ python
$ conda remove -n apriori --all
```
