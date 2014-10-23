bpe
===

Tools for estimating building energy performance


Requires:
Python 2.7

Package dependencies:
numpy
scipy
scikit-learn

Install these using the following commands from the command line:
```
$ pip install -U numpy
$ pip install -U scipy
$ pip install -U scikit-learn
```

Usage
===============================
Help (explanation of the available command line arguments)

```$ python bpe.py -h```

Simplest case - read in a file with energy consumption data, and make a (blind) prediction on the last one third of the file
```
python bpe.py InputFile.csv
```

Use 4 cores for training
```
$ python bpe.py InputFile.csv -n 4
```
Use all available cores for training
```
$ python bpe.py InputFile.csv -n -1
```

Increase the amount of computational time spent training by factor of 10
```
$ python bpe.py InputFile.csv -c 10.0
```
Reduce the amount of computational time spent training by factor of 10
```
$ python bpe.py InputFile.csv -c 0.1
```

Display verbose feedback, save detailed outputs, and plot a random sample of data
```
$ python bpe.py InputFile.csv -v -s -p
```
