Mave
====

Measurement and verification tool for estimating building energy performance

Installation
============

To install the tool on your machine, start by installing Python 2.7, virtualenv and python-pip. Then,

```
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

Install client package dependencies with [bower](http://bower.io/). (requires [npm](https://www.npmjs.com/package/npm))
```
bower install
```

Usage of the command line tool
==============================
Help (explanation of the available command line arguments)

The command line script is in `./mave/misc/` and can be used independently of the app or library.

```$ python mave_script.py -h```

Simplest case - read in a file with energy consumption data, and make a (blind) prediction on the last one third of the file
```
python mave_script.py InputFile.csv
```

Use 4 cores for training
```
$ python mave_script.py InputFile.csv -n 4
```
Use all available cores for training
```
$ python mave_script.py InputFile.csv -n -1
```

Increase the amount of computational time spent training by factor of 10
```
$ python mave_script.py InputFile.csv -c 10.0
```
Reduce the amount of computational time spent training by factor of 10
```
$ python mave_script.py InputFile.csv -c 0.1
```

Display verbose feedback, save detailed outputs, and plot a random sample of data
```
$ python mave_script.py InputFile.csv -v -s -p
```
