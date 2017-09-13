# Kot or Not

Kot or Not (from Russian word Кот, a Cat, wordplay) project is an implementation
of a [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
algorithm using `numpy` for matrix arithmetics.

Algorithm uses [`sigmoid`](https://en.wikipedia.org/wiki/Sigmoid_function)
function, forward and backward propagations as well as vectorization techniques.


## Data

Current implementation has training set with cardinality 209. Images scaled
down to `64 x 64 x 3` and processed as a feature vector `(64 * 64 * 3, 1)`.


## Usage

```
usage: main.py [-h] [-d] [-r] [-i INPUT]

Cat or not neural network

optional arguments:
  -h, --help            show this help message and exit
  -d, --dump            use model dump
  -r, --retrain         retrain model
  -i INPUT, --input INPUT
                        path to input image
```

To classify your image first you need to train a model

```
python3 main.py -r
```

The model will be dumped into a flat text files: `weights` column vector
and `bias` literal.

After training `--dump` can be used together with `--input` image path

```
python3 main.py -d -i images/moon.jpg
```

Baby can Moon is certainly a cat, and `kot-or-not` NN things the same.