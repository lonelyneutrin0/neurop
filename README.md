# neurop 

[![Custom shields.io](https://img.shields.io/badge/docs-brightgreen?logo=github&logoColor=green&label=gh-pages)](https://lonelyneutrin0.github.io/neurop/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/neurop.svg)](https://pypi.python.org/pypi/neurop/)
[![PyPI pyversions shields.io](https://img.shields.io/pypi/pyversions/neurop.svg)](https://pypi.python.org/pypi/neurop/)

## About neurop 
`neurop` is a Python package implementing [Neural Operators](https://en.wikipedia.org/wiki/Neural_operators#:~:text=Neural%20operators%20directly%20learn%20operators,be%20evaluated%20at%20any%20discretization.), which are a class of operating learning models. The package currently supports the following operator architectures -
- Fourier Neural Operator (FNO)
- Deep Operator Network (DeepONet)
- Complex Neural Operator (CoNO)

## Usage
`neurop` is published on [PyPi](https://pypi.python.org/pypi/neurop/) and can be installed using `pip` - 
```
pip install neurop
```

## License
`neurop` is available under the MIT License.

## Attribution
This project contains an independent implementation of the technique described in:

Karn Tiwari, N M Anoop Krishnan, Prathosh A P, "CoNO: Complex Neural Operator for Continuous Dynamical Systems," arXiv, 2023.  
Inspired by the code from [this repo](https://github.com/M3RG-IITD/Complex-Neural-Operator/tree/main), but not based on or derived from it.

## Contributing
`neurop` is 100% open-source and welcomes contributions! Please open an issue to suggest new features (perhaps new operators 👀) or bug fixes. 
<br/> <br/> 
You can install developer dependencies using 
```
pip install neurop[dev]
```
Type-checking, import checking and test cases can be executed using
```
mypy neurop/
ruff check neurop/
pytest tests/
```
within the root directory of the project. 
