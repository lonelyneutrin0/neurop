r""" 
.. include:: ../README.md
""" 

__version__ = "0.0.5"
__authors__ = [
    "Hrishikesh Belagali",
    "Aditya Narayan"
]
__author_emails__ = [
    "belagal1@msu.edu",
    "ma24btech11001@iith.ac.in"
]
__url__ = "https://github.com/lonelyneutrin0/neurop"

from .base import NeuralOperator as NeuralOperator

from . import operators as operators
from . import layers as layers
