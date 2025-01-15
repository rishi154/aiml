#!/usr/bin/env python
import sys
import warnings

from synthetic_data_generator.crew import SyntheticDataGenerator

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'schema': "{\"postOfficeBox\":{\"type\":\"string\",\"min_length\":\"5\",\"max_lenght\":\"7\"},\"extendedAddress\":{\"type\":\"string\"},\"streetAddress\":{\"type\":\"string\",\"min_length\":\"10\",\"max_lenght\":\"20\"},\"locality\":{\"type\":\"string\"},\"region\":{\"type\":\"string\"},\"postalCode\":{\"type\":\"string\"},\"countryName\":{\"type\":\"string\"},\"creditCardNumber\":{\"cardType\":\"visa\",\"length\":\"16 or 19\"}}"
    }

    SyntheticDataGenerator().crew().kickoff(inputs=inputs)