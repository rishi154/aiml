#!/usr/bin/env python
import sys
import warnings

from codegen.crew import Codegen

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    inputs = {
        'language': 'Python',
        'instructions' : "Generate a Python Flask-based REST API for managing Customer.\n"
                         "The API should support the following features:\n"
                         "1. List all Customers.\n"
                         "2. Get a specific customer by customerId.\n"
                         "3. Create new customer.\n"
                         "4. Update an existing customer.\n"
                         "5. Soft delete a customer.\n"
                         "Use MySQL as database\n"
                         "The API should return JSON responses and include appropriate HTTP status codes."
    }
    Codegen().crew().kickoff(inputs=inputs)