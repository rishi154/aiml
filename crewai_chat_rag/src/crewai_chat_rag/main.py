#!/usr/bin/env python
import sys
import warnings

from crewai_chat_rag.crew import CrewaiChatRag

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    customer_question = {
        "customer_question":"Who is US current president?"
    }
    CrewaiChatRag().crew().kickoff(inputs=customer_question)
