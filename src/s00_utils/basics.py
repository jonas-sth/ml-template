"""This module contains basic methods that are used throughout this project."""


def markdown(string: str):
    """
    Formats linebreaks and indentation to Markdown format.
    """
    formatted_string = string.replace(" ", "&nbsp;&nbsp;").replace("\n", "<br />")
    return formatted_string
