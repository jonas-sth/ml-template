"""This module contains basic methods that are used throughout this project."""


def markdown(string: str):
    """
    Formats linebreaks and indentation to Markdown format.
    """
    formatted_string = string.replace(" ", "&nbsp;&nbsp;").replace("\n", "<br />")
    return formatted_string


def weight_reset(m):
    """
    Resets all weights in a model.
    Use: model.apply(weight_reset)
    """
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
