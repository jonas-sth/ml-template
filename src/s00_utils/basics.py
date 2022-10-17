"""This module contains basic methods that are used throughout this project."""


def markdown(string: str):
    """
    Formats linebreaks and indentation to Markdown format.
    """
    if string is not None:
        formatted_string = string.replace(" ", "&nbsp;&nbsp;").replace("\n", "<br />")
    else:
        formatted_string = "None"
    return formatted_string


def weight_reset(m):
    """
    Resets all weights in a model.
    Use: model.apply(weight_reset)
    """
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_lr_scheduler_params(lr_scheduler):
    """
    Returns a dictionary with the parameters of the given learning rate scheduler without the running parameters.
    """
    # Get all parameters
    params = lr_scheduler.state_dict()

    # Remove the running parameters
    params.pop("base_lrs")
    params.pop("last_epoch")
    params.pop("_step_count")
    params.pop("verbose")
    params.pop("_get_lr_called_within_step")
    params.pop("_last_lr")

    return params
