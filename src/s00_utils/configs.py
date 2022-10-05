import os
import json
import dataclasses


@dataclasses.dataclass
class Config:
    num_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    num_folds: int

    @classmethod
    def from_file(cls, file_path):
        """
        Loads config parameters from a file and returns a new config object.
        """
        with open(file_path, "r") as json_file:
            # Load the parameters from a json file as a dictionary
            config_dict = json.load(json_file)

            # Create the config and return it
            return cls(**config_dict)

    def to_file(self, file_path):
        """
        Saves config parameters to a file.
        """
        # Create necessary directories
        dir_path = os.path.split(file_path)[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Transform the config to a dictionary
        config_dict = dataclasses.asdict(self)

        # Save the dictionary to a json file
        with open(file_path, "w") as json_file:
            json.dump(config_dict, json_file)

    def as_dict(self):
        return json.dumps(dataclasses.asdict(self), indent=2)
