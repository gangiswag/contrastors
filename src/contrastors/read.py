import yaml
from contrastors.config import Config
from pydantic import ValidationError


def read_config(path):
    # read yaml and return contents
    with open(path, 'r') as file:
        try:
            return Config(**yaml.safe_load(file))
        except yaml.YAMLError as exc:
            print(exc)
        except ValidationError as exc:
            print(f"Validation error: {exc}")
            for error in exc.errors():
                print(f"Error in field '{error['loc']}': {error['msg']}")
