from datetime import time

import yaml

# Custom tag names for YAML
TIME_TAG = "!time"


def time_representer(dumper, data):
    return dumper.represent_scalar(TIME_TAG, data.isoformat())


def time_constructor(loader, node):
    value = loader.construct_scalar(node)
    return time.fromisoformat(value)


# Register representers
yaml.add_representer(time, time_representer)

# Register constructors
yaml.add_constructor(TIME_TAG, time_constructor)

# Register constructors for all common loaders
for Loader in [yaml.Loader, yaml.SafeLoader, yaml.FullLoader]:
    yaml.add_constructor(TIME_TAG, time_constructor, Loader)
