import yaml

DEFAULT_HPARAMS = {}
class BaselineParamsParser:
    def __init__(self, path_to_params):
        self.path_to_params = path_to_params
        self.hparams = DEFAULT_HPARAMS

    def parse_params(self):
        with open(self.path_to_params, 'rb') as f:
            params_from_file = yaml.safe_load(f)
        self.hparams.update(params_from_file)
        return self.hparams

