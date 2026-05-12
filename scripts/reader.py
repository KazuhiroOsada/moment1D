import os

import numpy as np


class reader:
    PARAMETER_FILE = "parameter.txt"
    ITER_DIGITS = 5
    MOMENT_COMPONENTS = 5
    FIELD_COMPONENTS = 6

    def __init__(self, prefix="../results/"):
        self.prefix = prefix
        self.parameters = {}
        self.problem_name = None
        self.Nx = None
        self.N_MOMENT = self.MOMENT_COMPONENTS
        self.N_FIELD = self.FIELD_COMPONENTS

    def read_parameters(self):
        file_path = os.path.join(self.prefix, self.PARAMETER_FILE)
        params = {}
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, value = line.split(maxsplit=1)
                params[key] = self._parse_parameter_value(key, value)

        self.parameters = params
        for key, value in params.items():
            setattr(self, key, value)
        return params

    def read_moment(self, name, it):
        nx = self._require_nx()
        file_path = os.path.join(self.prefix, f"{name}_{it:0{self.ITER_DIGITS}d}.dat")
        with open(file_path, "rb") as f:
            U = np.fromfile(f, dtype=np.float64).reshape(self.N_MOMENT, nx)
        return U

    def read_field(self, it):
        nx = self._require_nx()
        file_path = os.path.join(self.prefix, f"fields_{it:0{self.ITER_DIGITS}d}.dat")
        with open(file_path, "rb") as f:
            U = np.fromfile(f, dtype=np.float64).reshape(self.N_FIELD, nx)
        return U

    def _require_nx(self):
        if self.Nx is None:
            self.read_parameters()
        return self.Nx

    @staticmethod
    def _parse_parameter_value(key, value):
        if key == "problem_name":
            return value
        if key in {"Nx", "N_ghost", "Nx_total", "lb", "ub", "N_MOMENT", "N_FIELD", "max_iters", "diag_interval"}:
            return int(value)
        if key in {"gamma", "c", "dt"}:
            return float(value)
        return value
