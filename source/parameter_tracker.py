import numpy as np


class ParameterTracker:
    def __init__(self):
        self.last_parameter = None
        self.log = []

    def track(self, parameter):
        parameter = parameter.reshape(-1).copy()
        param_extent = np.sqrt(np.average(np.sum(parameter ** 2)))
        if self.last_parameter is None:
            self.last_parameter = parameter
            self.log.append((0, param_extent))
        else:
            change = np.sum(np.abs(self.last_parameter - parameter))
            change = np.log1p(change)
            self.last_parameter = parameter
            self.log.append({"change": change, "weight-sum": param_extent})

    def __str__(self):
        return "\n".join([f'{i}: {each}' for i, each in enumerate(self.log)])
