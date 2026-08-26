import numpy as np

class Optimizer:
    def __init__(self, params: list):
        self.params = params
        self._pgms = []
        self._set_pgm()
        self.t = 0

    def _set_pgm(self):
        raise NotImplementedError(f"{self.__class__.__name__}._set_pgmは継承先で使用されます.")

    def _increment_t(self):
        self.t += 1

    def cleargrad(self):
        for param in self.params:
            param.cleargrad()

class ParamGradMemorizer:
    def __repr__(self):
        return f"PGM({self.__dict__})"
    
class Adam(Optimizer):
    def __init__(self, params: list):
        super().__init__(params)

    def _set_pgm(self):

        for param in self.params:
            pgm = ParamGradMemorizer()
            pgm.m = np.zeros_like(param.data)
            pgm.v = np.zeros_like(param.data)
            self._pgms.append(pgm)

    def step(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=10e-8):

        self._increment_t()

        for i, param in enumerate(self.params):
            
            pgm = self._pgms[i]

            pgm.m = beta1*pgm.m + (1 - beta1)*param.grad
            pgm.v = beta2*pgm.v + (1 - beta2)*param.grad**2

            m = pgm.m / (1 - beta1**self.t)
            v = pgm.v / (1 - beta2**self.t)

            param.data = param.data - alpha*m / (np.sqrt(v) + epsilon)

        self.cleargrad()

class SGD(Optimizer):
    def __init__(self, params):
        super().__init__(params)

    def _set_pgm(self):
        return

    def step(self, alpha=0.001):
        self._increment_t()

        for param in self.params:
            param.data = param.data - alpha*param.grad

        self.cleargrad()