import numpy as np
from cobaya.likelihood import Likelihood

class Phantom:
    def __init__(self):
        pass

    @staticmethod
    def is_phantom(w, wa):
        z = np.linspace(0, 1100)
        a = 1 / (z + 1)
        wDE = w + wa * (1 - a)
        for i in wDE:
            if i < -1:
                return True  # Trovato almeno un elemento negativo
        return False  # Nessun elemento negativo trovato

    @staticmethod
    def loglike(w, wa):
        if Phantom.is_phantom(w, wa):
            chi2 = np.inf
        else:
            chi2 = 0
        #print('got w0=',w, 'and wa=', wa, 'loglike:',-1 / 2 * chi2)
        return -1 / 2 * chi2

class Exclude_Phanthom(Likelihood):
    name: str = "phantomLike"

    def initialize(self):
        pass

    def get_requirements(self):
        """
        return dictionary specifying quantities calculated by a theory code are needed
        """
        return {'w': None, 'wa': None}
        #return {'w0_fld': None, 'wa_fld': None}

    def logp(self, **params_values):
        like = Phantom()

        w0 = self.provider.get_param("w")
        wa = self.provider.get_param("wa")
        #w0 = self.provider.get_param("w0_fld")
        #wa = self.provider.get_param("wa_fld")

        logp = like.loglike(w=w0, wa=wa)
        #print('got w0=',w0, 'and wa=', wa, 'loglike:',logp)
        return logp

