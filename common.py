from enum import Enum

class State(Enum):
    Immobile = "0"
    Confined = "1"
    Free = "2"
    Directed = "3"


class Model(Enum):
    SINGLE_STATE = 'single_state',
    MULTI_STATE = 'multi_state',
    IMMOBILE_TRAPS ='immobile_traps',
    DIMERIZATION = 'dimerization',
    CONFINEMENT = 'confinement'
    def __str__(self):
        return str(self.value[0])
    def get_model_by_id(id_num):
        if id_num == 0:
            return Model.SINGLE_STATE
        if id_num == 1:
            return Model.MULTI_STATE
        if id_num == 2:
            return Model.IMMOBILE_TRAPS
        if id_num == 3:
            return Model.DIMERIZATION
        if id_num == 4:
            return Model.CONFINEMENT
    


class EnsembleResult():
    def __init__(self, model, alphas_mean, alphas_var, d_mean, d_var, weights):
        self.model = model
        self.alphas_mean  = alphas_mean
        self.alphas_var = alphas_var
        self.d_mean = d_mean
        self.d_var = d_var
        self.weights = weights
    def get_number_of_states(self):
        return len(self.alphas_mean)
    


class SegmentProperties:
    def __init__(self, K, alpha, state, endpoint, length):
        self.K = K
        self.alpha = alpha
        self.state = state
        self.endpoint = endpoint
        self.length = length
    def __str__(self):
        return str(self.K) + ", " + str(self.alpha) + ", " + self.state.value + ", " + str(self.endpoint)