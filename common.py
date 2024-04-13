from enum import Enum

class State(Enum):
    Immobile = "0"
    Confined = "1"
    Free = "2"
    Directed = "3"


class SegmentProperties:
    def __init__(self, K, alpha, state, endpoint):
        self.K = K
        self.alpha = alpha
        self.state = state
        self.endpoint = endpoint
    def __str__(self):
        return str(self.K) + ", " + str(self.alpha) + ", " + self.state.value + ", " + str(self.endpoint)