import numpy as np
from keras.saving import load_model


class CPDetector:

    def __init__(self, model_path, window_length=15):
        self.model = load_model(model_path)
        self.window_length = window_length

    def _convert_to_windows(self, trajectory):
        """
        Split trajectory into windows of size self.window_length
        :param trajectory: numpy.ndarray, array of (x,y) coordinates
        """
        X = []
        for i in range(len(trajectory) - self.window_length):
            X.append(trajectory[i:i+self.window_length, :])
        return np.array(X)
    
    def get_segments(self, prob_seq, threshold=0.5):
        """
        Get the segement of subsequent frames for which probability of change point is greater than threshold.
        """
        segments = []
        segment_now = False
        new_segment = []
        for i in range(len(prob_seq)):
            if prob_seq[i] > threshold:
                if not segment_now:
                    segment_now = True
                    new_segment = [i]
                else:
                    new_segment.append(i)
            else:
                if segment_now:
                    segment_now = False
                    segments.append(new_segment)
        return segments

    def get_change_points_preds(self, prob_seq) -> list:
        """
        Return change points given a sequence of probabilities - choose frame with highest probability.
        """
        change_points = []
        offset = self.window_length // 2
        segments = self.get_segments(prob_seq)
        for segment in segments:
            prob_seq_subset = prob_seq[segment]
            max_idx = np.argmax(prob_seq_subset)
            change_points.append(segment[max_idx] + offset)
        return change_points

    def predict(self, trajectory) -> list:
        if (trajectory.shape[0] <= self.window_length):
            return [trajectory.shape[0]]
        windows = self._convert_to_windows(trajectory)
        prob_seq = self.model.predict(windows).flatten()
        return self.get_change_points_preds(prob_seq) + [trajectory.shape[0]]

    def __call__(self, trajectory):
        return self.predict(trajectory)
