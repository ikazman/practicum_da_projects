import pandas as pd


class HypoPrioritization:
    """Приоритезируем гипотезы по медотам ICE и RICE."""

    def __init__(self, hypothesis):
        self.hypothesis = pd.read_csv(hypothesis)
        self.scores = self.hypothesis.copy()

    def score(self):
        """Приоретизация по методу ICE и RICE."""
        reach = self.scores['Reach']
        impact = self.scores['Impact']
        confidence = self.scores['Confidence']
        efforts = self.scores['Efforts']

        self.scores['ICE'] = round(impact * confidence / efforts, 2)
        self.scores['RICE'] = round(reach * impact * confidence / efforts, 2)

    def get_priority(self):
        self.score()
        return self.scores
