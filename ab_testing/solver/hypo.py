import pandas as pd


class HypoPrioritization:
    """Приоритезируем гипотезы по медотам ICE и RICE."""

    def __init__(self, hypothesis):
        self.hypothesis = pd.read_csv(hypothesis)
        self.scores = self.hypothesis.copy()
        self.styled_scores = None

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

    def make_pretty(self, styler, subset):
        styler.background_gradient(axis=1,
                                   vmin=styler.min(), vmax=styler.max(),
                                   cmap="YlGnBu",
                                   subset=[subset])
        return styler
