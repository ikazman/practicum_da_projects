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

    def sorted_scores(self, col='ICE'):
        """Выравниваем наименование колонки с текстом и текст по левому краю."""
        self.styled_scores = self.scores.copy()
        self.styled_scores.sort_values(by=col, ascending=False, inplace=True)
        self.styled_scores = self.styled_scores.style.set_table_styles(
            [dict(selector='th', props=[('text-align', 'left')])])
        self.styled_scores = self.styled_scores.style.set_table_styles(
            [dict(selector='th', props=[('text-align', 'left')])])
        return self.styled_scores

    def get_priority(self):
        self.score()
        return self.scores
