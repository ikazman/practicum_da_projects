import pandas as pd
from bayesian_testing.experiments import BinaryDataTest, DeltaLognormalDataTest

from .stat_tests import MannWhitneyU


class BayesianTesting(MannWhitneyU):

    def bayesian_tests(self, data, data_samples):
        """Подсчитываем вероятность быть лучшей."""
        result = []

        for sample in data_samples:
            sample_title = sample[0]
            group_a = data[sample[1]]
            group_b = data[sample[2]]

            # инициализируем тест
            test = DeltaLognormalDataTest()

            # добавляем варианты для тестирования
            test.add_variant_data('A', group_a, replace=True)
            test.add_variant_data('B', group_b, replace=True)

            # получаем результат, фиксируя случайные числа
            test_result = test.probabs_of_being_best(seed=42)

            result.append((sample_title, test_result['A'], test_result['B']))

        return result

    def result(self):
        """Подготовим отчет о результатах тестирования."""

        # соберем данные для тестирования
        prepared_data = self.prepare_data_for_stat()

        # подготовим наборы данных
        data_samples = [
            ('Конверсия по сырым', 'sample_raw_a', 'sample_raw_b'),
            ('Конверсия по очищенным', 'sample_clean_a', 'sample_clean_b'),
            ('Средний чек по сырым', 'revenue_a', 'revenue_b'),
            ('Средний чек по очищенным', 'clean_rev_a', 'clean_rev_b')
        ]

        result = self.bayesian_tests(prepared_data, data_samples)

        result = pd.DataFrame(result)
        result.columns = ['Выборка',
                          'Вероятность, что лучше А (%)',
                          'Вероятность, что лучше В (%)']
        result['Вероятность, что лучше А (%)'] = round(
            result['Вероятность, что лучше А (%)'] * 100, 2)
        result['Вероятность, что лучше В (%)'] = round(
            result['Вероятность, что лучше В (%)'] * 100, 2)

        return result
