import numpy as np
import pandas as pd
from scipy.stats import kstest, shapiro, wilcoxon


class NormalityCheck:
    """Проверяем данныe на нормальность по критериям Шапиро 
    и Колмогорова-Смирнова."""

    def __init__(self, samples):
        self.samples = samples

    def hypo_check(self, row):
        """Функуия для закопления колонки сведениями о принятой или 
        отвергнутой гипотезе."""
        if row['p-value < alpha']:
            return 'Н1'
        return 'Н0'

    def shapiro(self):
        """Проверка данных на нормальность. Критерий Шапиро-Уилка."""

        result = []

        # проверим можно ли считать выборку сетевых заведений
        # нормально распределённой
        alpha = 0.05
        zero_hypothesis = 'Выборка сетевых заведений нормально распределена'
        alt_hypothesis = 'Распределение выборки сетевых заведений не нормально'
        chain_results = shapiro(self.samples[0])
        p_value = chain_results[1]

        result.append(('Сетевые заведения',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       p_value))

        # проверим можно ли считать выборку несетевых заведений
        # нормально распределённой
        alpha = 0.05
        zero_hypothesis = 'Выборка несетевых заведений нормально распределена'
        alt_hypothesis = ('Распределение выборки несетевых'
                          ' заведений не нормально')
        not_chained_results = shapiro(self.samples[1])
        p_value = not_chained_results[1]

        result.append(('Сетевые заведения',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       p_value))

        result = pd.DataFrame(self, result)
        result.columns = ['Выборка', 'Нулевая гипотеза (Н0)',
                          'Альтернативная гипотеза (Н1)', 'alpha', 'p-value']
        result['p-value < alpha'] = result['p-value'] < result['alpha']
        result['Н0/Н1'] = result.apply(self.hypo_check, axis=1)

        return result

    def kolmogorov_smirnov(self):
        """Проверка данных на нормальность. Критерий Колмогорова-Смирнова."""

        result = []

        # проверим можно ли считать выборку сетевых заведений
        # нормально распределённой
        alpha = 0.05
        zero_hypothesis = 'Выборка сетевых заведений нормально распределена'
        alt_hypothesis = 'Распределение выборки сетевых заведений не нормально'
        chain_results = kstest(self.samples[0], 'norm')
        p_value = chain_results[1]

        result.append(('Сетевые заведения',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       p_value))

        # проверим можно ли считать выборку несетевых заведений
        # нормально распределённой
        alpha = 0.05
        zero_hypothesis = 'Выборка несетевых заведений нормально распределена'
        alt_hypothesis = ('Распределение выборки несетевых'
                          ' заведений не нормально')
        not_chained_results = kstest(self.samples[1], 'norm')
        p_value = not_chained_results[1]

        result.append(('Сетевые заведения',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       p_value))

        result = pd.DataFrame(result)
        result.columns = ['Выборка', 'Нулевая гипотеза (Н0)',
                          'Альтернативная гипотеза (Н1)', 'alpha', 'p-value']
        result['p-value < alpha'] = result['p-value'] < result['alpha']
        result['Н0/Н1'] = result.apply(self.hypo_check, axis=1)

        return result


class Wilcoxon(NormalityCheck):
    """Применяем непараметрический статистический критерий 
    Уилкоксона для проверки гипотезы."""

    def __init__(self, chained, not_chained, full_data, num_limit):
        self.chained = chained
        self.not_chained = not_chained
        self.cumulated = full_data
        self.num_limit = num_limit

    def get_anomalies(self, chained, not_chained):
        """Получаем данные с аномалиями."""

        # Отберем выбросы по числу мест
        outliers_chained = (chained
                            .query(f'number > {self.num_limit}')
                            ['id'])
        outliers_not_chained = (not_chained
                                .query(f'number > {self.num_limit}')
                                ['id'])

        # Объединим выбросы
        anomalies = (pd.concat([outliers_chained, outliers_not_chained],
                               axis=0)
                     .drop_duplicates().sort_values())

        return anomalies

    def get_placeholders(self, data, chain_status):
        """Заполняем выборки нулями."""
        full_length = self.cumulated
        index = np.arange(full_length['id'].max() - len(data))
        placeholders = pd.Series(0, index=index, name='number')
        return placeholders

    def prepare_data_for_stat(self):
        """Готовим данные для статистических тестов."""

        # Получим плейсхолдеры для выравнивания выборок
        placeholders_chained = self.get_placeholders(self.chained, 'да')
        placeholders_not_chained = self.get_placeholders(
            self.not_chained, 'нет')

        # Дополняем сырые выборки с заведениями плейсхолдерами
        sample_raw_chained = pd.concat([self.chained['number'],
                                        placeholders_chained], axis=0)
        sample_raw_not_chained = pd.concat([self.not_chained['number'],
                                            placeholders_not_chained], axis=0)

        # Выявим аномалии
        anomalies = self.get_anomalies(self.chained, self.not_chained)

        # Выборки без аномалий
        clean_number_chained = self.chained.query(
            'id not in @anomalies')['number']
        clean_number_not_chained = self.not_chained.query(
            'id not in @anomalies')['number']

        # Дополняем выборки без аномалий с заведениями плейсхолдерами
        placeholders_chained = self.get_placeholders(
            clean_number_chained, 'да')
        placeholders_not_chained = self.get_placeholders(
            clean_number_not_chained, 'нет')
        sample_clean_chained = pd.concat(
            [clean_number_chained, placeholders_chained])
        sample_clean_not_chained = pd.concat(
            [clean_number_not_chained, placeholders_not_chained])

        prepared_data = {'sample_raw_chained': sample_raw_chained,
                         'sample_raw_not_chained': sample_raw_not_chained,
                         'sample_clean_chained': sample_clean_chained,
                         'sample_clean_not_chained': sample_clean_not_chained}

        return prepared_data

    def wilcoxon_stat(self):
        """Проводим статистические тесты, собираем результат."""

        result = []

        # Подготовим данные
        prepared_data = self.prepare_data_for_stat()

        # Считаем статистическую значимость различий
        # в среднем числе посадочных мест между заведениями по «сырым» данным
        alpha = 0.05
        zero_hypothesis = 'В числе посадочных мест нет различий'
        alt_hypothesis = 'В числе посадочных мест есть различия'
        raw_num_result = wilcoxon(prepared_data['sample_raw_chained'],
                                  prepared_data['sample_raw_not_chained'])

        result.append(('Среднее число мест по сырым',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       raw_num_result[1]))

        # Считаем статистическую значимость различий в среднем числе
        # посадочных мест между заведениями по «очищенным» данным
        alpha = 0.05
        zero_hypothesis = 'В числе посадочных мест нет различий'
        alt_hypothesis = 'В числе посадочных мест есть различия'
        clean_num_result = wilcoxon(prepared_data['sample_clean_chained'],
                                    prepared_data['sample_clean_not_chained'])
        result.append(('Среднее число мест по очищенным',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       clean_num_result[1]))

        result = pd.DataFrame(result)
        result.columns = ['Выборка', 'Нулевая гипотеза (Н0)',
                          'Альтернативная гипотеза (Н1)', 'alpha', 'p-value']
        result['p-value < alpha'] = result['p-value'] < result['alpha']
        result['Н0/Н1'] = result.apply(self.hypo_check, axis=1)

        return result
