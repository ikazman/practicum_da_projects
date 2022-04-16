import pandas as pd
import numpy as np
import scipy.stats as stats


class MannWhitneyU():

    def __init__(self, visitors, orders, cumulated):
        self.visitors = visitors
        self.orders = orders
        self.cumulated = cumulated

    def get_anomalies(self, orders_by_a, orders_by_b):
        """Получаем данные с аномалиями."""

        # Отберем выбросы по заказам
        outliers_orders_a = orders_by_a.query('orders > 2')['visitorId']
        outliers_orders_b = orders_by_b.query('orders > 2')['visitorId']
        outliers_orders = pd.concat([outliers_orders_a,
                                     outliers_orders_b], axis=0)

        # Отберем выбросы по выручке
        outliers_revenue = self.orders.query('revenue > 28000')['visitorId']

        # Объединим выбросы
        anomalies = (pd.concat([outliers_orders, outliers_revenue], axis=0)
                     .drop_duplicates().sort_values())

        return anomalies

    def get_placeholders(self, data, group):
        """Заполняем выборки нулями."""
        full_length = self.cumulated.query(f'group == "{group}"')['visitors']
        index = np.arange(full_length.sum() - len(data))
        placeholders = pd.Series(0, index=index, name='orders')
        return placeholders

    def prepare_data_for_stat(self):
        """Готовим данные для статистических тестов."""

        # Сгруппируем данные по группам
        orders_by_a = (self.orders.query('group == "A"')
                                  .groupby('visitorId', as_index=False)
                                  .agg({'transactionId': 'nunique'}))
        orders_by_b = (self.orders.query('group == "B"')
                                  .groupby('visitorId', as_index=False)
                                  .agg({'transactionId': 'nunique'}))
        orders_by_a.columns = ['visitorId', 'orders']
        orders_by_b.columns = ['visitorId', 'orders']

        # Получим плейсхолдеры для пользователей без заказов
        placeholders_a = self.get_placeholders(orders_by_a, 'A')
        placeholders_b = self.get_placeholders(orders_by_b, 'B')

        # Сырые выборки с заказами
        sample_raw_a = pd.concat([orders_by_a['orders'],
                                  placeholders_a], axis=0)
        sample_raw_b = pd.concat([orders_by_b['orders'],
                                  placeholders_b], axis=0)

        # Сформируем данные с выручкой с аномалиями
        revenue_a = self.orders.query('group =="A"')['revenue']
        revenue_b = self.orders.query('group =="B"')['revenue']

        # Выявим аномалии
        anomalies = self.get_anomalies(orders_by_a, orders_by_b)

        # Выборки с выручкой  без аномалий
        clean_revenue_a = self.orders.query(
            'group == "A" and visitorId not in @anomalies')['revenue']
        clean_revenue_b = self.orders.query(
            'group == "B" and visitorId not in @anomalies')['revenue']

        # Выборки с заказами без аномалий
        cleaned_by_a = orders_by_a.query(
            'visitorId not in @anomalies')['orders']
        cleaned_by_b = orders_by_b.query(
            'visitorId not in @anomalies')['orders']

        # Выборки с выручкой без аномалий
        sample_clean_a = pd.concat([cleaned_by_a, placeholders_a])
        sample_clean_b = pd.concat([cleaned_by_b, placeholders_b])

        prepared_data = {'sample_raw_a': sample_raw_a,
                         'sample_raw_b': sample_raw_b,
                         'revenue_a': revenue_a,
                         'revenue_b': revenue_b,
                         'clean_rev_a': clean_revenue_a,
                         'clean_rev_b': clean_revenue_b,
                         'sample_clean_a': sample_clean_a,
                         'sample_clean_b': sample_clean_b}

        return prepared_data

    def hypo_check(self, row):
        """Функуия для закопления колонки сведениями о принятой или 
        отвергнутой гипотезе."""
        if row['p-value < alpha']:
            return 'Н1'
        return 'Н0'

    def mannwhitneyu(self):
        """Проводим статистические тесты, собираем результат."""
        result = []

        # Подготовим данные
        prepared_data = self.prepare_data_for_stat()

        # Считаем статистическую значимость различий
        # в конверсии между группами по «сырым» данным
        alpha = 0.05
        zero_hypothesis = 'В конверсии между группами нет различий'
        alt_hypothesis = 'В конверсии между группами есть различия'

        raw_conv_result = stats.mannwhitneyu(prepared_data['sample_raw_a'],
                                             prepared_data['sample_raw_b'])[1]
        result.append(('Конверсия по сырым',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       raw_conv_result))

        # Считаем статистическую значимость различий
        # в среднем чеке заказа между группами по «сырым» данным
        alpha = 0.05
        zero_hypothesis = 'Отличий в среднем чеке между группами нет'
        alt_hypothesis = 'Отличия в среднем чеке между группами есть'

        raw_revenue_result = stats.mannwhitneyu(prepared_data['revenue_a'],
                                                prepared_data['revenue_b'])[1]
        result.append(('Средний чек по сырым',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       raw_revenue_result))

        # Считаем статистическую значимость различий
        # в конверсии между группами по «очищенным» данным
        alpha = 0.05
        zero_hypothesis = 'В конверсии между группами нет различий'
        alt_hypothesis = 'В конверсии между группами есть различия'

        clean_conv_result = stats.mannwhitneyu(
            prepared_data['sample_clean_a'], prepared_data['sample_clean_b'])[1]
        result.append(('Конверсия по очищенным',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       clean_conv_result))

        # Считаем статистическую значимость различий
        # в среднем чеке заказа между группами по «очищенным» данным
        alpha = 0.05
        zero_hypothesis = 'Отличий в среднем чеке между группами нет'
        alt_hypothesis = 'Отличия в среднем чеке между группами есть'

        clean_rev_result = stats.mannwhitneyu(prepared_data['clean_rev_a'],
                                              prepared_data['clean_rev_b'])[1]
        result.append(('Средний чек по очищенным',
                       zero_hypothesis,
                       alt_hypothesis,
                       alpha,
                       clean_rev_result))

        result = pd.DataFrame(result)
        result.columns = ['Выборка', 'Нулевая гипотеза (Н0)',
                          'Альтернативная гипотеза (Н1)', 'alpha', 'p-value']
        result['p-value < alpha'] = result['p-value'] < result['alpha']
        result['Н0/Н1'] = result.apply(self.hypo_check, axis=1)

        return result
