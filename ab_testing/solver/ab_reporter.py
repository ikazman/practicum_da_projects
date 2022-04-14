import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import seaborn as sns


class ABReporter:
    """Создаем профили пользователя, расчитываем и визуализируем метрики."""

    def __init__(self, visitors, orders):
        self.visitors = pd.read_csv(visitors, parse_dates=['date'])
        self.orders = pd.read_csv(orders, parse_dates=['date'])

    def columns_fixer(self):
        """Приводим колонки к одному регистру, переименовываем по
        необходимости, конвертируем формат."""
        datasets = [self.visitorss, self.orders]

        for dataset in datasets:
            dataset.columns = [name.lower().replace(' ', '_') for name
                               in dataset.columns.values]

    def cumulate_column(self, df, column):
        """Cуммируем элементы колонки с накоплением."""
        grouped_by_a = df[df['group'] == 'A'][column].cumsum()
        grouped_by_b = df[df['group'] == 'B'][column].cumsum()
        cumulated = pd.concat([grouped_by_a, grouped_by_b]).reset_index()
        cumulated.sort_values(by='index', inplace=True)
        return cumulated.set_index('index')

    def grouped_summary(self):
        """Получаем сводную таблицу посетителей и заказов."""
        columns = ['date', 'group', 'orders', 'buyers', 'revenue', 'visitors']
        visitors = (self.visitors
                    .groupby(['date', 'group'], as_index=False)
                    .agg({'date': 'max',
                          'group': 'max',
                          'visitors': 'sum'}, axis=1))

        orders = (self.orders
                  .groupby(['date', 'group'], as_index=False)
                  .agg({'date': 'max',
                        'group': 'max',
                        'transactionId': 'nunique',
                        'visitorId': 'nunique',
                        'revenue': 'sum'}, axis=1))

        visitors['visitors'] = self.cumulate_column(visitors,'visitors')
        orders['revenue'] = self.cumulate_column(orders,'revenue')
        orders['transactionId'] = self.cumulate_column(orders, 'transactionId')
        orders['visitorId'] = self.cumulate_column(orders, 'visitorId')

        result = orders.merge(visitors)
        result.columns = columns

        return result

