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
        self.cumulated = None

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

        result['conversion'] = result['orders'] / result['visitors']

        self.cumulated = result

        return result

    def plot_cumulative_metrics(self):
        """Функция для визуализации кумулятивных метрик."""
        plt.figure(figsize=(15, 10))

        columns_to_pick = ['date','revenue', 'orders']
        revenue_a = self.cumulated.query('group == "A"')[columns_to_pick]
        revenue_b = self.cumulated.query('group == "B"')[columns_to_pick]

        # кривые удержания платящих пользователей
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(revenue_a['date'], revenue_a['revenue'], grid=True, label='A', ax=ax1)
        plt.plot(revenue_b['date'], revenue_b['revenue'], grid=True, label='B', ax=ax1)
        plt.legend()
        plt.xlabel('Лайфтайм')
        plt.title('Удержание платящих пользователей')

        # кривые удержания неплатящих
        ax2 = plt.subplot(2, 2, 2, sharey=ax1)
        retention.query('payer == False').droplevel('payer').T.plot(
            grid=True, ax=ax2)
        plt.legend()
        plt.xlabel('Лайфтайм')
        plt.title('Удержание неплатящих пользователей')

        # динамика удержания платящих
        ax3 = plt.subplot(2, 2, 3)
        columns = [name for name in retention_hist.index.names
                   if name not in ['dt', 'payer']]
        filtered_data = retention_hist.query('payer == True').pivot_table(
            index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
        filtered_data = self.filter_data(filtered_data, window)
        filtered_data.plot(grid=True, ax=ax3, sharey=ax1)
        plt.xlabel('Дата привлечения')
        plt.title('Динамика удержания платящих '
                  f'пользователей на {horizon}-й день')

        # динамика удержания неплатящих
        ax4 = plt.subplot(2, 2, 4, sharey=ax3)
        filtered_data = retention_hist.query('payer == False').pivot_table(
            index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
        self.filter_data(filtered_data, window).plot(grid=True, ax=ax4)
        plt.xlabel('Дата привлечения')
        plt.title('Динамика удержания неплатящих '
                  f'пользователей на {horizon}-й день')

        plt.tight_layout()
        plt.show()