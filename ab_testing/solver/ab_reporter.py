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

        visitors['visitors'] = self.cumulate_column(visitors, 'visitors')
        orders['revenue'] = self.cumulate_column(orders, 'revenue')
        orders['transactionId'] = self.cumulate_column(orders, 'transactionId')
        orders['visitorId'] = self.cumulate_column(orders, 'visitorId')

        result = orders.merge(visitors)
        result.columns = columns

        result['conversion'] = result['orders'] / result['visitors']

        self.cumulated = result

        return result

    def plot_cumulative_metrics(self):
        """Функция для визуализации кумулятивных метрик."""
        plt.figure(figsize=(25, 10))
        plt.style.use('seaborn-darkgrid')

        columns_to_pick = ['date', 'revenue', 'orders', 'conversion']
        cumulated_copy = self.cumulated.copy()
        cumulated_copy['date'] = cumulated_copy['date'].dt.date
        revenue_a = cumulated_copy.query('group == "A"')[columns_to_pick]
        revenue_b = cumulated_copy.query('group == "B"')[columns_to_pick]
        merged_revenues = revenue_a.merge(revenue_b,
                                          left_on='date', right_on='date',
                                          how='left', suffixes=['_a', '_b'])
        mean_b_a_revenue_ratio = (((merged_revenues['revenue_b'] /
                                    merged_revenues['orders_b']) /
                                    (merged_revenues['revenue_a'] /
                                    merged_revenues['orders_a']) - 1))
        conversion_b_a_ratio = (merged_revenues['conversion_b'] /
                                merged_revenues['conversion_a'] - 1)

        merged_revenues['mean_revenue_ratio'] = mean_b_a_revenue_ratio
        merged_revenues['conversion_b_a'] = conversion_b_a_ratio

        # кривые удержания платящих пользователей
        ax1 = plt.subplot(2, 3, 1)
        ax1.set_xticks(revenue_a['date'][::7])
        ax1.set_xticklabels(revenue_a['date'][::7])
        plt.plot(revenue_a['date'], revenue_a['revenue'], label='группа A')
        plt.plot(revenue_b['date'], revenue_b['revenue'], label='группа B')
        plt.legend()
        plt.ylabel('Выручка')
        plt.xlabel('Лайфтайм')
        plt.title('Графики кумулятивной выручки по дням и группам')

        ax2 = plt.subplot(2, 3, 2, sharex=ax1)
        plt.plot(revenue_a['date'], revenue_a['revenue'] /
                 revenue_a['orders'], label='группа A')
        plt.plot(revenue_b['date'], revenue_b['revenue'] /
                 revenue_b['orders'], label='группа B')
        plt.legend()
        plt.ylabel('Средняя сумма чека')
        plt.xlabel('Лайфтайм')
        plt.title('Графики среднего чека по группам')

        ax3 = plt.subplot(2, 3, 3, sharex=ax1)
        plt.plot(merged_revenues['date'],
                 merged_revenues['mean_revenue_ratio'])
        plt.axhline(y=0, color='black', linestyle='--')
        plt.ylabel('Отношение средних чеков')
        plt.xlabel('Лайфтайм')
        plt.title('График относительного различия для среднего чека')

        ax4=plt.subplot(2, 3, 4, sharex=ax1)
        plt.plot(revenue_a['date'], revenue_a['conversion'], label='A')
        plt.plot(revenue_b['date'], revenue_b['conversion'], label='B')
        plt.legend()
        plt.ylabel('Конверсия')
        plt.xlabel('Лайфтайм')
        plt.title('График кумулятивной конверсии')

        ax5=plt.subplot(2, 3, 5, sharex=ax1)
        plt.plot(merged_revenues['date'], merged_revenues['conversion_b_a'])
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axhline(y=-0.1, color='grey', linestyle='--')
        plt.title('Относительный прирост конверсии группы '
                  'B относительно группы A')
        plt.tight_layout()
        plt.show()
