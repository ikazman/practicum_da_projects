import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class ABReporter:
    """Создаем профили пользователя, расчитываем и визуализируем метрики."""

    def __init__(self, visitors, orders):
        self.visitors = pd.read_csv(visitors, parse_dates=['date'])
        self.orders = pd.read_csv(orders, parse_dates=['date'])
        self.cumulated = None

    def cumulate_column(self, df, column):
        """Cуммируем элементы колонки с накоплением."""
        grouped_by_a = df[df['group'] == 'A'][column].cumsum()
        grouped_by_b = df[df['group'] == 'B'][column].cumsum()
        cumulated = pd.concat([grouped_by_a, grouped_by_b]).reset_index()
        cumulated.sort_values(by='index', inplace=True)
        return cumulated.set_index('index')

    def grouped_summary(self):
        """Получаем сводную таблицу посетителей и заказов."""
        columns = {'transactionId': 'orders',
                   'visitorId': 'buyers'}
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

        visitors['visitors_cm'] = self.cumulate_column(visitors,
                                                       'visitors')
        orders['revenue_cm'] = self.cumulate_column(orders, 'revenue')
        orders['orders_cm'] = self.cumulate_column(orders,
                                                   'transactionId')
        orders['buyers_cm'] = self.cumulate_column(orders, 'visitorId')

        result = orders.merge(visitors)
        result.rename(columns=columns, inplace=True)

        result['conversion_cm'] = result['orders_cm'] / result['visitors_cm']

        self.cumulated = result

        return result

    def prepare_data_for_cm_plot(self):
        """Готовим данные для визуализации кумулятивных метрик."""
        columns_to_pick = ['date', 'revenue_cm', 'orders_cm', 'conversion_cm']
        cumulated_copy = self.cumulated.copy()
        cumulated_copy['date'] = cumulated_copy['date'].dt.date
        revenue_a = cumulated_copy.query('group == "A"')[columns_to_pick]
        revenue_b = cumulated_copy.query('group == "B"')[columns_to_pick]
        merged_revenues = revenue_a.merge(revenue_b,
                                          left_on='date', right_on='date',
                                          how='left', suffixes=['_a', '_b'])
        mean_b_a_revenue_ratio = (((merged_revenues['revenue_cm_b'] /
                                    merged_revenues['orders_cm_b']) /
                                   (merged_revenues['revenue_cm_a'] /
                                    merged_revenues['orders_cm_a']) - 1))
        conversion_b_a_ratio = (merged_revenues['conversion_cm_b'] /
                                merged_revenues['conversion_cm_a'] - 1)

        merged_revenues['mean_revenue_ratio'] = mean_b_a_revenue_ratio
        merged_revenues['conversion_b_a'] = conversion_b_a_ratio

        return revenue_a, revenue_b, merged_revenues

    def plot_cumulative_metrics(self):
        """Функция для визуализации кумулятивных метрик."""
        plt.figure(figsize=(25, 10))
        plt.style.use('seaborn-darkgrid')

        revenue_a, revenue_b, merged_revenues = self.prepare_data_for_cm_plot()

        ax1 = plt.subplot(2, 3, 1)
        ax1.set_xticks(revenue_a['date'][::7])
        ax1.set_xticklabels(revenue_a['date'][::7])
        plt.plot(revenue_a['date'], revenue_a['revenue_cm'], label='группа A')
        plt.plot(revenue_b['date'], revenue_b['revenue_cm'], label='группа B')
        plt.legend()
        plt.ylabel('Выручка')
        plt.xlabel('Лайфтайм')
        plt.title('Графики кумулятивной выручки по дням и группам')

        ax2 = plt.subplot(2, 3, 2, sharex=ax1)
        plt.plot(revenue_a['date'], revenue_a['revenue_cm'] /
                 revenue_a['orders_cm'], label='группа A')
        plt.plot(revenue_b['date'], revenue_b['revenue_cm'] /
                 revenue_b['orders_cm'], label='группа B')
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

        ax4 = plt.subplot(2, 3, 4, sharex=ax1)
        plt.plot(revenue_a['date'],
                 revenue_a['conversion_cm'],
                 label='группа A')
        plt.plot(revenue_b['date'],
                 revenue_b['conversion_cm'],
                 label='группа B')
        ax4.set_ylim(0, revenue_a['conversion_cm'].max() + 0.01)
        plt.legend()
        plt.ylabel('Конверсия')
        plt.xlabel('Лайфтайм')
        plt.title('График кумулятивной конверсии')

        ax5 = plt.subplot(2, 3, 5, sharex=ax1)
        plt.plot(merged_revenues['date'], merged_revenues['conversion_b_a'])
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axhline(y=merged_revenues['conversion_b_a'].mean(),
                    color='grey', linestyle='--',
                    label='Среднее отношение конверсии')
        plt.title('Относительный прирост конверсии группы '
                  'B относительно группы A')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plotter(self, data, column, column_name):
        """Функция для гистограммы, диаграммы рассеивания и размаха."""
        plt.figure(figsize=(25, 5))

        x_values = pd.Series(range(0, len(self.orders['revenue'])))

        ax1 = plt.subplot(1, 3, 1)
        sns.histplot(data=data, x=column)
        ax1.axvline(data[column].median(),
                    linestyle='--',
                    color='#FF1493',
                    label='median')
        ax1.axvline(data[column].mean(),
                    linestyle='--',
                    color='orange',
                    label='mean')
        ax1.axvline(data[column].quantile(0.1),
                    linestyle='--',
                    color='yellow',
                    label='1%')
        ax1.axvline(data[column].quantile(0.99),
                    linestyle='--',
                    color='yellow',
                    label='99%')
        plt.ylabel('Число пользователей')
        plt.xlabel('Сумма')
        plt.legend()
        plt.title(f'{column_name}: распределение ')

        ax2 = plt.subplot(1, 3, 2)
        sns.scatterplot(ax=ax2, x=x_values, y=data[column],
                        hue=data[column], size=data[column],
                        sizes=(1, 200), linewidth=0, data=data)
        plt.legend()
        plt.ylabel('Сумма')
        plt.xlabel('Число заказов')
        plt.title(f'{column_name}: диаграмма рассеивания')

        ax3 = plt.subplot(1, 3, 3)
        sns.boxplot(x=data[column])
        plt.xlabel(f'{column_name}')
        plt.title(f'{column_name}: диаграмма размаха')
        plt.tight_layout()
        plt.show()
