import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import seaborn as sns

class MetricCalculator:
    """Создаем профили пользователя, расчитываем и визуализируем метрики."""

    def __init__(self, visits, orders, costs):
        self.visits = pd.read_csv(visits, parse_dates=['Session Start',
                                                       'Session End'])
        self.orders = pd.read_csv(orders, parse_dates=['Event Dt'])
        self.costs = pd.read_csv(costs, parse_dates=['dt'])

    def columns_fixer(self):
        """Приводим колонки к одному регистру, переименовываем по
        необходимости, конвертируем формат."""
        self.visits.columns = self.visits.columns.str.lower()
        self.orders.columns = self.orders.columns.str.lower()
        self.costs.columns = self.costs.columns.str.lower()

        self.costs['dt'] = self.costs['dt'].dt.date

        self.visits.rename(columns={'user id': 'user_id',
                                    'session start': 'session_start',
                                    'session end': 'session_end'},
                           inplace=True)

        self.orders.rename(columns={'user id': 'user_id',
                                    'event dt': 'event_dt'},
                           inplace=True)

    def acquisitions_date(self, profiles, observation,
                          horizon, ignore_horizon):
        """Исключаем пользователей, не «доживших» до горизонта анализа."""
        if ignore_horizon:
            acquisition_date = observation
        acquisition_date = observation - timedelta(days=horizon - 1)
        raw_data = profiles.query('dt <= @acquisition_date')
        return raw_data

    def group_by_dimensions(self, df, dims, horizon,
                            aggfunc='nunique', cumsum=False):
        """Группировка таблицы по желаемым признакам."""
        result = df.pivot_table(index=dims, columns='lifetime',
                                values='user_id', aggfunc=aggfunc)

        if cumsum:
            result = result.fillna(0).cumsum(axis=1)

        cohort_sizes = (df.groupby(dims)
                        .agg({'user_id': 'nunique'})
                        .rename(columns={'user_id': 'cohort_size'}))
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon))]
        result['cohort_size'] = cohort_sizes
        return result

    def cac_roi(self, df, grouped_df, dims, horizon):
        """Считаем CAC и ROI на треуголной таблице."""

        # датафрейм с данными пользователей CAC, добавляем dimensions
        cac = df[['user_id', 'acquisition_cost'] + dims].drop_duplicates()

        # считаем средний CAC по параметрам из dimensions
        cac = (cac.groupby(dims)
               .agg({'acquisition_cost': 'mean'})
               .rename(columns={'acquisition_cost': 'cac'}))

        # считаем ROI: делим LTV на CAC
        roi = grouped_df.div(cac['cac'], axis=0)

        # удаляем строки с бесконечным ROI
        roi = roi[~roi['cohort_size'].isin([np.inf])]
        cohort_sizes = (df.groupby(dims).agg({'user_id': 'nunique'}).rename(
            columns={'user_id': 'cohort_size'}))
        # восстанавливаем размеры когорт в таблице ROI
        roi['cohort_size'] = cohort_sizes

        # добавляем CAC в таблицу ROI
        roi['cac'] = cac['cac']

        # в финальной таблице оставляем размеры когорт, CAC
        # и ROI в лайфтаймы, не превышающие горизонт анализа
        roi = roi[['cohort_size', 'cac'] + list(range(horizon))]

        # возвращаем таблицы LTV и ROI
        return roi

    def lifetime_calculation(self, df, to_merge, columns_to_merge, last_date):
        """Добавляем данные о покупках и рассчитываем лайфтайм пользователя для
        каждой покупки."""
        df = df.merge(to_merge[columns_to_merge], on='user_id', how='left')
        df['lifetime'] = (df[last_date] - df['first_ts']).dt.days
        return df

    def dimensions_check(self, df, dims):
        """Функция для группировки по коготам если в dims пусто."""
        if len(dims) == 0:
            df['cohort'] = 'All users'
            dims = dims + ['cohort']
        return df, dims

    def filter_data(self, df, window):
        """Функция для сглаживания фрейма: применяем скользящее среднее."""
        for column in df.columns.values:
            df[column] = df[column].rolling(window).mean()
        return df

    def get_profiles(self):
        """Cоздаем пользовательские профили."""
        ad_costs = self.costs.copy()

        # находим параметры первых посещений
        profiles = (self.visits
                    .sort_values(by=['user_id', 'session_start'])
                    .groupby('user_id').agg({'session_start': 'first',
                                             'channel': 'first',
                                             'device': 'first',
                                             'region': 'first', })
                    .rename(columns={'session_start': 'first_ts'})
                    .reset_index())

        # для когортного анализа определяем дату первого посещения
        # и первый день месяца, в который это посещение произошло
        profiles['dt'] = profiles['first_ts'].dt.date
        profiles['month'] = profiles['first_ts'].astype('datetime64[M]')

        # добавляем признак платящих пользователей
        profiles['payer'] = profiles['user_id'].isin(
            self.orders['user_id'].unique())

        # считаем количество уникальных пользователей
        # с одинаковыми источником и датой привлечения
        new_users = (profiles.groupby(['dt', 'channel'])
                     .agg({'user_id': 'nunique'})
                     .rename(columns={'user_id': 'unique_users'})
                     .reset_index())

        # объединяем траты на рекламу и число привлечённых пользователей
        ad_costs = ad_costs.merge(new_users, on=['dt', 'channel'], how='left')

        # делим рекламные расходы на число привлечённых пользователей
        ad_costs['acquisition_cost'] = (
            ad_costs['costs'] / ad_costs['unique_users'])

        # добавляем стоимость привлечения в профили
        profiles = profiles.merge(
            ad_costs[['dt', 'channel', 'acquisition_cost']],
            on=['dt', 'channel'],
            how='left')

        # стоимость привлечения органических пользователей равна нулю
        profiles['acquisition_cost'] = profiles['acquisition_cost'].fillna(0)

        return profiles

    def get_retention(self, profiles, observation_date, horizon,
                      dimensions=[], ignore_horizon=False):
        """Функция для расчёта удержания."""

        # добавляем столбец payer в передаваемый dimensions список
        dimensions = ['payer'] + dimensions

        # исключаем пользователей, не «доживших» до горизонта анализа
        retention_raw = self.acquisitions_date(profiles, observation_date,
                                               horizon, ignore_horizon)

        # собираем «сырые» данные для расчёта удержания
        retention_raw = self.lifetime_calculation(retention_raw, self.visits,
                                                  ['user_id', 'session_start'],
                                                  'session_start')

        # получаем таблицу удержания
        retention_grouped = self.group_by_dimensions(retention_raw,
                                                     dimensions,
                                                     horizon)

        # получаем таблицу динамики удержания
        retention_hist = self.group_by_dimensions(retention_raw,
                                                  dimensions + ['dt'],
                                                  horizon)

        # сырые данные, таблица RR, таблица динамики RR
        return retention_raw, retention_grouped, retention_hist

    def get_conversion(self, profiles, observation_date, horizon,
                       dimensions=[], ignore_horizon=False):
        """Функция для расчёта конверсии (CR)."""

        # исключаем пользователей, не «доживших» до горизонта анализа
        cr_raw = self.acquisitions_date(profiles, observation_date,
                                        horizon, ignore_horizon)

        # определяем дату и время первой покупки для каждого пользователя
        first_purchases = (self.orders.sort_values(by=['user_id', 'event_dt'])
                           .groupby('user_id')
                           .agg({'event_dt': 'first'})
                           .reset_index())

        # добавляем данные о покупках в профили
        cr_raw = self.lifetime_calculation(cr_raw, first_purchases,
                                           ['user_id', 'event_dt'],
                                           'event_dt')

        # группируем по cohort, если в dimensions ничего нет
        cr_raw, dimensions = self.dimensions_check(cr_raw, dimensions)

        # получаем таблицу конверсии
        cr_grouped = self.group_by_dimensions(cr_raw,
                                              dimensions,
                                              horizon)

        # для таблицы динамики конверсии убираем 'cohort' из dimensions
        if 'cohort' in dimensions:
            dimensions = []

        # получаем таблицу динамики конверсии
        cr_hist = self.group_by_dimensions(cr_raw,
                                           dimensions + ['dt'],
                                           horizon)

        # сырые данные, таблица CR, таблица динамики CR
        return cr_raw, cr_grouped, cr_hist

    def get_ltv(self, profiles, observation_date, horizon,
                dimensions=[], ignore_horizon=False):
        """Функция для расчёта LTV и ROI."""

        # исключаем пользователей, не «доживших» до горизонта анализа
        ltv_raw = self.acquisitions_date(profiles, observation_date,
                                         horizon, ignore_horizon)

        # добавляем данные о покупках в профили
        to_merge_columns = ['user_id', 'event_dt', 'revenue']
        ltv_raw = self.lifetime_calculation(ltv_raw, self.orders,
                                            to_merge_columns,
                                            'event_dt')

        # группируем по cohort, если в dimensions ничего нет
        ltv_raw, dimensions = self.dimensions_check(ltv_raw, dimensions)

        # получаем таблицы LTV и ROI
        ltv_grouped = self.group_by_dimensions(ltv_raw,
                                               dimensions,
                                               horizon,
                                               aggfunc='sum',
                                               cumsum=True)
        roi = self.cac_roi(ltv_raw,
                           ltv_grouped,
                           dimensions,
                           horizon)

        # для таблиц динамики убираем 'cohort' из dimensions
        if 'cohort' in dimensions:
            dimensions = []

        # получаем таблицы динамики LTV и ROI
        ltv_hist = self.group_by_dimensions(ltv_raw,
                                            dimensions + ['dt'],
                                            horizon,
                                            aggfunc='sum',
                                            cumsum=True)
        roi_hist = self.cac_roi(ltv_raw,
                                ltv_hist, dimensions + ['dt'],
                                horizon)

        # сырые данные, таблица LTV, динамика LTV, ROI, динамика ROI
        return ltv_raw, ltv_grouped, ltv_hist, roi, roi_hist

    def plot_retention(self, retention, retention_hist, horizon, window=7):
        """Функция для визуализации удержания."""
        plt.figure(figsize=(15, 10))

        retention = retention.drop(columns=['cohort_size', 0])
        retention_hist = retention_hist.drop(
            columns=['cohort_size'])[[horizon - 1]]

        if retention.index.nlevels == 1:
            retention['cohort'] = 'All users'
            retention = retention.reset_index().set_index(['cohort', 'payer'])

        # кривые удержания платящих пользователей
        ax1 = plt.subplot(2, 2, 1)
        retention.query('payer == True').droplevel('payer').T.plot(
            grid=True, ax=ax1)
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
        filtered_data.plot(grid=True, ax=ax3)
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

    def plot_conversion(self, conversion, conversion_hist, horizon, window=7):
        """Функция для визуализации конверсии."""

        plt.figure(figsize=(15, 5))

        conversion = conversion.drop(columns=['cohort_size'])
        conversion_hist = conversion_hist.drop(
            columns=['cohort_size'])[[horizon - 1]]

        # кривые конверсии
        ax1 = plt.subplot(1, 2, 1)
        conversion.T.plot(grid=True, ax=ax1)
        plt.legend()
        plt.xlabel('Лайфтайм')
        plt.title('Конверсия пользователей')

        # динамика конверсии
        ax2 = plt.subplot(1, 2, 2, sharey=ax1)
        columns = [name for name in conversion_hist.index.names
                   if name not in ['dt']]
        filtered_data = conversion_hist.pivot_table(
            index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
        filtered_data = self.filter_data(filtered_data, window)
        filtered_data.plot(grid=True, ax=ax2)
        plt.xlabel('Дата привлечения')
        plt.title(f'Динамика конверсии пользователей на {horizon}-й день')

        plt.tight_layout()
        plt.show()

    def plot_ltv_roi(self, ltv, ltv_hist, roi, roi_hist, horizon, window=7):
        """Функция для визуализации LTV и ROI."""

        plt.figure(figsize=(20, 10))

        ltv = ltv.drop(columns=['cohort_size'])
        ltv_hist = ltv_hist.drop(columns=['cohort_size'])[[horizon - 1]]
        cac_hist = roi_hist[['cac']]
        roi = roi.drop(columns=['cohort_size', 'cac'])
        roi_hist = roi_hist.drop(columns=['cohort_size', 'cac'])[[horizon - 1]]

        # кривые ltv
        ax1 = plt.subplot(2, 3, 1)
        ltv.T.plot(grid=True, ax=ax1)
        plt.legend()
        plt.xlabel('Лайфтайм')
        plt.title('LTV')

        # динамика ltv
        ax2 = plt.subplot(2, 3, 2, sharey=ax1)
        columns = [name for name in ltv_hist.index.names if name not in ['dt']]
        filtered_data = ltv_hist.pivot_table(
            index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
        self.filter_data(filtered_data, window).plot(grid=True, ax=ax2)
        plt.xlabel('Дата привлечения')
        plt.title(f'Динамика LTV пользователей на {horizon}-й день')

        # динамика cac
        ax3 = plt.subplot(2, 3, 3, sharey=ax1)
        columns = [name for name in cac_hist.index.names if name not in ['dt']]
        filtered_data = cac_hist.pivot_table(
            index='dt', columns=columns, values='cac', aggfunc='mean')
        self.filter_data(filtered_data, window).plot(grid=True, ax=ax3)
        plt.xlabel('Дата привлечения')
        plt.title('Динамика стоимости привлечения пользователей')

        # кривые roi
        ax4 = plt.subplot(2, 3, 4)
        roi.T.plot(grid=True, ax=ax4)
        plt.axhline(y=1, color='red', linestyle='--',
                    label='Уровень окупаемости')
        plt.legend()
        plt.xlabel('Лайфтайм')
        plt.title('ROI')

        # динамика roi
        ax5 = plt.subplot(2, 3, 5, sharey=ax4)
        columns = [name for name in roi_hist.index.names if name not in ['dt']]
        filtered_data = roi_hist.pivot_table(
            index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
        self.filter_data(filtered_data, window).plot(grid=True, ax=ax5)
        plt.axhline(y=1, color='red', linestyle='--',
                    label='Уровень окупаемости')
        plt.xlabel('Дата привлечения')
        plt.title(f'Динамика ROI пользователей на {horizon}-й день')

        plt.tight_layout()
        plt.show()

    def histogram(self, data, n_bins, range_start, range_end, grid,
                  cumulative=False, x_label='', y_label='', title=''):
        """Простая гистограмма

        Пример:
        histogram(df, 100, 0, 150, True, 'Количество иксов',
                  'Количество игриков', 'Заголовок')

        data - датасет
        n_bins - количество корзин
        range_start - минимальный икс для корзины
        range_end - максимальный икс для корзины
        grid - рисовать сетку или нет (False / True)


        histogram(data, n_bins, range_start, range_end, grid,
                  x_label = "", y_label = "", title = "")
        """

        # Создаем объект - график
        _, ax = plt.subplots()

        # Задаем параметры
        ax.hist(data, bins=n_bins, range=(range_start, range_end),
                cumulative=cumulative, color='#4169E1')

        # Добавляем сетку
        if grid == True:
            ax.grid(color='grey', linestyle='-', linewidth=0.5)
        else:
            pass

        # Добавляем медиану, среднее и квартили
        ax.axvline(data.median(), linestyle='--',
                   color='#FF1493', label='median')
        ax.axvline(data.mean(), linestyle='--', color='orange', label='mean')
        ax.axvline(data.quantile(0.1), linestyle='--',
                   color='yellow', label='1%')
        ax.axvline(data.quantile(0.99), linestyle='--',
                   color='yellow', label='99%')
        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)

    def sns_catplot(self, x, y, data, title='', 
                    xlabel='', ylabel='', *args, **kwargs):
        """Гистограмма."""
        plt.style.use('seaborn-darkgrid')
        sns.catplot(x=x, y=y,
                    kind='bar', color='orange',
                    data=data,
                    height=7, aspect=1.9, saturation=.5)
        _ = plt.title(title, fontsize=18, loc='left')
        _ = plt.xlabel(xlabel, fontsize=18)
        _ = plt.ylabel(ylabel, fontsize=18)
        _ = plt.xticks(rotation=45)
        plt.show()
