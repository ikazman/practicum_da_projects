import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


class MetricCalculator:
    """Создаем профили пользователя, расчитываем и визуализируем метрики."""

    def __init__(self, visits, orders, costs):
        self.visits = pd.read_csv(visits, parse_dates=['Session Start'])
        self.orders = pd.read_csv(orders, parse_dates=['Event Dt'])
        self.costs = pd.read_csv(costs, parse_dates=['dt'])
        self.profiles = None

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
        """Исключаем пользователей, не «доживших» до горизонта анализа"""

        if ignore_horizon:
            acquisition_date = observation
        acquisition_date = observation - timedelta(days=horizon - 1)
        result_raw = profiles.query('dt <= @acquisition_date')
        return result_raw

    def group_by_dimensions(self, df, dims, horizon,
                            aggfunc='nunique', cumsum=False):
        """Группировка таблицы по желаемым признакам."""

        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc=aggfunc)
        if cumsum:
            result = result.fillna(0).cumsum(axis=1)
        cohort_sizes = (df.groupby(dims).agg({'user_id': 'nunique'}).rename(
            columns={'user_id': 'cohort_size'}))
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon))]
        result['cohort_size'] = cohort_sizes
        return result

    def cac_roi(self, df, grouped_df, dims, horizon):
        """Считаем CAC и ROI на треуголной таблице"""

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
        result_raw = self.acquisitions_date(profiles, observation_date,
                                            horizon, ignore_horizon)

        # собираем «сырые» данные для расчёта удержания
        result_raw = result_raw.merge(
            self.visits[['user_id', 'session_start']],
            on='user_id', how='left')

        result_raw['lifetime'] = (
            result_raw['session_start'] - result_raw['first_ts']
        ).dt.days

        # получаем таблицу удержания
        result_grouped = self.group_by_dimensions(result_raw,
                                                  dimensions,
                                                  horizon)

        # получаем таблицу динамики удержания
        result_in_time = self.group_by_dimensions(result_raw,
                                                  dimensions + ['dt'],
                                                  horizon)

        # возвращаем обе таблицы и сырые данные
        return result_raw, result_grouped, result_in_time

    def get_conversion(self, profiles, observation_date, horizon,
                       dimensions=[], ignore_horizon=False):
        """Функция для расчёта удержания"""

        # исключаем пользователей, не «доживших» до горизонта анализа
        result_raw = self.acquisitions_date(profiles, observation_date,
                                            horizon, ignore_horizon)

        # определяем дату и время первой покупки для каждого пользователя
        first_purchases = (self.orders.sort_values(by=['user_id', 'event_dt'])
                           .groupby('user_id')
                           .agg({'event_dt': 'first'})
                           .reset_index())

        # добавляем данные о покупках в профили
        result_raw = result_raw.merge(
            first_purchases[['user_id', 'event_dt']], on='user_id', how='left')

        # рассчитываем лайфтайм для каждой покупки
        result_raw['lifetime'] = (
            result_raw['event_dt'] - result_raw['first_ts']
        ).dt.days

        # группируем по cohort, если в dimensions ничего нет
        if len(dimensions) == 0:
            result_raw['cohort'] = 'All users'
            dimensions += ['cohort']

        # получаем таблицу конверсии
        result_grouped = self.group_by_dimensions(result_raw,
                                                  dimensions,
                                                  horizon)

        # для таблицы динамики конверсии убираем 'cohort' из dimensions
        if 'cohort' in dimensions:
            dimensions = []

        # получаем таблицу динамики конверсии
        result_in_time = self.group_by_dimensions(result_raw,
                                                  dimensions + ['dt'],
                                                  horizon)

        # возвращаем обе таблицы и сырые данные
        return result_raw, result_grouped, result_in_time

    def get_ltv(self, profiles, observation_date, horizon, dimensions=[], ignore_horizon=False):
        """Функция для расчёта LTV и ROI"""

        # исключаем пользователей, не «доживших» до горизонта анализа
        result_raw = self.acquisitions_date(profiles, observation_date,
                                            horizon, ignore_horizon)

        # добавляем данные о покупках в профили
        result_raw = result_raw.merge(
            self.orders[['user_id', 'event_dt', 'revenue']], on='user_id', how='left'
        )
        # рассчитываем лайфтайм пользователя для каждой покупки
        result_raw['lifetime'] = (
            result_raw['event_dt'] - result_raw['first_ts']
        ).dt.days

        # группируем по cohort, если в dimensions ничего нет
        if len(dimensions) == 0:
            result_raw['cohort'] = 'All users'
            dimensions = dimensions + ['cohort']

        # получаем таблицы LTV и ROI
        result_grouped = self.group_by_dimensions(result_raw,
                                                  dimensions,
                                                  horizon,
                                                  aggfunc='sum',
                                                  cumsum=True)
        roi_grouped = self.cac_roi(result_raw,
                                   result_grouped,
                                   dimensions,
                                   horizon)

        # для таблиц динамики убираем 'cohort' из dimensions
        if 'cohort' in dimensions:
            dimensions = []

        # получаем таблицы динамики LTV и ROI
        result_in_time = self.group_by_dimensions(result_raw,
                                                  dimensions + ['dt'],
                                                  horizon,
                                                  aggfunc='sum',
                                                  cumsum=True)
        roi_in_time = self.cac_roi(result_raw,
                                   result_in_time, dimensions + ['dt'],
                                   horizon)

        return (
            result_raw,  # сырые данные
            result_grouped,  # таблица LTV
            result_in_time,  # таблица динамики LTV
            roi_grouped,  # таблица ROI
            roi_in_time,  # таблица динамики ROI
        )
