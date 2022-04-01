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

    def group_by_dimensions(self, df, dims, horizon_days, aggfunc, cumsum=False):
        """Группировка таблицы по желаемым признакам."""
        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc=aggfunc)
        if cumsum:
            result = result.fillna(0).cumsum(axis=1)
        cohort_sizes = (df.groupby(dims).agg({'user_id': 'nunique'}).rename(
            columns={'user_id': 'cohort_size'}))
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon_days))]
        result['cohort_size'] = cohort_sizes

    def get_profiles(self, ad_costs):
        """Cоздаем пользовательские профили."""

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
        new_users = (
            profiles.groupby(['dt', 'channel'])
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'unique_users'})
            .reset_index()
        )

        # объединяем траты на рекламу и число привлечённых пользователей
        self.costs = self.costs.merge(
            new_users, on=['dt', 'channel'], how='left')

        # делим рекламные расходы на число привлечённых пользователей
        self.costs['acquisition_cost'] = (
            self.costs['costs'] / self.costs['unique_users'])

        # добавляем стоимость привлечения в профили
        profiles = profiles.merge(
            ad_costs[['dt', 'channel', 'acquisition_cost']],
            on=['dt', 'channel'],
            how='left',
        )

        # стоимость привлечения органических пользователей равна нулю
        profiles['acquisition_cost'] = profiles['acquisition_cost'].fillna(0)

        return profiles
