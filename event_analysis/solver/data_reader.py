import warnings

import numpy as np
import pandas as pd

# Отключаем предупреждения
warnings.filterwarnings('ignore')


class DataReader:
    """Читаем данные, переименовываем колонки, добавляем столбцы с датой."""

    # Зададим константы для пути, колонок, событий и групп
    PATH = 'datasets/'

    COLUMNS = {'EventName': 'event',
               'DeviceIDHash': 'id',
               'EventTimestamp': 'origin_timestamp',
               'ExpId': 'group'}

    EVENTS = {'MainScreenAppear': 'main screen',
              'PaymentScreenSuccessful': 'payment',
              'CartScreenAppear': 'cart',
              'OffersScreenAppear': 'offer screen',
              'Tutorial': 'tutorial'}

    GROUPS = {246: 'A_one',
              247: 'A_two',
              248: 'B_test'}

    def __init__(self, filename):
        """Инициализируем данные, сохраняем оригинал."""
        self.origin_data = pd.read_csv(self.PATH + filename, sep='\t')
        self.data = self.origin_data.copy()

    def get_info(self, data):
        """Получаем базовую информацию о таблице."""

        length = len(data)
        # Получаем типы данных и считаем пропуски
        types = pd.DataFrame(data.dtypes)
        nulls = pd.DataFrame(data.count())

        # Объединяем типы и пропуски, переименовываем колонки
        info = pd.concat([types, nulls], axis=1).reset_index()
        info.columns = ['Column', 'Dtype', 'Non-Null Count']

        # Считаем процент пропусков
        info['% of nulls'] = (100 -
                              round(info['Non-Null Count'] / length * 100, 2))

        return info

    def get_describe(self, data, numeric=False):
        """Считаем описательную статистику."""

        # Если нет числовых значений
        if not numeric:
            describes = data.describe(exclude=[np.number]).fillna('---').T
            top_percent = round((describes['freq'] * 100 /
                                 describes['count']).astype(float), 2)
            describes.insert(4, '% of tops freq', top_percent)
            return describes.sort_values(by='freq')
        return describes

    def basic_info_printer(self, data, numeric=False):
        """Выводим первые и пооследние пять строк, 
        базовую информацию, статистику."""

        # Получаем базовую информацию
        info = self.get_info(data)

        # Получаем базовую статистику
        describes = self.get_describe(data, numeric)

        display('Пять первых и последних строк', data,
                'Общая информация о датасете', info,
                'Описательная статистика', describes)

    def columns_fixer(self):
        """Переименовываем колонки."""
        self.data.rename(columns=self.COLUMNS, inplace=True)

    def group_and_event_values_fixer(self):
        """Переименовываем группы пользователей и события 
        для читабельности."""

        # Задаем пары колонка-словарь для замены
        columns = {'group': self.GROUPS,
                   'event': self.EVENTS}

        # Переименовываем события и группы
        for column, pairs in columns.items():
            self.data[column].replace(pairs, inplace=True)

    def dates_parser(self):
        """Пересчитываем дату."""
        column = self.data['origin_timestamp']
        self.data['timestamp'] = pd.to_datetime(column, unit='s')
        self.data['date'] = self.data['timestamp'].dt.date

    def data_cleaner(self):
        """Базовый пайплайн."""
        self.columns_fixer()
        self.dates_parser()
        self.group_and_event_values_fixer()

        return self.data


def simple_grouper(data, grouper, agg_dict={'id': 'count'}):
    """Группируем данные, считаем процентное отношение."""
    data = data.groupby(grouper).agg(agg_dict).reset_index()
    if 'id' in agg_dict.keys():
        data['percent'] = round(data['id'] * 100 / data['id'].sum(), 2)
        data = data.sort_values(by='id', ascending=False)
    return data


def get_profiles(data):
    """Cоздаем пользовательские профили."""

    # Находим параметры первых посещений
    profiles = (data
                .sort_values(by=['id', 'date'])
                .groupby('id').agg({'timestamp': 'first',
                                    'date': 'first',
                                    'group': 'first'})
                .rename(columns={'date': 'first_ts'})
                .reset_index())

    # Для когортного анализа определяем дату первого посещения
    # и первый день месяца, в который это посещение произошло
    profiles['dt'] = profiles['timestamp'].dt.date
    profiles['month'] = profiles['timestamp'].astype('datetime64[M]')

    # Добавим в профиль сведения о последнем событии для каждого пользователя
    last_event = (data.groupby(['id', 'event', 'date'])
                  .last()
                  .reset_index()
                  .drop_duplicates(subset=['id'], keep='last')
                  [['id', 'event']])
    last_event.columns = ['id', 'last_event']
    profiles = profiles.merge(last_event)

    # Отбираем заплативших пользователей
    orders = data.query('event == "payment"')

    # Добавляем признак платящих пользователей
    profiles['payer'] = profiles['id'].isin(orders['id'].unique())

    return profiles
