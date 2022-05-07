from io import BytesIO

import numpy as np
import pandas as pd
import requests


class OriginDataLoader:
    """Собираем данные из первоисточника, объединяем в одну таблицу."""

    COLUMNS = {'ID': 'id',
               'Name': 'object_name',
               'OperatingCompany': 'oper_name',
               'AdmArea': 'admo',
               'District': 'district',
               'Address': 'address',
               'Longitude_WGS84': 'longitude',
               'Latitude_WGS84': 'latitude'}

    PATH = 'datasets/'

    def __init__(self, data_list):
        self.data_list = data_list
        self.df_merged = None

    def columns_fixer(self, data):
        """Обираем нужные колонки и меняем наименование."""
        data = data[self.COLUMNS.keys()].copy()
        data.rename(columns=self.COLUMNS, inplace=True)
        data = data.astype({'id': 'int64'})
        return data

    def oper_name_fixer(self, data):
        """Приводим к единообразному виду наименование заведения."""
        data['oper_name_fixed'] = np.where(data.oper_name.isnull(),
                                           data.object_name,
                                           data.oper_name)
        return data

    def merger(self):
        """Объединяем таблицы."""
        first = True
        for data_name in self.data_list:
            df = pd.read_excel(self.PATH + data_name + '.xlsx')
            df = self.columns_fixer(df)
            df = self.oper_name_fixer(df)
            if first:
                self.df_merged = df
                first = False
            else:
                self.df_merged = df.merge(self.df_merged, how='outer')

    def save_file(self, name):
        """Сохраняем итоговый файл."""
        self.df_merged.drop('oper_name', axis=1, inplace=True)
        self.df_merged.to_csv(self.PATH + name + '.csv', index=False)


def upload_to_notebook(spreadsheet_id):
    """Получаем файл в тетрадку."""
    spreadsheet_id = spreadsheet_id
    file_name = ('https://docs.google.com/spreadsheets/d/'
                 f'{spreadsheet_id}/export?format=csv')
    request = requests.get(file_name)
    data = pd.read_csv(BytesIO(request.content))
    return data


def simple_grouper(data, grouper, agg_dict={'id': 'count'}):
    """Группируем данные, считаем процентное отношение."""
    data = data.groupby(grouper).agg(agg_dict).reset_index()
    if 'id' in agg_dict.keys():
        data['percent'] = round(data['id'] * 100 / data['id'].sum(), 2)
        data = data.sort_values(by='id', ascending=False)
    return data


def bootstrap(data, n_sample=500, n_trials=10000):

    # зафиксируем случайные числа
    np.random.seed(42)
    result = []
    # определим размер подвыборки
    n_sample = min(len(data), n_sample)
    # инициализируем цикл попыток
    for _ in range(n_trials):
        # делаем подвыборку размера n_sample
        subsample = np.random.choice(data, size=(n_sample,))
        # рассчитываем статистику: среднее
        stat = subsample.mean()
        # добавлям статистику в результат
        result.append(stat)
    return pd.Series(result)
