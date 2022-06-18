import numpy as np
import pandas as pd


class BasicInfo:
    """Выводим базовую информацию о датасете."""

    def __init__(self, data):
        self.data = data

    def get_info(self):
        """Получаем базовую информацию о таблице."""

        length = len(self.data)
        # Получаем типы данных и считаем пропуски
        types = pd.DataFrame(self.data.dtypes)
        nulls = pd.DataFrame(self.data.count())

        # Объединяем типы и пропуски, переименовываем колонки
        info = pd.concat([types, nulls], axis=1).reset_index()
        info.columns = ['Column', 'Dtype', 'Non-Null Count']

        # Считаем процент пропусков
        info['% of nulls'] = (100 -
                              round(info['Non-Null Count'] / length * 100, 2))

        return info

    def get_describe(self):
        """Считаем описательную статистику."""
        describes = self.data.describe().fillna('---').T
        return describes

    def basic_info_printer(self):
        """Выводим первые и пооследние пять строк, 
        базовую информацию, статистику."""

        # Получаем базовую информацию
        info = self.get_info()

        # Получаем базовую статистику
        describes = self.get_describe()

        display('Пять первых и последних строк', self.data,
                'Общая информация о датасете', info,
                'Описательная статистика', describes)
