import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Picasso:
    """Класс для рисования графиков."""

    def plotter(data, column, column_name, y_label='', x_label=''):
        """Функция для гистограммы, диаграммы рассеивания и размаха."""

        sns.set_style('whitegrid')
        plt.figure(figsize=(18, 8))

        # добавим сеть для прихотливого размещения графиков
        grid = plt.GridSpec(2, 2)

        # Строим гистограмму
        ax1 = plt.subplot(grid[0, 0])
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
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend()
        plt.title(f'{column_name}: распределение ')

        # спрячем сетку
        plt.grid(False)

        # Строим диаграмму рассеивания
        ax2 = plt.subplot(grid[0:, 1])
        x_values = pd.Series(range(0, len(data)))
        sns.scatterplot(ax=ax2, x=x_values, y=data[column],
                        hue=data[column], size=data[column],
                        sizes=(1, 200), linewidth=0,
                        data=data, palette='viridis')
        ax2.axhline(data[column].mean(),
                    linestyle='--',
                    color='orange',
                    label='mean')
        ax2.axhline(data[column].quantile(0.99),
                    linestyle='--',
                    color='red',
                    label='99%')
        plt.legend()
        plt.ylabel(column_name)
        plt.xlabel('Наблюдения в таблице')
        plt.title(f'{column_name}: диаграмма рассеивания')

        # спрячем сетку
        plt.grid(False)

        # Строим диаграмму размаха
        ax3 = plt.subplot(grid[1, 0], sharex=ax1)
        sns.boxplot(x=data[column])
        plt.xlabel(f'{column_name}')
        plt.title(f'{column_name}: диаграмма размаха')

        # спрячем сетку
        plt.grid(False)

        plt.tight_layout()
        plt.show()

    def pie_chart_categories(df, column, desc=''):
        """Круговая диаграмма."""

        # Подготовим данные для визуализации,
        # посчитаем проценты
        data = df.copy()
        data = pd.DataFrame(data[column].value_counts())
        data['percent'] = (data[column] /
                           sum(data[column])) * 100

        # Отбросим значния менее 1%
        data = data[data['percent'] > 1]

        sns.set_style('whitegrid')
        plt.figure(figsize=(7, 7))

        # Нарисуем основную диаграмму
        plt.pie(data['percent'].values, labels=data.index,
                autopct='%0.2f%%', pctdistance=0.7)

        # Добавим круг по центру
        centre_circle = plt.Circle((0, 0), 0.4, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        # Выбрем тип заведения с самым большой долей
        max_percent = data['percent'].max()
        type_with_max_percent = data.loc[data['percent']
                                         == data['percent'].max()].index[0]

        # Добавим сообщение в круг
        message_in_circle = f'{type_with_max_percent} \n  {max_percent:.2f} %'
        plt.text(0, 0, message_in_circle, fontsize=32,
                 ha='center', va='center')

        # Добавим заголовок, источник данных
        plt.title(desc, loc='left', fontsize=13, fontweight='bold')
        source = "Источник: портал открытых данных правительства Москвы"
        plt.text(-1, -1.2, source, color='#a2a2a2', fontsize=10)

        plt.tight_layout()
        plt.axis('square')

    def percent_bar(data, column, labels=None):
        """Столбчатый график для визуализации процентного соотношения."""

        # если не переданы подписи
        if not labels:
            labels = {'bar_label': '',
                      'xlabel': '',
                      'ylabel': '',
                      'title': '',
                      'source': ''}

        sns.set_style('whitegrid')
        plt.figure(figsize=(10, 7))

        # зададим ширину столбца
        bar_width = 0.85

        # рисуем столбчатый график
        plt.bar(data[column],
                data['percent'],
                color='#076fa2',
                edgecolor='white',
                width=bar_width,
                label=labels['bar_label'] if labels else '')

        # подпишем проценты над столбцами
        for object_type, point in zip(data[column],
                                      data['percent'].values):
            plt.text(object_type, point + 5.5,
                     f'{point}%', ha='center', va='top')

        # второй столбчатый график белого цвета
        plt.bar(data[column],
                100 - data['percent'],  # высота столбца
                bottom=data['percent'],  # нулевая точка
                color='white',
                edgecolor='white',
                alpha=0,
                width=bar_width)

        # повернем засечки, подпишем оси
        plt.xticks(rotation=45, ha="right")

        # добавим легкие горизонтальные линии
        plt.grid(axis='y', alpha=0.15)
        plt.grid(axis='x', alpha=0)

        # спрячем лишние границы
        sns.despine(left=True, bottom=True)

        # подпишем график, добавим источник, выведем легенду,
        plt.xlabel(labels['xlabel'], labelpad=30)
        plt.ylabel(labels['ylabel'])
        plt.title(labels['title'], loc='left', pad=30,
                  fontsize=12, fontweight='bold')
        source = 'Источник: ' + labels['source']
        plt.text(-1, -35, source, color='#a2a2a2', fontsize=10)
        plt.legend(loc=0)

    def overlaid_histogram(data1, data2, labels=None, n_bins=0):
        """Строим соотносимые гистограммы для двух выборок"""

        # если не переданы подписи
        if not labels:
            labels = {'data1_name': '',
                      'data2_name': '',
                      'xlabel': '',
                      'ylabel': '',
                      'title': ''}

        # Устанавливаем границы для корзин так чтобы оба распределения
        # на графике были соотносимы
        max_nbins = 10
        data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
        binwidth = (data_range[1] - data_range[0]) / max_nbins

        if n_bins == 0:
            bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)
        else:
            bins = n_bins

        # рисуем графики
        sns.set_style('whitegrid')

        plt.figure(figsize=(10, 8))
        plt.hist(data1, bins=bins, color='red',
                 alpha=0.65, label=labels['data1_name'])
        plt.hist(data2, bins=bins, color='royalblue',
                 alpha=0.65, label=labels['data2_name'])

        # добавляем средние
        plt.axvline(data1.mean(), linestyle='--',
                    color='lime', label=f'Среднее {labels["data1_name"]}')

        plt.axvline(data2.mean(), linestyle='--',
                    color='coral', label=f'Среднее {labels["data2_name"]}')

        # спрячем сетку
        plt.grid(False)

        # подписываем график и оси
        plt.xlabel(labels['xlabel'], labelpad=30)
        plt.ylabel(labels['ylabel'])
        plt.title(labels['title'], loc='left', pad=30,
                  fontsize=12, fontweight='bold')
        plt.legend(loc=0)
        plt.tight_layout()

    def density_plotter(data, columns, labels=None):
        """Строим диаграмму рассевиания с линейной регрессией 
        и диаграмму плотности."""

        plt.figure(figsize=(25, 7))

        # если не переданы подписи
        if not labels:
            labels = {'source': '',
                      'xlabel': '',
                      'ylabel': '',
                      'title': ''}

        # Диаграмма рассевиания с линейной регрессией

        sns.set_style('whitegrid')

        ax1 = plt.subplot(1, 3, 1)

        # Распакуем столбцы
        x_col, y_col = columns

        sns.regplot(x=x_col,
                    y=y_col,
                    data=data,
                    line_kws={'color': 'r', 'alpha': 0.7, 'lw': 2})

        # прячем сетку
        plt.grid(False)

        # Зададим шаг засечек по абсциссе
        plt.xticks(np.arange(min(data[x_col]),
                             max(data[x_col]), step=15))
        plt.yticks(np.arange(min(data[y_col]),
                             max(data[y_col]), step=10))

        # Подпишем графики (один заголовок на два),
        # укажем источник
        plt.title(labels['title'], loc='left', pad=30,
                  fontsize=12, fontweight='bold')
        plt.xlabel(labels['xlabel'], labelpad=15)
        plt.ylabel(labels['ylabel'])
        source = 'Источник: ' + labels['source']
        plt.text(-1, -35, source, color='#a2a2a2', fontsize=10)

        # Диаграмма плотности
        ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        plt.grid(False)
        plt.hist2d(x=x_col,
                   y=y_col,
                   data=data,
                   bins=(20, 20),
                   cmap='BuGn')

        # Подпишем ось абсцисс
        plt.xlabel(labels['xlabel'], labelpad=15)

        plt.tight_layout()
        plt.show()

    def sns_catplot(data, columns, labels=None):
        """Гистограмма."""

        # если не переданы подписи
        if not labels:
            labels = {'bar_label': '',
                      'xlabel': '',
                      'ylabel': '',
                      'title': '',
                      'source': ''}

        x_col, y_col = columns

        sns.set_style('whitegrid')

        plt.figure(figsize=(10, 10))

        sns.catplot(x=x_col, y=y_col,
                    kind='bar', color='#076fa2',
                    data=data,
                    height=7, aspect=.85)

        # добавим легкие горизонтальные линии
        plt.grid(axis='y', alpha=0.15)
        plt.grid(axis='x', alpha=0)

        # подпишем график, добавим источник
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(labels['xlabel'], labelpad=15)
        plt.ylabel(labels['ylabel'])
        plt.title(labels['title'], loc='left', pad=30,
                  fontsize=12, fontweight='bold')
        source = 'Источник: ' + labels['source']
        plt.text(-1, -45, source, color='#a2a2a2', fontsize=10)
        plt.tight_layout()
        plt.show()

    def horizontal_bar(data, names, count, labels):
        """Горизонтальный столбчатый график."""

        # если не переданы подписи
        if not labels:
            labels = {'source': '',
                      'xlabel': '',
                      'title': ''}

        fig, ax = plt.subplots(figsize=(12, 7))

        # задаем ось y
        y_position = range(len(data[names]))

        # строим базовый горизонтальный столбчатый график
        ax.barh(y_position,
                data[count],
                height=0.55,
                align='edge',
                color='#076fa2')

        # оси уберем за график
        ax.set_axisbelow(True)

        # засечки на оси y не требуется
        ax.yaxis.set_visible(False)

        # добавим легкие вертикальные линии
        ax.grid(axis='x', color='#A8BAC4', lw=1.2, alpha=0.25)

        # уберем лишние границы
        sns.despine()

        # подпишем столбцы
        padding = 0.3
        for name, y_pos in zip(data[names], y_position):
            pos = 0
            ax.text(pos + padding, y_pos + 0.5 / 2, name,
                    color='white', fontsize=13, va='center')

        # подпишем график
        plt.title(labels['title'], loc='left', pad=30,
                  fontsize=12, fontweight='bold')
        # подпишем ось х
        plt.xlabel(labels['xlabel'],  labelpad=20)
        # укажем источник
        source = 'Источник: ' + labels['source']
        plt.text(-1, -3, source, color='#a2a2a2', fontsize=10)

    def boxplotter(data, columns, labels=None):
        """Диаграмма размаха."""

        # если не переданы подписи
        if not labels:
            labels = {'xlabel': '',
                      'ylabel': '',
                      'title': '',
                      'source': ''}

        x_col, y_col = columns

        sns.set_style('whitegrid')
        plt.figure(figsize=(15, 10))

        sns.boxplot(x=x_col, y=y_col, data=data)

        # подпишем график, добавим источник
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(labels['xlabel'], labelpad=15)
        plt.ylabel(labels['ylabel'])
        plt.title(labels['title'], loc='left', pad=15,
                  fontsize=12, fontweight='bold')
        source = 'Источник: ' + labels['source']
        plt.text(-1, -125, source, color='#a2a2a2', fontsize=10)
        plt.tight_layout()
        plt.show()
