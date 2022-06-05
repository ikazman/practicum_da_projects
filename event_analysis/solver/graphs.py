import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns


# Определяем способ вывода графиков plotly
pio.renderers.default = 'iframe'


def histogram(data, bins='rice', labels=None, stats=False):
    """Строим гистограмму."""

    # если не переданы подписи
    if not labels:
        labels = {'xlabel': '',
                  'title': ''}

    # Задаем стиль
    sns.set_style('whitegrid')

    plt.figure(figsize=(15, 10))

    # Строим график
    plt.hist(data, bins=bins, color='#575fcf')

    # Добавим легкие вертикальные линии
    plt.grid(axis='x', alpha=0.0)
    plt.grid(axis='y', alpha=0.15)

    # Добавим среднее, медиану, 1-й и 99-й перцентили
    if stats:
        plt.axvline(data.median(), color='#ffd32a',
                    label='медиана', linestyle='--')
        plt.axvline(data.mean(), color='#ff3f34',
                    label='среднее', linestyle='--')
        plt.axvline(data.quantile(0.1), color='#1e272e',
                    label='1%', linestyle='--')
        plt.axvline(data.quantile(0.99), color='#1e272e',
                    label='99%', linestyle='--')
        plt.legend()

    # Подпишем оси и график
    plt.xlabel(labels['xlabel'], labelpad=30)
    plt.ylabel('Число наблюдений')
    plt.title(labels['title'], loc='left', pad=30,
              fontsize=12, fontweight='bold')


def funnel_plot(data, groups_labels=None, x='id', y='event',
                title='', *args, **kwargs):
    """Строим вороноку."""

    fig = go.Figure()

    # Если переданы маркеры групп
    if groups_labels:
        for group in groups_labels:
            sample = data.query(f'group == "{group}"')
            fig.add_trace(go.Funnel(
                name=group,
                y=sample[y],
                x=sample[x], *args, **kwargs))

    else:  # Строим по всем данным
        fig.add_trace(go.Funnel(name='Все пользователи',
                                y=data[y],
                                x=data[x], *args, **kwargs))

    # Изменим размер, фон, добавим заголовок
    fig.update_layout(autosize=False,
                      width=700,
                      height=500,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title=title)
    fig.show()
