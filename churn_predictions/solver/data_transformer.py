import pandas as pd


def sorted_corr(data, attr):
    correlated = pd.DataFrame(data.corr()[attr].sort_values(ascending=False))
    return correlated


def simple_grouper(data, grouper, agg_dict={'id': 'count'}):
    """Группируем данные, считаем процентное отношение."""
    data = data.groupby(grouper).agg(agg_dict).reset_index()
    if 'id' in agg_dict.keys():
        data['percent'] = round(data['id'] * 100 / data['id'].sum(), 2)
        data = data.sort_values(by='id', ascending=False)
    return data
