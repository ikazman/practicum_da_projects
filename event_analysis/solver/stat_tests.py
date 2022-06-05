import numpy as np
import pandas as pd
from scipy.stats import shapiro
from statsmodels.stats.proportion import proportions_ztest


class StatTest:
    """Класс для проведения статистических тестов и сборов результатов."""

    def __init__(self, data, total_ratio=True):
        self.data = data.copy()
        self.events = self.data.query('event != "tutorial"')['event'].unique()
        self.shapiro_flag = False
        self.result = self.df_constructor()
        self.total_ratio = total_ratio
        self.groups_total = (self.data.groupby('group')
                                      .agg({'id': 'nunique'})
                                      .reset_index())
        self.agg_groups = self.group_constructor()
        self.united_group = (self.agg_groups.query('group in ["A_one", "A_two"]')
                                            .groupby('event')
                                            .sum().reset_index())

    def df_constructor(self):
        df = pd.DataFrame(columns=['Выборка',
                                   'Нулевая гипотеза (Н0)',
                                   'Альтернативная гипотеза (Н1)',
                                   'alpha', 'p-value',
                                   'p-value < alpha',
                                   'Н0/Н1'])
        return df

    def group_constructor(self):
        """Собираем аггрегированные данные по группам."""

        # Зафикисируем порядок и исключим обучение
        order = self.data.drop_duplicates('event')[['event']]
        order = order.query('event != "tutorial"').reset_index()

        # Сгруппируем логи по событиям и группам,
        # посчитаем уникальных пользователей
        groups = self.data.groupby(['event', 'group']).agg(
            {'id': 'nunique'}).reset_index()

        # Объединим две таблицы, оставим тольк нужные стобцы
        groups = (groups.merge(order)
                        .sort_values(by='index')[['event', 'group', 'id']]
                        .reset_index(drop=True))

        for group_marker in groups.group.unique():
            row = groups.group == f'{group_marker}'

            # Считаем число пользователей на предыдущем шаге
            groups.loc[row, 'prev'] = (
                groups.query(f'group == "{group_marker}"').id +
                abs(groups.query(f'group == "{group_marker}"').id.diff())
            )

        for i in groups.group.unique():
            sum_of = int(self.groups_total.query(f'group == "{i}"')['id'])
            groups.loc[groups.group == f'{i}', 'total'] = sum_of

        # Исправляем пропук на первом шаге
        groups['prev'].fillna(groups['total'], inplace=True)

        return groups

    def groups_picker(self, groups, event, united):
        """Выбираем группы из данных."""

        if united:
            group_a = self.united_group.query(f'event == "{event}"')
        else:
            group_a = self.agg_groups.query(f'group in {groups[0]} and '
                                            f'event == "{event}"')
        group_b = self.agg_groups.query(f'group in {groups[1]} and '
                                        f'event == "{event}"')

        return (group_a, group_b)

    def shapiro_test(self):
        """Проверка данных на нормальность. Критерий Шапиро-Уилка."""

        # Сгруппируем данные по дате, посчитам число посещений в день
        # оставим для теста только хэши пользователя
        sample = (self.data.groupby('date')
                  .agg({'id': 'count'})
                  .reset_index()['id'])

        p_value = shapiro(sample)
        return p_value[1]

    def z_test(self, samples):
        """Проводим двухвыборочный Z-критерий."""

        group_a = samples[0]
        group_b = samples[1]

        # Фиксируем число попыток
        count = np.array([group_a['id'], group_b['id']])

        # Пропорции либо от общего, либо от предыдущего шага
        if self.total_ratio:
            nobs = np.array([group_a['total'], group_b['total']])
        else:
            nobs = np.array([group_a['prev'], group_b['prev']])

        # Считаем статистику
        _, p_value = proportions_ztest(count, nobs)
        return p_value[0]

    def test_designer(self, params):
        """Проводим статистические тесты, собираем результаты."""

        # Получаем данные для теста
        sample_name = params['sample_name']
        alpha = params['alpha']
        zero_hypothesis = params['zero_hypothesis']
        alt_hypothesis = params['alt_hypothesis']

        # Проевряем какой тест проводим
        if params['test'] == 'shapiro':
            # Флаг, чтобы исключить из фиальной таблицы
            self.shapiro_flag = True
            p_value = self.shapiro_test()
        else:
            if self.shapiro_flag:  # Если флаг установлен
                self.result = self.df_constructor()  # Очищаем таблицу
                self.shapiro_flag = False
            p_value = self.z_test(params['samples'])

        pvalue_alpha_comparsion = p_value < alpha
        hypo_check = 'Н1' if pvalue_alpha_comparsion else 'Н0'
        row = pd.Series([sample_name, zero_hypothesis,
                        alt_hypothesis, alpha, p_value,
                        pvalue_alpha_comparsion, hypo_check],
                        index=self.result.columns)

        self.result = self.result.append(row, ignore_index=True)

    def test_routine(self, params_of_test, groups_labels,
                     sample_name, united=False):
        """Проводим множественные тесты."""
        for event in list(self.agg_groups.event.unique()):
            # Делаем выборки
            groups_to_test = self.groups_picker(
                groups_labels, event, united=united)
            params_of_test['samples'] = [*groups_to_test]
            # Передаем имя
            params_of_test['sample_name'] = f'{sample_name}: {event}'
            self.test_designer(params_of_test)


def report_styler(report):
    """Выводим таблицу с заданным форматом."""
    display(report.style
                  .set_properties(**{'text-align': 'left'})
                  .set_table_styles([{'selector': 'th',
                                      'props': [('text-align', 'left')]}])
                  .format({'alpha': "{:.2f}", 'p-value < alpha': bool}))


def get_intersections(data, groups, test_group=None):
    """Проверяем, что пользователи из групп изолированы."""

    # Будем собирать данные о пересечениях
    result = {}

    # Отбираем группы из данных
    group_a = data.query(f'group == "{groups[0]}"')['id']
    group_b = data.query(f'group == "{groups[1]}"')['id']

    # Получаем пользователей, попавших в обе группы
    group_intersections = list(np.intersect1d(group_a,
                                              group_b))

    # Если группы не изолированы - печатаем сообщение
    if group_intersections:
        print('Группы не изолированы.')
        result['Персечение групп'] = group_intersections

    # Если передан маркер для тестовой группы и нет пересечений
    if test_group:
        test_group = data.query(f'group == "{test_group[0]}"')['id']
        control_group = pd.concat([group_a, group_b])
        control_groups_intersections = list(np.intersect1d(control_group,
                                                           test_group))

        # Если тестовая группа не изолирована от контрольных - сообщение
        if control_groups_intersections:
            print('Тестовая группа не изолирована от контрольных.')
            result['Персечение с тестовой'] = control_groups_intersections
            return result

    # Если группы изолированы - печатаем сообщение
    if not result:
        print('Пересечений нет, группы изолированы.')

    return result
