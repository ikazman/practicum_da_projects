import pandas as pd
from bayesian_testing.experiments import BinaryDataTest, DeltaLognormalDataTest

from .stat_tests import MannWhitneyU


class BayesianTesting(MannWhitneyU):

    def bayesian_tests(self, data, data_keys, sample_tile):
        test = DeltaLognormalDataTest()
        test.add_variant_data('группа A',
                              data[data_keys[0]])
        test.add_variant_data('группа B',
                              data[data_keys[1]])
        test_result = test.probabs_of_being_best()
        return sample_tile, test_result['группа A'], test_result['группа B']

    def result(self):

        result = []

        # соберем данные для тестирования
        prepared_data = self.prepare_data_for_stat()

        # проведем тест конверсии на сырых данных



        raw_conversion_test = DeltaLognormalDataTest()
        raw_conversion_test.add_variant_data('группа A',
                                             prepared_data['sample_raw_a'])
        raw_conversion_test.add_variant_data('группа B',
                                             prepared_data['sample_raw_b'])
        raw_conv_result = raw_conversion_test.probabs_of_being_best()
        result.append(('Конверсия по сырым',
                       raw_conv_result['группа A'],
                       raw_conv_result['группа B']))

        # проведем тест конверсии по «очищенным» данным
        clean_conversion_test = DeltaLognormalDataTest()
        clean_conversion_test.add_variant_data('группа A',
                                               prepared_data['sample_clean_a'])
        clean_conversion_test.add_variant_data('группа B',
                                               prepared_data['sample_clean_b'])
        clean_conv_result = clean_conversion_test.probabs_of_being_best()
        result.append(('Конверсия по очищенным',
                       clean_conv_result['группа A'],
                       clean_conv_result['группа B']))

        # проведем тест выручки на сырых данных
        raw_revenue_test = DeltaLognormalDataTest()
        raw_revenue_test.add_variant_data('группа A',
                                          prepared_data['revenue_a'])
        raw_revenue_test.add_variant_data('группа B',
                                          prepared_data['revenue_b'])
        raw_revenue_result = raw_revenue_test.probabs_of_being_best()
        result.append(('Средний чек по сырым',
                       raw_revenue_result['группа A'],
                       raw_revenue_result['группа B']))

        # проведем тест выручки по «очищенным» данным
        clean_revenue_test = DeltaLognormalDataTest()
        clean_revenue_test.add_variant_data('группа A',
                                            prepared_data['clean_rev_a'])
        clean_revenue_test.add_variant_data('группа B',
                                            prepared_data['clean_rev_b'])
        clean_revenue_test = clean_revenue_test.probabs_of_being_best()
        result.append(('Средний чек по очищенным',
                       clean_revenue_test['группа A'],
                       clean_revenue_test['группа B']))

        result = pd.DataFrame(result)
        result.columns = ['Выборка', 'группа А (%)',
                          'группа В (%)']
        result['группа А'] = round(result['группа А'] * 100, 2)
        result['группа В'] = round(result['группа В'] * 100, 2)

        return result
