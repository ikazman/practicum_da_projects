from bayesian_testing.experiments import BinaryDataTest, DeltaLognormalDataTest
from stat_tests import MannWhitneyU


class BayesianTesting(MannWhitneyU):

    def bayesian_tests(self):

        result = []

        # соберем данные для тестирования
        prepared_data = self.prepare_data_for_stat()

        # проведем тест конверсии на сырых данных (Бета-распределение)
        raw_conversion_test = BinaryDataTest()
        raw_conversion_test.add_variant_data('группа A',
                                             prepared_data['sample_raw_a'])
        raw_conversion_test.add_variant_data('группа B',
                                             prepared_data['sample_raw_b'])
        raw_conv_result = raw_conversion_test.probabs_of_being_best()
        result.append(('Конверсия по сырым',
                       raw_conv_result))

        # проведем тест конверсии по «очищенным» данным
        raw_conversion_test = BinaryDataTest()
        raw_conversion_test.add_variant_data('группа A',
                                             prepared_data['sample_raw_a'])
        raw_conversion_test.add_variant_data('группа B',
                                             prepared_data['sample_raw_b'])
        raw_conv_result = raw_conversion_test.probabs_of_being_best()
        result.append(('Конверсия по сырым',
                       raw_conv_result))
        
