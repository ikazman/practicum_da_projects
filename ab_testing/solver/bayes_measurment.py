from bayesian_testing.experiments import BinaryDataTest, DeltaLognormalDataTest
from stat_tests import MannWhitneyU


class BayesianTesting(MannWhitneyU):

    def bayesian_tests(self):

        result = []
