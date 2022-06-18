import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (classification_report, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')
np.random.seed(42)


class DataExplorer:

    def __init__(self, data):
        self.final_report = None
        self.best_estimator = []
        self.predictions = []
        self.X = data.drop('Churn', axis=1)
        self.y = data['Churn']

    def data_spliter(self, upsample_data=False):
        """Делим выборку на обучающую и тестовую, стандартизируем данные."""
        X, y = self.X.copy(), self.y.copy()
        x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            random_state=42)
        if upsample_data:
            x_train, y_train = self.upsample(x_train, y_train, 3)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test, y_train, y_test

    def upsample(self, features, target, repeat):
        features_zeros = features[target == 0]
        features_ones = features[target == 1]
        target_zeros = target[target == 0]
        target_ones = target[target == 1]

        features_upsampled = pd.concat(
            [features_zeros] + [features_ones] * repeat)
        target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

        features_upsampled, target_upsampled = shuffle(
            features_upsampled, target_upsampled, random_state=42)

        return features_upsampled, target_upsampled

    def metrics_plot(self, model, model_title, features_valid, target_valid):
        """Выводит на экран PR-кривую и ROC-кривую."""

        probabilities_valid = model.predict_proba(features_valid)
        precision, recall, thresholds = precision_recall_curve(
            target_valid, probabilities_valid[:, 1])
        fpr, tpr, thresholds = roc_curve(
            target_valid, probabilities_valid[:, 1])

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        sns.lineplot(recall, precision, drawstyle='steps-post', ax=ax[0])
        ax[0].set_xlabel('Recall')
        ax[0].set_ylabel('Precision')
        ax[0].set_ylim([0.0, 1.05])
        ax[0].set_xlim([0.0, 1.0])
        ax[0].set_title('Кривая Precision-Recall ' + model_title)

        sns.lineplot(fpr, tpr, ax=ax[1])
        ax[1].plot([0, 1], [0, 1], linestyle='--')
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel('False Positive Rate')
        ax[1].set_ylabel('True Positive Rate')
        ax[1].set_title('ROC-кривая ' + model_title)

    def auc_roc(self, model, features_valid, target_valid):
        """Посчитывает значение ROC-AUC."""

        probabilities_valid = model.predict_proba(features_valid)
        probabilities_one_valid = probabilities_valid[:, 1]
        auc_roc = roc_auc_score(target_valid, probabilities_one_valid)

        return auc_roc

    def grid_search(self, model, param_grid, cv, scoring, x, y):
        """Поиск по сетке с заданными параметрами."""

        grid_model = GridSearchCV(model, param_grid=param_grid,
                                  cv=cv, scoring=scoring,
                                  verbose=1, n_jobs=-1)
        grid_model.fit(x, y)
        best_estimator = grid_model.best_estimator_

        return best_estimator

    def reporter(self, models, score, scoring, upsample_data=False):
        """Обучаем модели, собираем метрики."""
        started = time.time()
        report = []
        estimators = []
        predictions = []
        score_name = str(score).split(' ')[1]

        x_train, x_test, y_train, y_test = self.data_spliter(upsample_data)

        for model in models:
            print('\n', model[0], '\n')
            grid_search = self.grid_search(model[1], model[2], 5,
                                           scoring, x_train, y_train)
            print(grid_search)

            predicted = np.ravel(grid_search.predict(x_test))
            score = f1_score(y_test, predicted)
            roc_auc = self.auc_roc(grid_search, x_test, y_test)

            report.append((model[0], score, roc_auc))
            estimators.append((model[0], grid_search))
            predictions.append((model[0], predicted))
            self.metrics_plot(grid_search, model[0], x_test, y_test)
            print('\n', 'Classification report for ' +
                  model[0], '\n\n', classification_report(y_test, predicted))

        self.final_report = pd.DataFrame(
            report, columns=['model', score_name, 'ROC-AUC'])
        self.best_estimator = pd.DataFrame(
            estimators, columns=['model', 'grid_params'])
        self.predictions = pd.DataFrame(
            predictions, columns=['model', 'predictions'])
        ended = time.time()
        print('Обучение с кросс-валидацей и поиском параметров '
              f'выполнено за {(ended-started) // 60} минуты.')
