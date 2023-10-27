from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from scipy.optimize import minimize
from EntropyHub import ApEn
from scipy import stats, optimize
from whittle import whittle


class Operacoes:
    @staticmethod
    def matrix_multipler(*matrices):
        # Verifica se há pelo menos duas matrizes para multiplicar
        if len(matrices) < 2:
            raise ValueError("Pelo menos duas matrizes são necessárias para multiplicação matricial.")

        # Inicializa o resultado com a primeira matriz
        result = matrices[0]

        # Multiplica as matrizes sequencialmente
        for matrix in matrices[1:]:
            result = np.dot(result, matrix)

        return result

    @staticmethod
    def calculate_acf(data, lag_max):
        acf_values = acf(data, nlags=lag_max)
        return acf_values

    @staticmethod
    def local_whittle_estimator(acf_values):
        return whittle(acf_values)

    # @staticmethod
    # def geweke_porter_hudak_estimator(serie):
    #     def sample_GHE(serie, tau):
    #         return np.abs(serie[tau:] - serie[:-tau])
    #
    #     scaling_range = [2 ** n for n in range(int(np.log2(len(serie))) - 2)]
    #     sample_t0 = sample_GHE(serie, tau=scaling_range[0])
    #     f = lambda h: np.sum(
    #         [stats.ks_2samp(sample_t0, sample_GHE(serie, tau=tau) / (tau ** h)).statistic for tau in scaling_range[1:]])
    #     w = optimize.fmin(f, x0=0.5, disp=False)
    #     return w[0]

    @staticmethod
    def geweke_porter_hudak_estimator(time_series, max_lag=20):
        """Returns the Hurst Exponent of the time series"""

        lags = range(2, max_lag)

        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

        reg = np.polyfit(np.log(lags), np.log(tau), 1)

        return reg[0]

    @staticmethod
    def genton_fractal_dimension(data, k=2):
        n = len(data)
        sum_diff = 0

        for t in range(1, n - k + 1):
            sum_diff += abs(data[t + k - 1] - data[t - 1])

        genton_dimension = 1 + np.log(2 * sum_diff / (n - k)) / np.log(n)

        return genton_dimension

    # @staticmethod
    # def hall_wood_fractal_dimension(data, k=2):
    #     n = len(data)
    #     sum_diff = 0
    #
    #     for t in range(1, n - k + 1):
    #         sum_diff += abs(data[t + k - 1] - data[t - 1])
    #
    #     hall_wood_dimension = np.log(sum_diff) / np.log(n / k)
    #
    #     return hall_wood_dimension

    @staticmethod
    def hall_wood_fractal_dimension(data, L=2):
        n = len(data)
        lags = np.arange(1, n // 2)
        box_counts = []

        for lag in lags:
            num_boxes = n // lag
            boxes = np.array_split(data, num_boxes)
            box_areas = [np.trapz(box) for box in boxes]  # Área de cada caixa.

            box_counts.append(np.sum(box_areas))

        # Ajuste do log-log para calcular a dimensão fractal.
        log_lags = np.log(lags)
        log_counts = np.log(box_counts)
        p = np.polyfit(log_lags, log_counts, 1)
        fractal_dimension = -p[0]

        return fractal_dimension

    @staticmethod
    def aproximacao_entropia(data):
        valor = ApEn(data, r=2 * np.std(data))[0]
        final = np.average(valor[1:3])
        return final

    @staticmethod
    def indice_eficiencia():
        return "Hello"
