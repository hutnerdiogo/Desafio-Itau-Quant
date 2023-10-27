from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from scipy.optimize import minimize
from calculos import Operacoes


@dataclass
class Ativo:
    Papel: str
    SerieTemporalBruta: pd.Series
    SerieRetornos: pd.Series
    Expoente_Hurst_LWE: float = 0
    Expoente_Hurst_GPH: float = 0
    Dimensao_fractal_genton: float = 0
    Dimensao_fractal_Hall_wood: float = 0
    Aproximacao_de_Entropia: float = 0
    Indice_de_Eficiencia: float = 0

    def __post_init__(self):
        self.Indice_de_Eficiencia = (
            (self.Expoente_Hurst_LWE - 0.5) ** 2 +
            (self.Expoente_Hurst_GPH - 0.5) ** 2 +
            # (self.Dimensao_fractal_genton - 1.5) ** 2 +
            # (self.Dimensao_fractal_Hall_wood - 1.5) ** 2 +
            (self.Aproximacao_de_Entropia/2) ** 2
        ) ** 1/2









