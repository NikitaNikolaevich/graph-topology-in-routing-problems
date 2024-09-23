import numpy as np

def optimal_a_star(N):
    """
        Calculates the optimal value of a* for a given value of N.

        Parameters
        ----------
        N : int
            Count of nodes

        Returns
        -------
        float
            Optimal value of a*
    """
    return 8.09 * N ** -0.48 * (1 - 19.4 / (4.8 * np.log(N) + 8.8))

def myround(x, base=10):
    """
        Rounds a number to the nearest multiple of a given base.

        Parameters
        ----------
        x : float
            Number to be rounded
        base : int, optional
            Base to round to (default is 10)

        Returns
        -------
        int
            Rounded number
    """
    return base * round(x / base)

# Общая функция для расчета теоретического ускорения
def theoretical_acceleration(x, N):
    """
        Calculates the theoretical acceleration for a given x and N.

        Parameters
        ----------
        x : float
            The alpha value
        N : int
            Count of nodes

        Returns
        -------
        float
            Theoretical acceleration
    """
    k = x * (1 + np.log(x) / np.log(N)) + 46 / (N ** 0.72 * x ** 0.5) * (0.28 - np.log(x) / np.log(N) / 2)
    return 1 / k

# Функция для оптимизации ускорения
def optimized_acceleration(x, a, N):
    """
        Calculates the optimized acceleration for a given x, a, and N.

        Parameters
        ----------
        x : float
            Input value
        a : float
            Input value
        N : int
            Input value

        Returns
        -------
        float
            Optimized acceleration
    """

    k = x * (1 + np.log(x) / np.log(N)) + a / x ** 0.5
    return 1 / k