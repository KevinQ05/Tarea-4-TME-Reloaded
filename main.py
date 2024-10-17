import numpy as np
from simplex_tme import metodo_simplex_talegon, pretty_print


def main():
    c = np.array([50, 60, 75])
    A_eq = np.array([[1, 1, 1]])
    b_eq = np.array([475])

    bounds = [(0, 160), (0, 300), (50, 150)]

    res = (metodo_simplex_talegon(c=c, A_eq=A_eq, b_eq=b_eq,
                                  bounds=bounds, verbose=True))


if __name__ == '__main__':
    main()
