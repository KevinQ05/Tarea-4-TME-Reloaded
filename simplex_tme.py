from pydantic import BaseModel, ConfigDict
from typing import TypeAlias
import numpy as np

Array: TypeAlias = np.ndarray


class SimplexBasis(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    c: Array
    A: Array
    b: Array
    variable_loc: list[int]

    def swap_variables(self, i1: int, i2: int):
        self.c[[i1, i2]] = self.c[[i2, i1]]
        self.A[:, [i1, i2]] = self.A[:, [i2, i1]]
        self.variable_loc[i1], self.variable_loc[i2] = self.variable_loc[i2], self.variable_loc[i1]

    def delete_variable(self, col: int):
        self.c = np.delete(self.c, col, axis=0)
        self.A = np.delete(self.A, col, axis=1)
        self.variable_loc.pop(col)

    def copy(self):
        return SimplexBasis(
            c=self.c.copy(),
            A=self.A.copy(),
            b=self.b.copy(),
            variable_loc=self.variable_loc.copy()
        )


class IterationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    z: float
    x: Array
    c_red: Array
    cocientes_bY: Array


def calculate_iteration(basis: SimplexBasis) -> IterationResult:
    c = basis.c
    A = basis.A
    b = basis.b

    m, n = A.shape

    # Separar matriz A en variables básicas y no básicas
    B = A.copy()[:, 0:m]
    N = A.copy()[:, m:n]

    # Costos c básicos y no básicos
    c_B = c.copy()[0:m, 0:]
    c_N = c.copy()[m:n, 0:]

    B_inv = np.linalg.inv(B)

    # solución inicial z0 = z(x0)
    x0 = B_inv@b
    z0 = (c_B.T)@x0

    # Variables duales
    pi = c_B.T@B_inv
    costos_reducidos = np.subtract(pi@N, c_N.T)

    # Necesario a posterior para decidir cuales variables intercambiar
    b_barra = B_inv@b
    Y = B_inv@N
    cocientes_bY = np.divide(b_barra[:, None], Y)
    cocientes_bY = np.where(cocientes_bY > 0, cocientes_bY, np.inf)

    # x0 incluyendo las variables no básicas, que son cero
    result_x = np.zeros((n, 1))
    result_x[0:m, :] = x0[:, None]

    res = IterationResult(
        z=z0, x=result_x, c_red=costos_reducidos, cocientes_bY=cocientes_bY)
    return res


def metodo_simplex_revisado(basis: SimplexBasis, verbose=False) -> dict:
    # Calcular primera solucion basica factible
    iteracion = calculate_iteration(basis)
    if verbose:
        pretty_print(iteracion, basis)
    # si todos los costos reducidos son negativos o cero, es una solución óptima
    if all(c <= 0 for c in iteracion.c_red[0]):
        return {"iter": iteracion, "basis": basis}

    m, _ = basis.A.shape

    # Entra a la base la variable no basica con el costo reducido mayor, y sale la que tenga el coeficiente bY menor positivo
    idx_entrante = iteracion.c_red.argmax()
    idx_saliente = iteracion.cocientes_bY[:, idx_entrante].argmin()

    basis.swap_variables(idx_entrante + m, idx_saliente)

    return metodo_simplex_revisado(basis)


def build_auxiliary_basis(c: Array, A_eq: Array, b_eq: Array, bounds: list[tuple], A_ub: Array | None = None, b_ub: Array | None = None) -> dict:
    A = A_eq
    b = b_eq

    # Apilar condiciones
    if b_ub is not None:
        b = np.append(b, b_ub, axis=0)
    if A_ub is not None:
        A = np.append(A, A_ub, axis=0)

    for idx, (_, upp) in enumerate(bounds):
        if upp > 0:
            new_row = np.zeros((1, A.shape[1]))
            new_row[0, idx] = 1
            A = np.append(A, new_row, axis=0)
            b = np.append(b, np.array([upp]), axis=0)

    for idx, (low, _) in enumerate(bounds):
        if low > 0:
            new_row = np.zeros((1, A.shape[1]))
            new_row[0, idx] = 1
            A = np.append(A, new_row, axis=0)
            b = np.append(b, np.array([low]), axis=0)

    # Con limites inferiores se clasifican distinto a las de slack
    surplus_qty = 0
    for (low, _) in bounds:
        if low > 0:
            new_col = np.zeros((A.shape[0], 1))
            new_col[-1, 0] = -1
            A = np.append(new_col, A, axis=1)
            surplus_qty += 1

    # Se añade una matriz identidad para representar todas las variables de slack y artificiales necesarias para una solucion basica factible
    A = np.append(np.eye(A.shape[0]), A, axis=1)

    # Keeping track de cuantas variables hay por tipo
    real_qty = c.shape[0]
    eq_qty = A_eq.shape[0]
    artificial_qty = eq_qty + surplus_qty
    slack_qty = A.shape[0] - artificial_qty

    total_vars = real_qty + artificial_qty + slack_qty

    # Se busca minimizar, en primera instancia, la suma de las variables artificiales
    c_aux = np.zeros((A.shape[1], 1))
    c_aux[0:eq_qty] = np.ones((eq_qty, 1))
    c_aux[-(surplus_qty+real_qty+1):-(real_qty+1)] = np.ones((surplus_qty, 1))

    normal_vars = range(0, real_qty)
    virtual_vars = range(real_qty, total_vars + 1)
    variable_loc = [x for x in virtual_vars] + [x for x in normal_vars]

    # Despues se van a eliminar, necesito saber cuales son
    def find_artificial_vars(c_aux, variable_loc):
        one_indices = np.where(c_aux == 1)[0]

        result = [variable_loc[i] for i in one_indices]

        return result

    return {
        "basis": SimplexBasis(c=c_aux, A=A, b=b, variable_loc=variable_loc),
        "artificial_vars": find_artificial_vars(c_aux=c_aux, variable_loc=variable_loc)
    }


def metodo_simplex_talegon(c: Array, A_eq: Array, b_eq: Array, bounds: list[tuple], A_ub: Array | None = None, b_ub: Array | None = None, verbose=False):
    # Fase 1, encontrar una solucion basica factible inicial
    aux_basis_and_vars = build_auxiliary_basis(c=c, A_eq=A_eq, b_eq=b_eq,
                                               bounds=bounds, A_ub=A_ub, b_ub=b_ub)
    aux_basis = aux_basis_and_vars["basis"]
    artificial_vars = aux_basis_and_vars["artificial_vars"]

    aux_solution = metodo_simplex_revisado(aux_basis.copy(), verbose=verbose)

    new_basis = aux_solution["basis"]
    iter_result = aux_solution["iter"].x

    variables_a_dar_en_la_nuca = []
    for id, xi in zip(new_basis.variable_loc, iter_result):
        if xi[0] == 0:
            variables_a_dar_en_la_nuca.append(id)

    for var in variables_a_dar_en_la_nuca:
        if var in artificial_vars:
            new_basis.delete_variable(new_basis.variable_loc.index(var))

    # FASE 2 EL MERO METODO SIMPLEX REVISADO CON LA SOLUCION BASICA FACTIBLE ENCONTRADA
    phase_2_basis = new_basis.copy()

    phase_2_basis.c = np.zeros((len(phase_2_basis.variable_loc), 1))
    for var in phase_2_basis.variable_loc:
        if var in range(c.shape[0]):
            phase_2_basis.c[phase_2_basis.variable_loc.index(var)] = c[var]

    return metodo_simplex_revisado(phase_2_basis)


def pretty_print(iter: IterationResult, basis: SimplexBasis):
    print(f"Funcion objetivo z = {iter.z}")
    print("Variables")
    var_val_pairs = sorted(
        zip(basis.variable_loc, iter.x), key=lambda pair: pair[0])

    for var, val in var_val_pairs:
        print(f"x{var}: {val}")
