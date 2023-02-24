import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig, eigh
from functools import reduce

from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops import InteractionOperator, FermionOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator, eigenspectrum

from pyscf import gto, dft, scf, cc, df, ao2mo, fci
# from pyscf import fci

import pennylane as qml
import pennylane.numpy as qmlnp
from pennylane import qchem

from utils import pauli

def get_kmf(cell, kpt_mesh, with_gamma_point=True): # might be a decorator (as practice)
    kpts = cell.make_kpts(kpt_mesh, with_gamma_point=with_gamma_point)
    kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
    return kmf

# def Pauli_mul(I, J): return I@J
def get_qml_ham(ham, tol=1e-8):
    jw_ham = jordan_wigner(ham)

    coeffs = []
    obs = []
    E_const = 0.
    for op in list(jw_ham.get_operators()):
        op_dict = op.terms
        for P_tuple in op_dict:
            if P_tuple == ():
                const_val = op_dict[P_tuple]
                E_const += const_val
                continue
            P_list = list(P_tuple)
            obs_list = [pauli(*I) for I in P_list]
            obs_tmp = reduce(lambda X, Y: X@Y, obs_list)
            obs.append(obs_tmp)
            coeffs.append(op_dict[P_tuple])
    obs.append(qml.Identity(0))
    coeffs.append(E_const)
    qml_ham = qml.Hamiltonian(coeffs, obs)
    return qml_ham
