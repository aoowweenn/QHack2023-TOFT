""" utils.py doc
This module is used to accelerate the matrix operation of Hamiltonian, especially for qml_ham. It also supports tapering qubits off Hamiltonian.
Usage:
1. `from utils import PauSumHam, taper`
2. get PSH_ham from qml_ham: `PSH_ham = PauSumHam.from_qml(qml_ham)`
3. get sparse (csr) matrix: `mat = PSH_ham.to_sparse()`; we can calc the eigvals with scipy
4. tapering qubits: Similar usage as `qml.taper()`, just change `qml.taper` to `taper`, the parameters are all same for both `taper` (our method) and `qml.taper`. Our tapering method will output PSH_ham by default, plz set param `mode='qml'` if you want to output qml_ham.
5. PSH_ham -> qml_ham: `qml_ham = PSH_ham.to_qml()`
"""

import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import eigs, eigsh
# from scipy.linalg import eig, eigh
from functools import reduce

import jax ; jax.config.update('jax_enable_x64', True) #; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp


# from pyscf import gto, dft, scf, cc, df, ao2mo, fci
# from pyscf import fci

import pennylane as qml
import pennylane.numpy as qmlnp
from pennylane import qchem

def pauli(i, P):
    if P == 'X': return qml.PauliX(i)
    if P == 'Y': return qml.PauliY(i)
    if P == 'Z': return qml.PauliZ(i)

def bit_count(a):
    a -= ((a >> 1) & 0x55555555)
    a = (a & 0x33333333) + ((a >> 2) & 0x33333333)
    a = (a + (a >> 4)) & 0x0F0F0F0F
    return (a * 0x01010101) >> 24

@jax.jit
def packed_sum_outer_and(A, B):
    _A = A[:, None, :]
    _B = B[None, :, :]
    C = jnp.vectorize(lambda a,b: bit_count(jnp.uint32(a & b)).astype(jnp.uint8), signature='(i,k,l),(k,j,l)->(i,j,l)')(_A, _B)
    return jnp.sum(C.reshape(-1, C.shape[-1]), axis=1) # C.shape[-1] = 1
    # return C.flatten() # C.shape[-1] = 1

@jax.jit
def packed_outer_xor_2ch(A, B):
    Q = A.shape[2] if A.shape[2] > B.shape[2] else B.shape[2]
    C = (A[:, None, :, :] ^ B[None, :, :, :]).reshape(-1, 2, Q)
    return C

@jax.jit
def packed_sum_add(A, B):
    C = jnp.vectorize(lambda a,b: bit_count(jnp.uint32(a & b)).astype(jnp.uint8), signature='(i,j),(i,j)->(i,j)')(A, B)
    return C.sum(axis=1)

@jax.jit
def add_outer(A, B):
    N, M = A.shape[0], B.shape[0]
    return (A[:, None] + B[None, :]).flatten() #.reshape(N*M, 1)

@jax.jit
def multiply_outer(A, B):
    N, M = A.shape[0], B.shape[0]
    return (A[:, None] * B[None, :]).flatten() #.reshape(N*M, 1)

def impl_PauSumHam_mul(A_coeffs, A, B_coeffs, B):
    count_y_A, count_y_B = impl_count_y(A), impl_count_y(B)
    
    zl_xdot_xr = packed_sum_outer_and(A[:, 1, :], B[:, 0, :])
    C = packed_outer_xor_2ch(A, B)
    xx_dot_zz = packed_sum_add(C[:, 0, :], C[:, 1, :])

    # coeffs = 1j ** ((add_outer(count_y_A, count_y_B) + 2*zl_xdot_xr + 3*xx_dot_zz) % 4) # power of j problem in JAX
    coeffs = 1j ** np.asarray(((add_outer(count_y_A, count_y_B) + 2*zl_xdot_xr + 3*xx_dot_zz) % 4))
    coeffs *= multiply_outer(A_coeffs, B_coeffs)
    return coeffs, C

@jax.jit
def impl_count_y(A):
    _A0 =  A[:, 0, :]
    _A1 =  A[:, 1, :]
    C = jnp.vectorize(lambda a,b: bit_count(jnp.uint32(a & b)).astype(jnp.uint8), signature='(i,j),(i,j)->(i,j)')(_A0, _A1)
    return C.sum(axis=1)

@jax.jit
def impl_get_xstr(A):
    _A0 =  A[:, 0, :]
    _A1 = ~A[:, 1, :]
    C = jnp.vectorize(lambda a,b: bit_count(jnp.uint32(a & b)).astype(jnp.uint8), signature='(i,j),(i,j)->(i,j)')(_A0, _A1)
    return C

# @jax.jit # TracerIntegerConversionError: The __index__() method was called on the JAX Tracer object
def _prepare_sparse_mat_input(coeffs, num_state, y_counts, x_u, x_indices, Zs_ints):
    N = len(coeffs)
    N_u = len(x_u)

    data = np.zeros((num_state*N_u), dtype=np.complex128) # jnp
    indices = np.empty((num_state*N_u), dtype=np.uint32) # jnp
    indptr = np.arange(0, num_state*N_u+1, N_u) # jnp

    for out_idx in range(num_state):
        idx = out_idx * N_u
        for i in range(N):
            u_idx = x_indices[i]
            tot_idx = idx + u_idx
            new_in_idx = out_idx ^ x_u[u_idx]
            indices[tot_idx] = new_in_idx
            # indices = indices.at[tot_idx].set(new_in_idx)
        
            neg_phase_count = bit_count(jnp.uint32(new_in_idx & Zs_ints[i])) #.astype(jnp.uint8)
            phase = 1j ** np.asarray((y_counts[i] + (neg_phase_count << 1)) % 4) # need np.asarray ?
            data[tot_idx] += coeffs[i]*phase
            # data = data.at[tot_idx].add(coeffs[i]*phase)
    
    return data, indices, indptr

# @jax.jit # ConcretizationTypeError (jnp.unique)
def impl_to_sparse(coeffs, num_state, y_counts, Xs_ints, Zs_ints):
    x_u, x_indices, x_counts = jnp.unique(Xs_ints, axis=0, return_inverse=True, return_counts=True)
    data, indices, indptr = _prepare_sparse_mat_input(coeffs, num_state, y_counts, x_u, x_indices, Zs_ints)
    return data, indices, indptr

# @jax.jit # TracerIntegerConversionError: The __index__() method was called on the JAX Tracer object
def reduce_coeffs(A_coeffs, indices, N_simplified):
    N = len(A_coeffs)
    A_coeffs_simplified = np.zeros(N_simplified, dtype=A_coeffs.dtype)
    # for i in range(N):
        # A_coeffs_simplified[indices[i]] += A_coeffs[i]
        ## A_coeffs_simplified = A_coeffs_simplified.at[indices[i]].add(A_coeffs[i])
    
    # ref: https://stackoverflow.com/questions/55735716/how-to-sum-up-for-each-distinct-value-c-in-array-x-all-elements-yi-where-xi
    np.add.at(A_coeffs_simplified, indices, A_coeffs)
    return A_coeffs_simplified

# @jax.jit # ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected
def impl_simplify(A_coeffs, A_pauli, cutoff=1e-15):
    # N, P = A_pauli.shape[0], A_pauli.shape[2]
    A_pauli_simplified, indices = jnp.unique(A_pauli, axis=0, return_inverse=True)
    N_simplified = A_pauli_simplified.shape[0]
    A_coeffs_simplified = reduce_coeffs(A_coeffs, indices, N_simplified)
    ind_remain = jnp.where(jnp.abs(A_coeffs_simplified) >= cutoff) # keep coeffs greater than cutoff
    return A_coeffs_simplified[ind_remain], A_pauli_simplified[ind_remain]

class PauSumHam():
    def __init__(self, coeffs, packed_pauli_words, n_qubits) -> None:
        self.coeffs = coeffs
        self.packed_pauli_words = packed_pauli_words
        self.n_qubits = n_qubits

    # @jax.jit
    @classmethod
    def from_qml(cls, qml_h):
        coeffs = jnp.array(qml_h.coeffs)
        n_qubits = max(qml_h.wires.labels) + 1
        wire_map = {label: label for idx, label in enumerate(qml_h.wires.labels)}

        def pauli_word_to_packbits(pauli_word):
            b_mat = qml.pauli.pauli_to_binary(pauli_word, n_qubits, wire_map)
            return jnp.packbits(b_mat.astype(bool).reshape((2, n_qubits)), axis=-1, bitorder='big')
        
        words = jnp.array([pauli_word_to_packbits(pauli_word) for pauli_word in qml_h.ops])
        return cls(coeffs, words, n_qubits)
    
    def to_qml(self):
        map_dict = {(1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
        # y_counts = np.asarray(self.count_y()) # count numbers of pau_Y
        coeffs = self.coeffs # * (1j ** y_counts) # pau_X @ pau_Z = 1j * pau_Y
        pauli_words = self.unpack()
        N,_,M = pauli_words.shape

        qml_pauli_arr = np.empty(N, dtype=object)
        for i in range(N): # terms
            # identity_tag = True
            pauli_tmp_list = []
            for j in range(M): # qubits
                pau_arr = np.asarray(pauli_words[i, :, j])
                if (pau_arr != 0).any(): # not Identity
                    pauli_tmp_list.append(pauli(j, map_dict[tuple(pau_arr)]))
            if pauli_tmp_list:
                qml_pauli_arr[i] = reduce(lambda X, Y: X@Y, pauli_tmp_list)
            else: # Identity
                qml_pauli_arr[i] = qml.Identity(0)

        return qml.Hamiltonian(coeffs, qml_pauli_arr)
    
    def __matmul__(self, other):
        coeffs, words = impl_PauSumHam_mul(self.coeffs, self.packed_pauli_words, other.coeffs, other.packed_pauli_words)
        return PauSumHam(coeffs, words, max(self.n_qubits, other.n_qubits))
    
    def __imatmul__(self, other):
        self.coeffs, self.words = impl_PauSumHam_mul(self.coeffs, self.packed_pauli_words, other.coeffs, other.packed_pauli_words)
        # self.n_qubits = max(self.n_qubits, other.n_qubits) #?
        return self
    
    # @jax.jit # Err occured
    def unpack(self):
        return jnp.unpackbits(self.packed_pauli_words, axis=-1, count=self.n_qubits, bitorder='big')
    
    def count_y(self):
        return impl_count_y(self.packed_pauli_words)
    
    def get_onlyX(self):
        return impl_get_xstr(self.unpack()) # .packed_pauli_words
    
    # @jax.jit # Err occured
    def get_Xs_Zs_ints(self):
        A = self.packed_pauli_words.view()
        N = A.shape[0]

        Xs_ints = np.empty(N, dtype=np.uint32) # jnp
        Zs_ints = np.empty(N, dtype=np.uint32) # jnp

        offset = (8 - (self.n_qubits % 8)) % 8

        for i in range(N):
            Xs_ints[i], Zs_ints[i] = tuple(np.uint32(int.from_bytes(A[i, j, :].tobytes()[::-1], byteorder='little') >> offset) for j in (0, 1))
            # x_i, z_i = tuple(np.uint32(int.from_bytes(A[i, j, :].tobytes()[::-1], byteorder='little') >> offset) for j in (0, 1))
            # Xs_ints = Xs_ints.at[i].set(x_i)
            # Zs_ints = Zs_ints.at[i].set(z_i)
        
        return Xs_ints, Zs_ints
    
    def simplify(self):
        coeffs_simplified, packed_pauli_words_simplified = impl_simplify(self.coeffs, self.packed_pauli_words)
        return PauSumHam(coeffs_simplified, packed_pauli_words_simplified, self.n_qubits)
    
    def to_sparse(self):
        from scipy.sparse import csr_matrix

        y_counts = self.count_y()
        Xs_ints, Zs_ints = self.get_Xs_Zs_ints()
        num_state = 2**int(self.n_qubits)

        data, indices, indptr = impl_to_sparse(self.coeffs, num_state, y_counts, Xs_ints, Zs_ints)
        return csr_matrix((data, indices, indptr), shape=(num_state,)*2)

def taper(qml_h, generators, paulixops, paulix_sector, mode='psh'):
    from pennylane.qchem.tapering import clifford
    u = clifford(generators, paulixops)
    u_pauham = PauSumHam.from_qml(u)
    h_pauham = PauSumHam.from_qml(qml_h)

    h_pauham = ((u_pauham @ h_pauham).simplify() @ u_pauham).simplify()
    # return h_pauham # UHU test

    N = len(h_pauham.coeffs)

    wireset = list(range(h_pauham.n_qubits)) # u.wires + h.wires
    paulix_wires = [x.wires[0] for x in paulixops] # qubit_num of paulixop

    s_x = h_pauham.get_onlyX()  # unpack()

    val = np.ones(N, dtype=np.complex128)
    # val = jnp.ones(N, dtype=np.complex128)
    for idx, w in enumerate(paulix_wires): # idx: order of paulix; w: qubit-# of paulix
        for i in range(N):
            if s_x[i, w]: # s[w] == "X"
                val[i] *= paulix_sector[idx]
                # val = val.at[i].multiply(paulix_sector[idx])

    wires_tap = [i for i in wireset if i not in paulix_wires]
    T = len(wires_tap)
    s = h_pauham.unpack()
    new_pauli_words = s[:, :, wires_tap]

    # new_pauli_words = jnp.empty((N, 2, T), dtype=np.uint8) # []
    # for i in range(N):
    #     for jdx, j in enumerate(wires_tap):
    #         new_pauli_words.at[i, :, jdx].set(s[i, :, j])

    c = jnp.multiply(val, h_pauham.coeffs)
    packed_new_pauli_words = jnp.packbits(new_pauli_words, axis=-1, bitorder='big') # .reshape((2, n_qubits))
    if mode in ['psh', 'default']:
        return PauSumHam(c, packed_new_pauli_words, T).simplify()
    else: # qml
        return PauSumHam(c, packed_new_pauli_words, T).simplify().to_qml()

if __name__ == "__main__":
    ## test material: H2
    symbols = ["H", "H"]
    d = 1.5
    coordinates = qmlnp.array([[0., 0., -d/2], [0., 0., d/2]], requires_grad=False)
    n_electrons = 2
    ## get qml ham
    qham, n_qubits = qchem.molecular_hamiltonian(symbols, coordinates, basis='sto-3g', mapping='jordan_wigner')
    ## get generators, etc.
    generators = qml.symmetry_generators(qham)
    paulixops = qml.paulix_ops(generators, n_qubits)
    paulix_sector = qml.qchem.optimal_sector(qham, generators, n_electrons)
    ## taper the hamiltonian
    # qml taper: qml.taper(qham, generators, paulixops, paulix_sector)
    # our method: vvv
    ps_ham_tapered = taper(qham, generators, paulixops, paulix_sector)
    ## get tapered ground-state eigvals
    if ps_ham_tapered.n_qubits > 1: # sparse method
        E_tapered = eigsh(ps_ham_tapered.to_sparse(), k=1, which='SA')[0][0]
    else: # switch to normal method when n_qubits == 1 (sparse method would go wrong here)
        E_tapered = np.sort(jnp.linalg.eigvalsh(ps_ham_tapered.to_sparse().todense()))[0]
    ## print results
    print("Material:", symbols)
    # print("Structure:", coordinates)
    print("Original qubits required:", n_qubits)
    print("Qubits required after tapering:", ps_ham_tapered.n_qubits)
    print("Tapered ground-state energy:", E_tapered)
