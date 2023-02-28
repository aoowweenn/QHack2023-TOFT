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

### get ao/mo rep ########
def find_atom_orb_rep(atompos, orbdirect, rotations, round=8):
    rep_list = [[atompos, orbdirect]]
    # rep_list = [[np.round(atompos, round), orbdirect]]
    for i in range(len(rotations)):
        new_atom_pos = atompos @ rotations[i].T
        # new_atom_pos = np.round(new_atom_pos, round)
        new_orb_direction = orbdirect @ rotations[i].T
        identical_flag = False
        for rep in rep_list:
            if np.allclose(new_atom_pos, rep[0]) and np.allclose(new_orb_direction, rep[1]):
                identical_flag = True
                break
        if not identical_flag:
            rep_list.append([new_atom_pos, new_orb_direction])
    return rep_list

def _get_rep_dict(tags, new_pos, pos, rtol=1e-3, atol=1e-6, mode='atom'):
    rep_dict = {}
    for pair in tags:
        if 'orb' in mode:
            orb_pair = pair
            pair = range(len(pair))
        for j in pair:
            for i in pair:
                if np.allclose(new_pos[j], pos[i], rtol=rtol, atol=atol):
                    rep_dict[(i, j)] = 1
                    break
                elif (mode != 'atom') and np.allclose(new_pos[j], -pos[i], rtol=rtol, atol=atol):
                    rep_dict[(i, j)] = -1
                    break
                if i == pair[-1]: # no matches found
                    return None
    if 'orb' in mode:
        tmp_dict = {}
        for orb_pair in tags:
            for (i, j) in rep_dict:
                tmp_dict[(orb_pair[i], orb_pair[j])] = rep_dict[(i, j)]
        rep_dict = tmp_dict
    return rep_dict

def get_ao_rep(n_ao, rotations, atom_tags, ao_dict, orbdir_tags, atompos, orbdirect=np.eye(3), round=8):
    rep_list = find_atom_orb_rep(atompos, orbdirect, rotations, round=round)
    # print(rep_list)
    ao_rep_mat_list = []
    for new_atom_pos, new_orb_direction in rep_list:
        rep_mat = np.eye(n_ao)
        ## from orbdirect construct orb_rep
        orb_rep_dict = _get_rep_dict(orbdir_tags, new_orb_direction, orbdirect, mode='orb')
        if orb_rep_dict: # construct rep_mat from orb_rep_dict
            for (i, j) in orb_rep_dict: # i -> j
                rep_mat[i][i] = 0
                rep_mat[i][j] = orb_rep_dict[(i, j)]
        else:
            orbdir_mat = np.round(new_orb_direction @ inv(orbdirect), round)
            for i in range(orbdir_mat.shape[0]):
                for j in range(orbdir_mat.shape[1]):
                    for orb_pair in orbdir_tags:
                        rep_mat[orb_pair[i]][orb_pair[j]] = orbdir_mat[i][j]
        ## from atompos construct atom_rep
        # print("rep_mat after orb", rep_mat)
        atom_rep_dict = _get_rep_dict(atom_tags, new_atom_pos, atompos)
        # print(atom_rep_dict)
        if not atom_rep_dict: continue
        ao_rep_list = []
        for (i_label, j_label) in atom_rep_dict:
            for i_ao, j_ao in zip(ao_dict[i_label], ao_dict[j_label]):
                ao_rep_list.append((i_ao, j_ao))
        if len(ao_rep_list) != n_ao: print("Error! Length of ao_rep_list != n_ao!")
        ## swap rep_amt from ao_rep_list
        tmp_mat = np.eye(n_ao)
        for (i, j) in ao_rep_list:
            tmp_mat[:, j] = rep_mat[:, i]
        rep_mat = tmp_mat
        # print("new_atom_pos", new_atom_pos)
        # print("rep_mat after atom", rep_mat)
        ao_rep_mat_list.append(rep_mat)
    ## find indep rep
    tmp_list = []
    for rep_mat in ao_rep_mat_list:
        if np.allclose(rep_mat, np.eye(n_ao)): continue
        if len(tmp_list) == 0: tmp_list.append(rep_mat)
        identical_tag = False
        for tmp_mat in tmp_list:
            if np.allclose(rep_mat, tmp_mat): # find rep_mat in tmp_list, no need to append it and break the loop
                identical_tag = True
                break
        if not identical_tag: tmp_list.append(rep_mat) # append new rep_mat if not in tmp_list
    ao_rep_mat_list = tmp_list
    return ao_rep_mat_list

def get_mo_sym(ao_rep_list, mo_coeff):
    N = len(ao_rep_list)
    M = mo_coeff.shape[1]

    sym_list = [] # np.empty((N, M), dtype=np.int32)
    for i in range(N):
        sym_array = np.empty(M, dtype=np.int32)
        sym_tag = True
        for j in range(M):
            org_vec = mo_coeff[:, j]
            rep_vec = ao_rep_list[i] @ org_vec
            # print(org_vec, rep_vec)
            # if   np.allclose(rep_vec,  org_vec, rtol=1e-3, atol=1e-6): sym_array[j] =  1
            # elif np.allclose(rep_vec, -org_vec, rtol=1e-3, atol=1e-6): sym_array[j] = -1
            if   (np.abs(rep_vec - org_vec) <= 1e-6).all(): sym_array[j] =  1
            elif (np.abs(rep_vec + org_vec) <= 1e-6).all(): sym_array[j] = -1
            else:
                # print(sym_array)
                sym_tag = False
                break
        if sym_tag:
            # print(sym_array)
            sym_list.append(sym_array)
    
    return sym_list

### get generators #######
def get_generators(kernel):
    generators = []
    for i in range(kernel.shape[0]):
        pauli_tmp_list = []
        for j in range(kernel.shape[1]):
            if kernel[i, j] == 1:
                pauli_tmp_list.append(pauli(j, 'Z'))
        sym_pauli = reduce(lambda X, Y: X@Y, pauli_tmp_list)
        generators.append(qml.Hamiltonian([1.0], [sym_pauli]))
    return generators
