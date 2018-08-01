#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################################################
#                                                                                                              #
# Copyright (C) {2018}  {Jernigan lab, Iowa State University}                                                  #
# Author - Ambuj Kumar,                                                                                        #
# This program is free software: you can redistribute it and/or modify                                         #
# it under the terms of the GNU General Public License as published by                                         #
# the Free Software Foundation, either version 3 of the License, or                                            #
# (at your option) any later version.                                                                          #
#                                                                                                              #
# This program is distributed in the hope that it will be useful,                                              #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                                               #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                                #
# GNU General Public License for more details.                                                                 #
#                                                                                                              #
# This program comes with ABSOLUTELY NO WARRANTY;                                                              #
# This is free software, and you are welcome to redistribute it                                                #
# under certain conditions;                                                                                    #
#                                                                                                              #
################################################################################################################
# 'movie' module used for creating movie animation has been removed from the code.
# It had few error and I am trying to fix it asap.
#
# All the modules associated with analysing ENM data can be added to the class Analyze.
# Drop me an email if you have any question.
################################################################################################################


from __future__ import print_function

name = "pyMAVEN"
__author__ = "Ambuj Kumar <ambuj@iastate.edu>"
__date__ = "29th July 2018"
__version__ = "1.0"

__credits__ = """Prof. Robert Jernigan for being an amazing advisor.
    Prof. Guang Song and Dr. Tu-Liang Lin, the original creators of STeM."""


import sys

try:
    import numpy as np
except ImportError:
    raise MissingDependencyError("Numpy is a hard dependency for pyMAVEN. Install Numpy if you want to use pyMAVEN.")

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy import sparse as scipy_sparse
except ImportError:
    raise MissingDependencyError("Failed to import Scipy, which  is required for sparse matrix calculations. Install Scipy if you want to use pyMAVEN.")


class MissingDependencyError(ImportError):
    """Missing an external python dependency (subclass of ImportError).
        
        Used for missing Python modules (Scipy and Numpy).
        """
    
    pass


class MavenWarning(Warning):
    """Maven warning.
        
        Useful to silence all our warning messages should you wish to:
        
        >>> import warnings
        >>> from pyMAVEN import MavenWarning
        >>> warnings.simplefilter('ignore', MavenWarning)
        
        """
    
    pass



class ANM:
    def __init__(self, gamma=1.0, dr=15.0, pf=None):
        """Class to run Anisotropic Elastic Network model.
            A simple analytical approach (ANM) for estimating the mechanism of collective
            motions in proteins. ANM approach permits us to evaluate directional preferences.
            In principle, the ANM may be viewed as a simplified form of NMA, in which
            inter-residue interactions are as- sumed to be nonspecific.
            
            Operations:-
            >>> from pyMAVEN import ANM, Analyze
            >>> mod = ANM(); an = Analyze()
            >>> coords = an.coord('filename.pdb')
            >>> Hess = mod.Hessian(coords)
            >>> D, V = an.decompose(Hess)
            >>> bfr = an.calc_bfactor(D, V)
            
            User can select specific values of spring constant (gamma) and distance cutoff (dr).
            >>> mod = STEM(gamma = 0.8, dr=13.0) [Default: gamma=1.0, dr=15.0]
            
            """
        
        self.gamma   = gamma
        self.dr      = dr
        self.pf      = pf
        if self.pf != None and self.pf <= 0:
            raise Exception("pf value cannot be zero or negative")
        if self.gamma <= 0:
            raise Exception("gamma value cannot be zero or negative")
        if self.dr <= 0:
            raise Exception("distance cutoff value cannot be zero or negative")

    def Hessian(self, coords):
        n_atoms = len(coords)
        hessian = np.zeros((n_atoms*3, n_atoms*3), float)
        distance_mat = np.ones((n_atoms*3, n_atoms*3), float)
        for i in range(len(coords)):
            diff = coords[i+1:, :] - coords[i]
            squared_diff = diff*diff
            for j, s_ij in enumerate(squared_diff.sum(1)):
                if s_ij <= self.dr**2:
                    diff_coords = diff[j]
                    j = j + i + 1
                    derivative = np.outer(diff_coords, diff_coords)*(float(-self.gamma)/s_ij)
                    hessian[i*3:i*3+3, j*3:j*3+3] = derivative
                    hessian[j*3:j*3+3, i*3:i*3+3] = derivative
                    hessian[i*3:i*3+3, i*3:i*3+3] = hessian[i*3:i*3+3, i*3:i*3+3] - derivative
                    hessian[j*3:j*3+3, j*3:j*3+3] = hessian[j*3:j*3+3, j*3:j*3+3] - derivative
                    
                    d = np.sqrt(s_ij)
                    lobj = [[d,d,d],[d,d,d], [d,d,d]]
                    dmat = np.array(lobj)
                    distance_mat[i*3:i*3+3, j*3:j*3+3] = dmat
    
        if self.pf != None:
            hessian = hessian/distance_mat

        return hessian




class STEM:
    """Class to run Spring Tensor Elastic Network model.
        
        STEM is a Gō-like potential based elastic network model.
        
        Derived from a physically more realistic potential, STeM analysis allows us to achieve benefits
        of GNM and ANM in one single model. It thus lightens the burden to work with two separate models
        and to relate the modes of GNM with those of ANM at times. STEM examinins the contributions
        of different interaction terms in the Gō potential to estimate the fluctuation dynamics
        via three-body and four-body interactions.
        
        It also untilizes the concept of inverse distance square spring constants which performs better
        than the uniform ones,
        

        Reference:
        
        Tu-Liang and Guang Song. Generalized spring tensor models for protein fluctuation dynamics and
        conformation changes. BMC Struct Biol. 2010; 10(Suppl 1): S3.
        
        load STEM module
        >>> from pyMAVEN import STEM, Analyze
        
        initiate model
        >>> mod = STEM(); an = Analyze()
        
        pull PDB coordinate as Nx3 numpy array - float
        >>> coords = an.coord('filename.pdb')
        ...      array([[  5.194,  13.275, -17.249],
        ...       [  5.955,  13.221, -16.006],
        ...       [  7.228,  12.391, -16.165],
        ...       ...,
        ...       [ 26.427,   4.957,  11.402],
        ...       [ 27.332,   5.125,  10.578],
        ...       [ 26.458,   6.373,  13.487]])
        
        create four-body interactions Gō potential Hessian matrix
        >>> Hess = mod.Hessian(coords)
        ...      array([[ 5.38262170e+01, -2.52745103e+01,  4.88821405e+01, ...,
        ...       -9.36139107e-03,  3.03857793e-03, -1.35314012e-02],
        ...       [-2.52745103e+01,  3.68397538e+01, -2.14575153e+01, ...,
        ...       3.03857793e-03, -9.86280327e-04,  4.39210550e-03],
        ...       [ 4.88821405e+01, -2.14575153e+01,  8.91507524e+01, ...,
        ...       -1.35314012e-02,  4.39210550e-03, -1.95589329e-02],
        ...       ...,
        ...       [-9.36139107e-03,  3.03857793e-03, -1.35314012e-02, ...,
        ...       6.88167110e+01,  2.06183226e+01, -5.73051178e+00],
        ...       [ 3.03857793e-03, -9.86280327e-04,  4.39210550e-03, ...,
        ...       2.06183226e+01,  6.62660284e+01,  3.32375669e+01],
        ...       [-1.35314012e-02,  4.39210550e-03, -1.95589329e-02, ...,
        ...       -5.73051178e+00,  3.32375669e+01,  1.09979198e+02]])
        
        perform matrix decomposition to obtain eigenvectors and eigenvalues
        >>> D, V = an.decompose(Hess)
        >>> D
        ...      array([-2.50793777e-11, -2.58569811e-14,  7.95027076e-14, ...,
        ...       2.51574478e+05,  5.13107042e+05,  7.58135356e+05])
        >>> V
        ...      array([[ 6.87316329e-02,  0.00000000e+00,  0.00000000e+00, ...,
        ...       -4.92026845e-14,  1.52300720e-12, -2.74233047e-14],
        ...       [ 2.61795740e-02,  2.01243884e-02, -7.47996115e-03, ...,
        ...       5.73425131e-12,  5.65298895e-12, -3.85938820e-12],
        ...       [-1.10845339e-02, -2.22408127e-02,  1.83768793e-03, ...,
        ...       2.41105244e-11, -6.78597999e-12, -3.63097469e-12],
        ...       ...,
        ...       [ 2.48624519e-02,  3.58897850e-03, -2.50857966e-03, ...,
        ...       -3.03897708e-11,  1.11366868e-11,  1.31603645e-11],
        ...       [-2.04454648e-02,  2.14667084e-02,  1.43486320e-02, ...,
        ...       -1.10138361e-10,  2.69251985e-12,  2.08121265e-11],
        ...       [ 8.79535437e-03, -2.44223375e-02,  8.47496629e-03, ...,
        ...       -1.24399829e-10,  1.52599429e-11,  6.15407468e-11]])
        
        compute bfactors
        >>> bfr = an.calc_bfactor(D, V)
        ...      array([0.81372239, 0.22848427, 0.07876034, ..., 0.01280826, 0.01387003,
        ...       0.01749366])
        
        Together the entire analysis requires 6 easy steps
        >>> from pyMAVEN import STEM, Analyze
        >>> mod = STEM(); an = Analyze()
        >>> coords = an.coord('filename.pdb')
        >>> Hess = mod.Hessian(coords)
        >>> D, V = an.decompose(Hess)
        >>> bfr = an.calc_bfactor(D, V)
        
        STEM module can also be executed on GPU and MPI machines using following options
        set gpu=True to execute STEM on a GPU machine
        >>> mod = STEM(gpu=True)
        set mpi=True to execute STEM on an mpi machine (Not working at the moment)
        >>> mod = STEM(mpi=True)
        
        User can also define STEM model parameter values using following options
        epsilon = 0.4 (or any other value) can be defined while calling STEM class
        >>> mod = STEM(epsilon = 0.4) [Default=0.36]
        
        User can also choose to avoid first few rigid body modes by selecting ig=6 (ignoring first 6 modes)
        Furthermore, user can choose to select a number of non rigid body modes by defining num=50 (or any other value)
        [Default=20, picks mode starting from 7 to 26]
        >>> mod = STEM(epsilon = 0.4, ig=6, num=100)
        
        To get help type help(STEM) in python command line.
        
        Feel free to drop me an email at ambuj@iastate.edu if you have any question, suggestion or an error detail.
        
        """
    
    def __init__(self, epsilon=0.36, ig=6, num=20, gpu=False, mpi=False):
        """Initialize the class."""
        self.epsilon = epsilon
        self.ig      = ig
        self.num     = num
        self.gpu     = gpu
        self.mpi     = mpi
            

    def Hessian(self, coords):
        """Create STeM Hessian matrix
            """
        K_r = 100*self.epsilon
        K_theta = 20*self.epsilon
        K_phi1 = 1*self.epsilon
        K_phi3 = 0.5*self.epsilon
        numOfResidues = len(coords)
        distance_mat = squareform(pdist(coords))
        
        hessian = np.zeros((numOfResidues*3, numOfResidues*3))
        hv1 = _firstTerm(coords, hessian, K_r, distance_mat)
        hessian = np.zeros((numOfResidues*3, numOfResidues*3))
        hv2 = _secondTerm(hessian,coords,distance_mat,numOfResidues,K_theta)
        hessian = np.zeros((numOfResidues*3, numOfResidues*3))
        hv3 = _thirdTerm(hessian,coords,distance_mat,numOfResidues,K_phi1,K_phi3)
        hessian = np.zeros((numOfResidues*3, numOfResidues*3))
        hv4 = _fourthTerm(coords, hessian, self.epsilon, distance_mat)
        
        Hess = hv1+hv2+hv3+hv4
        return Hess


##################################################################################################
#                                       Public Functions                                         #
##################################################################################################


class Analyze:
    """This class contains operations to analyse ENM hessian matrizes and mode data
        """
    def __init__(self):
        """Class for ENM data analysis.
            """
        self.message = "Loading Analyze module"


    def decompose(self, Hess, gpu_flag=False, mpi_flag=False):
        """Perform matrix eigen decomposition
            """
        if Hess.any() == False:
            raise("Hessian matrix not found\n")
        
        if gpu_flag == True:
        
            try:
                import pycuda.gpuarray as gpuarray
                import pycuda.autoinit
                try:
                    from skcuda import linalg
                except ImportError, e:
                    print("skuda not found. Hessian matrix decomposition will be performed on CPU\n")
            except ImportError, e:
                print("pycuda not found. Hessian matrix decomposition will be performed on CPU\n")
        
            linalg.init()
            a_gpu = gpuarray.to_gpu(Hess)
            D, V = linalg.eigh(a_gpu, 'N', 'V', lib='cusolver')
        else:
            if mpi_flag == True:
                raise('MPI process is not supported in pySTEM at the moment. Keep an eye out for the update!')
                try:
                    from mpi4py import MPI
                    comm = MPI.COMM_WORLD
                    V, D = comm.Send(np.linalg.eigh(Hess))
                except ImportError:
                    print("Failed to import mpi4py, which is required for parallel processing\nSkipping parallel prcoessing :(")
            else:
                D, V = np.linalg.eigh(Hess)
        
        E = _sort_E(D)
        e2 = list(); i=list()
        for x, y in E:
            e2.append(y)
            i.append(x)
        e2=np.array(e2)
        v2 = np.zeros((len(V), len(V)))
        a = 0
        
        while a < len(E):
            v2[:,a] = V[:,i[a]]
            a = a + 1
                    
        return e2, v2


    def calc_bfactor(self, eval, evec, num=20, ig=6):
        """Given eigenvectors, V, and eigenvalues, E, calculate a vector of
            isotropic temperature factors from an ANM calculation (3N).
            """
        if eval is None:
            raise Exception("No eigen values given.")
        if evec is None:
            raise Exception("No eigen vectors given.")
        if len(eval) != len(evec):
            raise Exception("Eigen values and eigen vectors are not of same dimension.\nEigen vector - %s\nEigen value - %s" %(eval.shape, evec.shape))

        T  = 300.0          # temperature in Kelvin; 0C = 273K; F=C*(9/5) + 32; 300K = 80F
        Kb = 1.38065e-23  # Boltzmann Constant J/K
        Gamma = 1e-20      # It is the gas constant divided by Avogadro's number
        n=len(eval)
        B = np.zeros(n/3)
        cv1 = range(0, n, 3)
        cv2 = range(1, n, 3)
        cv3 = range(2, n, 3)
        i=0
        while i < num+ig:
            if i < ig:
                i=i+1
                continue
    
            v = np.square(evec[:,i])
            B = np.add(B, (float(1)/(eval[i]))*np.add(v[cv1,], v[cv2,], v[cv3,]))
            i = i + 1

        bfactor = (8.0/3)*(np.pi**2)*Kb*(T/Gamma)*B
            
        return bfactor

    def coord(self, pdbfile):
        """Parse PDB file
            """
        if pdbfile == None:
            raise Exception("No input pdb file given.")
        
        file = open(pdbfile, "r")
        coords = list()
        index_list = list()
        for lines in file:
            lobj = lines.split(" ")
            if len(lines) < 6 or "ANISOU" in lines:
                continue
            if "ATOM" == lobj[0] or "HETATM" == lobj[0]:
                try:
                    coords.append([float(lines[30:38]), float(lines[38:46]), float(lines[46:54])])
                except:
                    raise Exception("Error in parsing PDB coordinates. Please check you input file")
    
        if coords == []:
            raise Exception("No atomic coordinates found in the input file")
        if len(coords) < 10:
            raise Exception("Too few coordinates in the input file")


        coords = np.array(coords)
        return coords


##################################################################################################
#                                      Private Functions                                         #
##################################################################################################


############################### First Element ####################################################

def _derivative_consf1(el1, el2, K_r, distijsqr): return K_r*el1*el2/distijsqr

def _diag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, m, n, forward=True):
    """diagonals of off-diagonal super elements (1st term)"""
    if forward == True:
        hessian[m*3,n*3] = hessian[m*3,n*3]-(2*_derivative_consf1(bx, bx, K_r, distijsqr))
        hessian[m*3+1,n*3+1] = hessian[m*3+1,n*3+1]-(2*_derivative_consf1(by, by, K_r, distijsqr))
        hessian[m*3+2,n*3+2] = hessian[m*3+2,n*3+2]-(2*_derivative_consf1(bz, bz, K_r, distijsqr))
    else:
        hessian[m*3,n*3] = hessian[m*3,n*3]+(2*_derivative_consf1(bx, bx, K_r, distijsqr))
        hessian[m*3+1,n*3+1] = hessian[m*3+1,n*3+1]+(2*_derivative_consf1(by, by, K_r, distijsqr))
        hessian[m*3+2,n*3+2] = hessian[m*3+2,n*3+2]+(2*_derivative_consf1(bz, bz, K_r, distijsqr))
    return hessian

def _offdiag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, m, n, forward=True):
    """off-diagonals of off-diagonal super elements (1st term)"""
    if forward == True:
        hessian[m*3,n*3+1] = hessian[m*3,n*3+1]-(2*_derivative_consf1(bx, by, K_r, distijsqr))
        hessian[m*3,n*3+2] = hessian[m*3,n*3+2]-(2*_derivative_consf1(bx, bz, K_r, distijsqr))
        hessian[m*3+1,n*3] = hessian[m*3+1,n*3]-(2*_derivative_consf1(by, bx, K_r, distijsqr))
        hessian[m*3+1,n*3+2] = hessian[m*3+1,n*3+2]-(2*_derivative_consf1(by, bz, K_r, distijsqr))
        hessian[m*3+2,n*3] = hessian[m*3+2,n*3]-(2*_derivative_consf1(bz, bx, K_r, distijsqr))
        hessian[m*3+2,n*3+1] = hessian[m*3+2,n*3+1]-(2*_derivative_consf1(bz, by, K_r, distijsqr))
    else:
        hessian[m*3,n*3+1] = hessian[m*3,n*3+1]+(2*_derivative_consf1(bx, by, K_r, distijsqr))
        hessian[m*3,n*3+2] = hessian[m*3,n*3+2]+(2*_derivative_consf1(bx, bz, K_r, distijsqr))
        hessian[m*3+1,n*3] = hessian[m*3+1,n*3]+(2*_derivative_consf1(by, bx, K_r, distijsqr))
        hessian[m*3+1,n*3+2] = hessian[m*3+1,n*3+2]+(2*_derivative_consf1(by, bz, K_r, distijsqr))
        hessian[m*3+2,n*3] = hessian[m*3+2,n*3]+(2*_derivative_consf1(bz, bx, K_r, distijsqr))
        hessian[m*3+2,n*3+1] = hessian[m*3+2,n*3+1]+(2*_derivative_consf1(bz, by, K_r, distijsqr))
    return hessian


def _firstTerm(caArray, hessian, K_r, distance_mat):
    """derive the hessian of the first term (off diagonal)
        """
    i=0
    while i < len(caArray):
        j = i + 1
        
        if j == len(caArray):
            break
        
        bx=caArray[i,0] - caArray[j,0]
        by=caArray[i,1] - caArray[j,1]
        bz=caArray[i,2] - caArray[j,2]
        distijsqr = distance_mat[j, i]**2
        hessian = _diag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, i, j, forward=True)
        hessian = _offdiag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, i, j, forward=True)
        hessian = _diag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, j, i, forward=True)
        hessian = _offdiag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, j, i, forward=True)
        hessian = _diag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, i, i, forward=False)
        hessian = _offdiag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, i, i, forward=False)
        hessian = _diag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, j, j, forward=False)
        hessian = _offdiag_offdiagf1(hessian, bx, by, bz, K_r, distijsqr, j, j, forward=False)
        
        i = i + 1
    
    return hessian


############################### Second Element ################################################

def _derivative_consf2(K_theta, el1, el2, G): return float(K_theta)/(1-G**2)*el1*el2

def _diag_offdiagf2(hessian, K_theta, dGdXm, dGdYm, dGdZm, dGdXn, dGdYn, dGdZn, G, m, n):
    """diagonals of off-diagonal super elements (2nd term)
        """
    hessian[m*3,n*3] = hessian[m*3,n*3]+2*_derivative_consf2(K_theta, dGdXm, dGdXn, G)
    hessian[m*3+1,n*3+1] = hessian[m*3+1,n*3+1]+2*_derivative_consf2(K_theta, dGdYm, dGdYn, G)
    hessian[m*3+2,n*3+2] = hessian[m*3+2,n*3+2]+2*_derivative_consf2(K_theta, dGdZm, dGdZn, G)
    return hessian

def _offdiag_offdiagf2(hessian, K_theta, dGdXm, dGdYm, dGdZm, dGdXn, dGdYn, dGdZn, G, m, n):
    """off-diagonals of off-diagonal super elements (second term)
        """
    hessian[m*3,n*3+1] = hessian[m*3,n*3+1]+2*_derivative_consf2(K_theta, dGdXm, dGdYn, G)
    hessian[m*3,n*3+2] = hessian[m*3,n*3+2]+2*_derivative_consf2(K_theta, dGdXm, dGdZn, G)
    hessian[m*3+1,n*3] = hessian[m*3+1,n*3]+2*_derivative_consf2(K_theta, dGdYm, dGdXn, G)
    hessian[m*3+1,n*3+2] = hessian[m*3+1,n*3+2]+2*_derivative_consf2(K_theta, dGdYm, dGdZn, G)
    hessian[m*3+2,n*3] = hessian[m*3+2,n*3]+2*_derivative_consf2(K_theta, dGdZm, dGdXn, G)
    hessian[m*3+2,n*3+1] = hessian[m*3+2,n*3+1]+2*_derivative_consf2(K_theta, dGdZm, dGdYn, G)
    return hessian

def _secondTerm(hessian,caArray,distance_mat,numOfResidues,K_theta):
    """derive the hessian of the second term (off diagonal)
        """
    i=0
    while i < len(caArray):
        
        j = i + 1
        k = i + 2
        if k == len(caArray):
            break
        
        Xi=caArray[i,0]
        Yi=caArray[i,1]
        Zi=caArray[i,2]
        Xj=caArray[j,0]
        Yj=caArray[j,1]
        Zj=caArray[j,2]
        Xk=caArray[k,0]
        Yk=caArray[k,1]
        Zk=caArray[k,2]
        
        p = caArray[i,0:3]-caArray[j,0:3]
        q = caArray[k,0:3]-caArray[j,0:3]
        lpl = distance_mat[i,j]
        lql = distance_mat[k,j]
        G = np.dot(p,q)/(lpl*lql)
        
        
        dGdXi = ((Xk-Xj)*lpl*lql-np.dot(p,q)*(lql/lpl)*(Xi-Xj))/(lpl*lql)**2
        dGdYi = ((Yk-Yj)*lpl*lql-np.dot(p,q)*(lql/lpl)*(Yi-Yj))/(lpl*lql)**2
        dGdZi = ((Zk-Zj)*lpl*lql-np.dot(p,q)*(lql/lpl)*(Zi-Zj))/(lpl*lql)**2
        
        dGdXj = ((2*Xj-Xi-Xk)*lpl*lql-np.dot(p,q)*(lql/lpl)*(Xj-Xi)-np.dot(p,q)*(lpl/lql)*(Xj-Xk))/(lpl*lql)**2
        dGdYj = ((2*Yj-Yi-Yk)*lpl*lql-np.dot(p,q)*(lql/lpl)*(Yj-Yi)-np.dot(p,q)*(lpl/lql)*(Yj-Yk))/(lpl*lql)**2
        dGdZj = ((2*Zj-Zi-Zk)*lpl*lql-np.dot(p,q)*(lql/lpl)*(Zj-Zi)-np.dot(p,q)*(lpl/lql)*(Zj-Zk))/(lpl*lql)**2
        
        dGdXk = ((Xi-Xj)*lpl*lql-np.dot(p,q)*(lpl/lql)*(Xk-Xj))/(lpl*lql)**2
        dGdYk = ((Yi-Yj)*lpl*lql-np.dot(p,q)*(lpl/lql)*(Yk-Yj))/(lpl*lql)**2
        dGdZk = ((Zi-Zj)*lpl*lql-np.dot(p,q)*(lpl/lql)*(Zk-Zj))/(lpl*lql)**2
        
        hessian = _diag_offdiagf2(hessian, K_theta, dGdXi, dGdYi, dGdZi, dGdXj, dGdYj, dGdZj, G, i, j)
        hessian = _offdiag_offdiagf2(hessian, K_theta, dGdXi, dGdYi, dGdZi, dGdXj, dGdYj, dGdZj, G, i, j)
        hessian = _diag_offdiagf2(hessian, K_theta, dGdXj, dGdYj, dGdZj, dGdXi, dGdYi, dGdZi, G, j, i)
        hessian = _offdiag_offdiagf2(hessian, K_theta, dGdXj, dGdYj, dGdZj, dGdXi, dGdYi, dGdZi, G, j, i)
        
        hessian = _diag_offdiagf2(hessian, K_theta, dGdXj, dGdYj, dGdZj, dGdXk, dGdYk, dGdZk, G, j, k)
        hessian = _offdiag_offdiagf2(hessian, K_theta, dGdXj, dGdYj, dGdZj, dGdXk, dGdYk, dGdZk, G, j, k)
        hessian = _diag_offdiagf2(hessian, K_theta, dGdXk, dGdYk, dGdZk, dGdXj, dGdYj, dGdZj, G, k, j)
        hessian = _offdiag_offdiagf2(hessian, K_theta, dGdXk, dGdYk, dGdZk, dGdXj, dGdYj, dGdZj, G, k, j)
        
        hessian = _diag_offdiagf2(hessian, K_theta, dGdXi, dGdYi, dGdZi, dGdXk, dGdYk, dGdZk, G, i, k)
        hessian = _offdiag_offdiagf2(hessian, K_theta, dGdXi, dGdYi, dGdZi, dGdXk, dGdYk, dGdZk, G, i, k)
        hessian = _diag_offdiagf2(hessian, K_theta, dGdXk, dGdYk, dGdZk, dGdXi, dGdYi, dGdZi, G, k, i)
        hessian = _offdiag_offdiagf2(hessian, K_theta, dGdXk, dGdYk, dGdZk, dGdXi, dGdYi, dGdZi, G, k, i)
        
        hessian = _diag_offdiagf2(hessian, K_theta, dGdXi, dGdYi, dGdZi, dGdXi, dGdYi, dGdZi, G, i, i)
        hessian = _offdiag_offdiagf2(hessian, K_theta, dGdXi, dGdYi, dGdZi, dGdXi, dGdYi, dGdZi, G, i, i)
        
        hessian = _diag_offdiagf2(hessian, K_theta, dGdXj, dGdYj, dGdZj, dGdXj, dGdYj, dGdZj, G, j, j)
        hessian = _offdiag_offdiagf2(hessian, K_theta, dGdXj, dGdYj, dGdZj, dGdXj, dGdYj, dGdZj, G, j, j)
        
        hessian = _diag_offdiagf2(hessian, K_theta, dGdXk, dGdYk, dGdZk, dGdXk, dGdYk, dGdZk, G, k, k)
        hessian = _offdiag_offdiagf2(hessian, K_theta, dGdXk, dGdYk, dGdZk, dGdXk, dGdYk, dGdZk, G, k, k)
        
        i = i + 1
    
    return hessian


############################### Third Element ####################################################

def _glength(X): return np.sqrt(X[0]**2+X[1]**2+X[2]**2)

def _derivative1_f3(C1, C2, C3, el1, el2, el3, el4, el5, el6, el7, el8, el9):
    """Super element derivative calculation
        """
    ret1 = (2*C2*(el9-el8)+2*C3*(el5-el6))/(2*np.sqrt(C1**2+C2**2+C3**2))
    ret2 = (2*C1*(el8-el9)+2*C3*(el3-el2))/(2*np.sqrt(C1**2+C2**2+C3**2))
    ret3 = (2*C1*(el6-el5)+2*C2*(el2-el3))/(2*np.sqrt(C1**2+C2**2+C3**2))
    ret4 = (2*C2*(el7-el9)+2*C3*(el6-el4))/(2*np.sqrt(C1**2+C2**2+C3**2))
    ret5 = (2*C1*(el9-el7)+2*C3*(el1-el3))/(2*np.sqrt(C1**2+C2**2+C3**2))
    ret6 = (2*C1*(el4-el6)+2*C2*(el3-el1))/(2*np.sqrt(C1**2+C2**2+C3**2))
    ret7 = (2*C2*(el8-el7)+2*C3*(el4-el5))/(2*np.sqrt(C1**2+C2**2+C3**2))
    ret8 = (2*C1*(el7-el8)+2*C3*(el2-el1))/(2*np.sqrt(C1**2+C2**2+C3**2))
    ret9 = (2*C1*(el5-el4)+2*C2*(el1-el2))/(2*np.sqrt(C1**2+C2**2+C3**2))
    return ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8, ret9

def _g_derivativef3(dv1dXs, dv1dYs, dv1dZs, dv2dXs, dv2dYs, dv2dZs, v1, v2, lv1l, lv2l, dlv1ldXs, dlv1ldYs, dlv1ldZs, dlv2ldXs, dlv2ldYs, dlv2ldZs):
    """Calculating G derivatives values"""
    ret1=((np.dot(dv1dXs,v2)+np.dot(dv2dXs,v1))*lv1l*lv2l-np.dot(v1,v2)*(dlv1ldXs*lv2l+dlv2ldXs*lv1l))/(lv1l*lv2l)**2
    ret2=((np.dot(dv1dYs,v2)+np.dot(dv2dYs,v1))*lv1l*lv2l-np.dot(v1,v2)*(dlv1ldYs*lv2l+dlv2ldYs*lv1l))/(lv1l*lv2l)**2
    ret3=((np.dot(dv1dZs,v2)+np.dot(dv2dZs,v1))*lv1l*lv2l-np.dot(v1,v2)*(dlv1ldZs*lv2l+dlv2ldZs*lv1l))/(lv1l*lv2l)**2
    return ret1, ret2, ret3

def _diag_offdiagf3(hessian_inp, K_phi, dGdXu, dGdXv, dGdYu, dGdYv, dGdZu, dGdZv, G, u, v):
    """diagonals of off-diagonal super elements (3rd term)
        """
    hessian_inp[u*3,v*3]     = hessian_inp[u*3,v*3]      +  (2*K_phi)/(1-G**2)*dGdXu*dGdXv
    hessian_inp[u*3+1,v*3+1] = hessian_inp[u*3+1,v*3+1]  +  (2*K_phi)/(1-G**2)*dGdYu*dGdYv
    hessian_inp[u*3+2,v*3+2] = hessian_inp[u*3+2,v*3+2]  +  (2*K_phi)/(1-G**2)*dGdZu*dGdZv
    return hessian_inp

def _offdiag_offdiagf3(hessian_inp, K_phi, dGdXu, dGdXv, dGdYu, dGdYv, dGdZu, dGdZv, G, u, v):
    """off-diagonals of off-diagonal super elements (3rd term)
        """
    hessian_inp[u*3,v*3+1]   = hessian_inp[u*3,v*3+1]     +   (2*K_phi)/(1-G**2)*dGdXu*dGdYv
    hessian_inp[u*3,v*3+2]   = hessian_inp[u*3,v*3+2]     +   (2*K_phi)/(1-G**2)*dGdXu*dGdZv
    hessian_inp[u*3+1,v*3]   = hessian_inp[u*3+1,v*3]     +   (2*K_phi)/(1-G**2)*dGdYu*dGdXv
    hessian_inp[u*3+1,v*3+2] = hessian_inp[u*3+1,v*3+2]   +   (2*K_phi)/(1-G**2)*dGdYu*dGdZv
    hessian_inp[u*3+2,v*3]   = hessian_inp[u*3+2,v*3]     +   (2*K_phi)/(1-G**2)*dGdZu*dGdXv
    hessian_inp[u*3+2,v*3+1] = hessian_inp[u*3+2,v*3+1]   +   (2*K_phi)/(1-G**2)*dGdZu*dGdYv
    return hessian_inp

def _hessian_diag_offdiagf3(hessian_inp, K_phi, dGdXm, dGdXn, dGdYm, dGdYn, dGdZm, dGdZn, G, m, n, diag):
    """combining diagonals of off-diagonal and of off-diagonals of off-diagonal super elements (3rd term)
        """
    hessian_inp = _diag_offdiagf3(hessian_inp, K_phi, dGdXm, dGdXn, dGdYm, dGdYn, dGdZm, dGdZn, G, m, n)
    hessian_inp = _offdiag_offdiagf3(hessian_inp, K_phi, dGdXm, dGdXn, dGdYm, dGdYn, dGdZm, dGdZn, G, m, n)
    if diag==False:
        hessian_inp = _diag_offdiagf3(hessian_inp, K_phi, dGdXn, dGdXm, dGdYn, dGdYm, dGdZn, dGdZm, G, n, m)
        hessian_inp = _offdiag_offdiagf3(hessian_inp, K_phi, dGdXn, dGdXm, dGdYn, dGdYm, dGdZn, dGdZm, G, n, m)
    return hessian_inp

def _thirdTerm(hessian,caArray,distance,numOfResidues,K_phi1,K_phi3):
    """derive the hessian of the third term (off diagonal)
        """
    K_phi=K_phi1/2+K_phi3*9/2
    i=0
    while i < len(caArray):
        
        j = i+1; k = i+2; l = i+3
        if l == len(caArray):
            break
        
        Xi=caArray[i,0]
        Yi=caArray[i,1]
        Zi=caArray[i,2]
        Xj=caArray[j,0]
        Yj=caArray[j,1]
        Zj=caArray[j,2]
        Xk=caArray[k,0]
        Yk=caArray[k,1]
        Zk=caArray[k,2]
        Xl=caArray[l,0]
        Yl=caArray[l,1]
        Zl=caArray[l,2]
        
        a=caArray[j,0:3]-caArray[i,0:3]
        b=caArray[k,0:3]-caArray[j,0:3]
        c=caArray[l,0:3]-caArray[k,0:3]
        
        v1=np.cross(a,b)
        v2=np.cross(b,c)
        lv1l=_glength(v1)
        lv2l=_glength(v2)
        G=np.dot(v1,v2)/(lv1l*lv2l)
        
        dv1dXi=np.array((0, Zk-Zj, Yj-Yk));
        dv1dYi=np.array((Zj-Zk, 0, Xk-Xj));
        dv1dZi=np.array((Yk-Yj, Xj-Xk, 0))
        dv1dXj=np.array((0, Zi-Zk, Yk-Yi));
        dv1dYj=np.array((Zk-Zi, 0, Xi-Xk));
        dv1dZj=np.array((Yi-Yk, Xk-Xi, 0))
        dv1dXk=np.array((0, Zj-Zi, Yi-Yj));
        dv1dYk=np.array((Zi-Zj, 0, Xj-Xi));
        dv1dZk=np.array((Yj-Yi, Xi-Xj, 0))
        dv1dXl=np.array((0, 0, 0));
        dv1dYl=np.array((0, 0, 0));
        dv1dZl=np.array((0, 0, 0))
        dv2dXi=np.array((0, 0, 0));
        dv2dYi=np.array((0, 0, 0));
        dv2dZi=np.array((0, 0, 0))
        dv2dXj=np.array((0, Zl-Zk, Yk-Yl));
        dv2dYj=np.array((Zk-Zl, 0, Xl-Xk));
        dv2dZj=np.array((Yl-Yk, Xk-Xl, 0))
        dv2dXk=np.array((0, Zj-Zl, Yl-Yj));
        dv2dYk=np.array((Zl-Zj, 0, Xj-Xl));
        dv2dZk=np.array((Yj-Yl, Xl-Xj, 0))
        dv2dXl=np.array((0, Zk-Zj, Yj-Yk));
        dv2dYl=np.array((Zj-Zk, 0, Xk-Xj));
        dv2dZl=np.array((Yk-Yj, Xj-Xk, 0))
        
        K1=(Yj-Yi)*(Zk-Zj)-(Yk-Yj)*(Zj-Zi)
        K2=(Xk-Xj)*(Zj-Zi)-(Xj-Xi)*(Zk-Zj)
        K3=(Xj-Xi)*(Yk-Yj)-(Xk-Xj)*(Yj-Yi)
        
        dlv1ldXi, dlv1ldYi, dlv1ldZi, dlv1ldXj, dlv1ldYj, dlv1ldZj, dlv1ldXk, dlv1ldYk, dlv1ldZk = _derivative1_f3(K1, K2, K3, Xi, Xj, Xk, Yi, Yj, Yk, Zi, Zj, Zk)
        
        dlv1ldXl=0
        dlv1ldYl=0
        dlv1ldZl=0
        dlv2ldXi=0
        dlv2ldYi=0
        dlv2ldZi=0
        
        L1=(Yk-Yj)*(Zl-Zk)-(Yl-Yk)*(Zk-Zj)
        L2=(Xl-Xk)*(Zk-Zj)-(Xk-Xj)*(Zl-Zk)
        L3=(Xk-Xj)*(Yl-Yk)-(Xl-Xk)*(Yk-Yj)
        
        
        dlv2ldXj, dlv2ldYj, dlv2ldZj, dlv2ldXk, dlv2ldYk, dlv2ldZk, dlv2ldXl, dlv2ldYl, dlv2ldZl = _derivative1_f3(L1, L2, L3, Xj, Xk, Xl, Yj, Yk, Yl, Zj, Zk, Zl)
        
        #dG/dXi   dG/dYi  dG/dYi
        dGdXi, dGdYi, dGdZi = _g_derivativef3(dv1dXi, dv1dYi, dv1dZi, dv2dXi, dv2dYi, dv2dZi, v1, v2, lv1l, lv2l, dlv1ldXi, dlv1ldYi, dlv1ldZi, dlv2ldXi, dlv2ldYi, dlv2ldZi)
        #dG/dXj   dG/dYj  dG/dYj
        dGdXj, dGdYj, dGdZj = _g_derivativef3(dv1dXj, dv1dYj, dv1dZj, dv2dXj, dv2dYj, dv2dZj, v1, v2, lv1l, lv2l, dlv1ldXj, dlv1ldYj, dlv1ldZj, dlv2ldXj, dlv2ldYj, dlv2ldZj)
        #dG/dXk   dG/dYk  dG/dYk
        dGdXk, dGdYk, dGdZk = _g_derivativef3(dv1dXk, dv1dYk, dv1dZk, dv2dXk, dv2dYk, dv2dZk, v1, v2, lv1l, lv2l, dlv1ldXk, dlv1ldYk, dlv1ldZk, dlv2ldXk, dlv2ldYk, dlv2ldZk)
        #dG/dXl   dG/dYl  dG/dYl
        dGdXl, dGdYl, dGdZl = _g_derivativef3(dv1dXl, dv1dYl, dv1dZl, dv2dXl, dv2dYl, dv2dZl, v1, v2, lv1l, lv2l, dlv1ldXl, dlv1ldYl, dlv1ldZl, dlv2ldXl, dlv2ldYl, dlv2ldZl)
        
        # example - Hij
        # d^2V/dXidXj  d^2V/dXidYj  d^2V/dXidZj
        # d^2V/dYidXj  d^2V/dYidYj  d^2V/dYidZj
        # d^2V/dZidXj  d^2V/dZidYj  d^2V/dZidZj
        #
        # and Hji
        # d^2V/dXjdXi  d^2V/dXjdYi  d^2V/dXjdZi
        # d^2V/dYjdXi  d^2V/dYjdYi  d^2V/dYjdZi
        # d^2V/dZjdXi  d^2V/dZjdYi  d^2V/dZjdZi
        #
        # Similarly all the other Hessian elements are derived
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXi, dGdXj, dGdYi, dGdYj, dGdZi, dGdZj, G, i, j, diag=False)
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXi, dGdXl, dGdYi, dGdYl, dGdZi, dGdZl, G, i, l, diag=False)
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXk, dGdXj, dGdYk, dGdYj, dGdZk, dGdZj, G, k, j, diag=False)
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXi, dGdXk, dGdYi, dGdYk, dGdZi, dGdZk, G, i, k, diag=False)
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXl, dGdXj, dGdYl, dGdYj, dGdZl, dGdZj, G, l, j, diag=False)
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXl, dGdXk, dGdYl, dGdYk, dGdZl, dGdZk, G, l, k, diag=False)
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXi, dGdXi, dGdYi, dGdYi, dGdZi, dGdZi, G, i, i, diag=True)
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXj, dGdXj, dGdYj, dGdYj, dGdZj, dGdZj, G, j, j, diag=True)
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXk, dGdXk, dGdYk, dGdYk, dGdZk, dGdZk, G, k, k, diag=True)
        hessian = _hessian_diag_offdiagf3(hessian, K_phi, dGdXl, dGdXl, dGdYl, dGdYl, dGdZl, dGdZl, G, l, l, diag=True)
        
        i = i + 1
    
    return hessian


############################### Fourth Element ###################################################

def _derivative_consf4(el1, el2, Epsilon, distijsqr): return 120*Epsilon*el1*el2/distijsqr

def _fourthTerm(caArray, hessian, Epsilon, distance_mat):
    """derive the hessian of the fourth term (off diagonal)
        """
    i=0
    lim = len(caArray)
    while i < lim:
        j=0
        while j < lim:
            if abs(i-j)>3:
                bx=caArray[i,0] - caArray[j,0]
                by=caArray[i,1] - caArray[j,1]
                bz=caArray[i,2] - caArray[j,2]
                distijsqr = distance_mat[i, j]**4
                
                hessian[i*3,j*3] = hessian[i*3,j*3]-_derivative_consf4(bx, bx, Epsilon, distijsqr)
                hessian[i*3+1,j*3+1] = hessian[i*3+1,j*3+1]-_derivative_consf4(by, by, Epsilon, distijsqr)
                hessian[i*3+2,j*3+2] = hessian[i*3+2,j*3+2]-_derivative_consf4(bz, bz, Epsilon, distijsqr)
                
                hessian[i*3,j*3+1] = hessian[i*3,j*3+1]-_derivative_consf4(bx, by, Epsilon, distijsqr)
                hessian[i*3,j*3+2] = hessian[i*3,j*3+2]-_derivative_consf4(bx, bz, Epsilon, distijsqr)
                hessian[i*3+1,j*3] = hessian[i*3+1,j*3]-_derivative_consf4(by, bx, Epsilon, distijsqr)
                hessian[i*3+1,j*3+2] = hessian[i*3+1,j*3+2]-_derivative_consf4(by, bz, Epsilon, distijsqr)
                hessian[i*3+2,j*3] = hessian[i*3+2,j*3]-_derivative_consf4(bx, bz, Epsilon, distijsqr)
                hessian[i*3+2,j*3+1] = hessian[i*3+2,j*3+1]-_derivative_consf4(by, bz, Epsilon, distijsqr)
                
                hessian[i*3,i*3] = hessian[i*3,i*3]+_derivative_consf4(bx, bx, Epsilon, distijsqr)
                hessian[i*3+1,i*3+1] = hessian[i*3+1,i*3+1]+_derivative_consf4(by, by, Epsilon, distijsqr)
                hessian[i*3+2,i*3+2] = hessian[i*3+2,i*3+2]+_derivative_consf4(bz, bz, Epsilon, distijsqr)
                
                hessian[i*3,i*3+1] = hessian[i*3,i*3+1]+_derivative_consf4(bx, by, Epsilon, distijsqr)
                hessian[i*3,i*3+2] = hessian[i*3,i*3+2]+_derivative_consf4(bx, bz, Epsilon, distijsqr)
                hessian[i*3+1,i*3] = hessian[i*3+1,i*3]+_derivative_consf4(by, bx, Epsilon, distijsqr)
                hessian[i*3+1,i*3+2] = hessian[i*3+1,i*3+2]+_derivative_consf4(by, bz, Epsilon, distijsqr)
                hessian[i*3+2,i*3] = hessian[i*3+2,i*3]+_derivative_consf4(bx, bz, Epsilon, distijsqr)
                hessian[i*3+2,i*3+1] = hessian[i*3+2,i*3+1]+_derivative_consf4(by, bz, Epsilon, distijsqr)
            
            j = j + 1
        
        i = i + 1
    
    
    return hessian


def _sort_E(E):
    """sorting eigenvector and eigenvalues obtained from linalg eigh module
        """
    e_dict =  {index: key for index, key in enumerate(E)}
    return sorted(e_dict.iteritems(), key=lambda (k,v): (v,k))

def _condense(rlistObj): return [sum(rlistObj[x:x+3]) for x in range(0, len(rlistObj),3)]

##################################################################################################
    
if __name__ == "__main__":
    pass






