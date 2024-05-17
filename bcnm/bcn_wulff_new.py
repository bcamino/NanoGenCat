from __future__ import print_function
#from ctypes import Structure
from distutils.log import debug
#import os, time
#import subprocess
import copy
import numpy as np
import pandas as pd
#import glob

from os import remove
from re import findall
from random import seed,shuffle,choice
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import euclidean
#from scipy.spatial import ConvexHull
#import scipy.constants as constants
#from itertools import combinations,product
#from math import sqrt

from ase import Atoms,Atom
from ase.atoms import symbols2numbers
#from ase.neighborlist import NeighborList
from ase.utils import basestring
#from ase.cluster.factory import GCD
#from ase.visualize import view
#from ase.io import write,read
#from ase.data import chemical_symbols,covalent_radii
#from ase.spacegroup import Spacegroup
#from ase.build import surface as slabBuild

#from pymatgen.analysis.wulff import WulffShape
#from pymatgen.symmetry.analyzer import PointGroupAnalyzer
#from pymatgen.io.ase import AseAtomsAdaptor
#from pymatgen.core.structure import IMolecule
#from pymatgen.core.surface import SlabGenerator,generate_all_slabs
#from pymatgen.core.structure import Molecule

from pymatgen.analysis.wulff import WulffShape
from scipy.spatial import Delaunay, ConvexHull


nonMetals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                  'Si', 'P', 'S', 'Cl', 'Ar',
                  'Ge', 'As', 'Se', 'Br', 'Kr',
                  'Sb', 'Te', 'I', 'Xe',
                  'Po', 'At', 'Rn']
nonMetalsNumbers=symbols2numbers(nonMetals)

delta = 1e-10
_debug = False
seed(42)

class Nanoparticle:
    """
    A class to represent a Nanoparticle.

    Attributes:
        structure (pymatgen.Structure or ase.Atoms): The atomic structure of the nanoparticle.
    """

    def __init__(self, structure):
        """
        Initializes the Nanoparticle object.

        Args:
            structure (pymatgen.Structure or ase.Atoms): A pymatgen Structure object or an ASE Atoms object.
        """
        self.structure = structure

    def get_np0(self, surfaces, energies, size, center=None):
        """
        Builds the initial nanoparticle (np0).

        Args:
            surfaces (list): List of surfaces for the nanoparticle.
            energies (list): List of surface energies corresponding to the surfaces.
            size (int): Size of the nanoparticle.
            center (tuple, optional): Center of the nanoparticle. Defaults to None.

        Returns:
            np0: The initial nanoparticle object.
        """
        np0 = build_np0(self.structure, surfaces, energies, size, center=center)
        return np0

    def make_stoichiometric(self, surfaces, energies, size, maxiter=100, center=None,
                            optimisation='random', strategy='com', charges={}, N_np=100,
                            rounding='closest', remove_undercoord_first=True,
                            totalReduced=False, coordinationLimit='half', polar=False,
                            termNature='non-metal', neutralize=False, inertiaMoment=False,
                            debug=0, stoichiometryMethod=1, wl_method='hybridMethod'):
        """
        Generates a stoichiometric nanoparticle.

        Args:
            surfaces (list): List of surfaces for the nanoparticle.
            energies (list): List of surface energies corresponding to the surfaces.
            size (int): Size of the nanoparticle.
            maxiter (int, optional): Maximum number of iterations for the optimization. Defaults to 100.
            center (tuple, optional): Center of the nanoparticle. Defaults to None.
            optimisation (str, optional): Optimization method. Defaults to 'random'.
            strategy (str, optional): Strategy to use in optimization. Defaults to 'com'.
            charges (dict, optional): Charges dictionary. Defaults to {}.
            N_np (int, optional): Number of nanoparticles. Defaults to 100.
            rounding (str, optional): Rounding method. Defaults to 'closest'.
            remove_undercoord_first (bool, optional): Remove undercoordinated atoms first. Defaults to True.
            totalReduced (bool, optional): Total reduced flag. Defaults to False.
            coordinationLimit (str, optional): Coordination limit. Defaults to 'half'.
            polar (bool, optional): Polar flag. Defaults to False.
            termNature (str, optional): Termination nature. Defaults to 'non-metal'.
            neutralize (bool, optional): Neutralize flag. Defaults to False.
            inertiaMoment (bool, optional): Inertia moment flag. Defaults to False.
            debug (int, optional): Debug level. Defaults to 0.
            stoichiometryMethod (int, optional): Stoichiometry method. Defaults to 1.
            wl_method (str, optional): Method for weight loss. Defaults to 'hybridMethod'.

        Returns:
            tuple: Contains the stoichiometric nanoparticles and the quality index.
        """
        np0 = build_np0(self.structure, surfaces, energies, size, center=center, rounding='closest')

        if is_stoichiometric(np0, self.structure):
            print('The NP0 is stoichiometric')
            return np0

        undercoordinated_atoms, index_combinatorial = get_atoms_to_remove(np0, self.structure)

        sites_to_remove_all, quality_index_all = make_best_np(np0, undercoordinated_atoms, index_combinatorial, 
                                                             self.structure, optimisation=optimisation,
                                                             strategy=strategy, charges=charges, maxiter=maxiter, N_np=N_np)

        nps = build_stoich_np(np0, sites_to_remove_all)
        
        return nps, quality_index_all

def build_np0(structure_init,surfaces,energies,size,rounding='closest',center=None):
    """
    Builds the initial nanoparticle (np0) with a size closest to the requested one.

    Args:
        structure_init (pymatgen.Structure): Initial atomic structure of the nanoparticle.
        surfaces (list): List of surfaces for the nanoparticle.
        energies (list): List of surface energies corresponding to the surfaces.
        size (int): Desired size of the nanoparticle.
        rounding (str, optional): Rounding method. Defaults to 'closest'.
        center (tuple, optional): Center of the nanoparticle. Defaults to None.

    Returns:
        pymatgen.Structure: The initial nanoparticle structure.
    """
    
    structure = copy.deepcopy(structure_init)
    # Most/less abundant atoms
        
    laa = get_less_abundant_atom(structure_init)
    maa = get_more_abundant_atom(structure_init)
    
    #Get max coordination
    structure_tmp = copy.deepcopy(structure_init)
    structure_tmp.make_supercell(np.identity(3)*2)
    max_coord = np.max(get_coordination(structure_tmp))
    
    wulffshape = WulffShape(structure.lattice, surfaces, energies)

    atom_density = structure.num_sites/structure.volume

    ###Add rounding
    
    if center is not None:
        structure_center = copy.deepcopy(structure_init)
        structure_center.translate_sites(np.arange(structure_init.num_sites),center,
                                     frac_coords=True,to_unit_cell= False)
    else:
        structure_center = copy.deepcopy(structure_init)
        
    if type(size) == int:
        new_volume = size/atom_density
        ratio = new_volume**(1/3)/wulffshape.volume**(1/3)

        if debug == True:
            shape = ConvexHull((wulffshape.wulff_convex.points/(wulffshape.volume**(1/3)))*(new_volume**(1/3)))        
            print('NP0 size = ',shape.volume*atom_density)

        shape = Delaunay(wulffshape.wulff_convex.points*(ratio))
        
        #loop to make sure to use the smallest supercell
        num_sites = 0
        for i in range(1,10):
            structure = copy.deepcopy(structure_center)
            # Generate the supercell      
            supercell_matrix = np.zeros((3,3)) ###Make this better
            expansion = np.ceil((np.max(shape.points,axis=0)-np.min(shape.points,axis=0))/np.array(structure.lattice.abc))+i
            np.fill_diagonal(supercell_matrix,expansion)
            structure.make_supercell(supercell_matrix)

            # Center around the center of mass
            center_of_mass = np.average(structure.cart_coords,axis=0)
            structure.translate_sites(np.arange(structure.num_sites),
                                      -center_of_mass,frac_coords=False,
                                      to_unit_cell= False)

            # Generate the list of atoms to be removed
            np0_sites = []
            sites_to_remove = []
            for i,coord in enumerate(structure.cart_coords):
                if (shape.find_simplex(coord) >= 0) == False:
                    sites_to_remove.append(i)
                if (shape.find_simplex(coord) >= 0) == True:
                    np0_sites.append(i)
            sites_to_remove = np.array(sites_to_remove)        
            np0_sites = np.array(np0_sites)

            #Surface less abundant atom:
            laa_surface_sites = np0_sites[np.where(np.array(structure.atomic_numbers)[np0_sites] == laa)[0]]

            maa_surface_sites = (np.argsort(structure.distance_matrix[laa_surface_sites],axis=1)[:,1:max_coord+1])


            # Create a boolean mask indicating which elements to keep
            mask = ~np.isin(sites_to_remove, maa_surface_sites)

            # Apply the mask to the original array
            sites_to_remove_new = sites_to_remove[mask]

            structure.remove_sites(sites_to_remove_new)

            # Translate the sites back to their orginal position
            structure.translate_sites(np.arange(structure.num_sites),
                                      center_of_mass,frac_coords=False,
                                      to_unit_cell= False)

            np0 = copy.deepcopy(structure)
            if np0.num_sites > num_sites:
                num_sites = copy.deepcopy(np0.num_sites)
            else:
                return np0
    print('The supercell expansion could not be found in 10 cycles')
    return None


def get_atoms_to_remove(np0,initial_structure,remove_undercoord_first=True):

    """
    Identifies atoms to be removed to achieve stoichiometry.

    Args:
        np0 (pymatgen.Structure): Initial nanoparticle structure.
        initial_structure (pymatgen.Structure): Initial atomic structure.
        remove_undercoord_first (bool, optional): Remove undercoordinated atoms first. Defaults to True.

    Returns:
        tuple: Undercoordinated atoms and combinatorial indices of atoms to be removed.
    """

    from scipy.special import comb
    
    nano = copy.deepcopy(np0)

    atomic_numbers = np.array(nano.atomic_numbers)
    atom_coord = get_coordination(nano)
    excess_atom_number = np.unique(atomic_numbers)[np.where(get_excess_atoms(np0,initial_structure=initial_structure) != 0)][0]

    excess_atom_index = np.where(atomic_numbers == excess_atom_number)[0]

    excess_atom_coord = atom_coord[excess_atom_index]
    unique_coord, coord_count = np.unique(excess_atom_coord, return_counts=True)


    max_coord = unique_coord[-1]
    n_excess_atom = np.max(get_excess_atoms(nano, initial_structure))

    if np.sum(coord_count[:-1]) < n_excess_atom:
        print('There are not enough surface atoms')
        

    undercoordinated_atom_index = [False]*len(excess_atom_index)
    for i, n_atoms_in_coord in enumerate(coord_count):

        if n_excess_atom - n_atoms_in_coord > 0:
            n_excess_atom -= n_atoms_in_coord
            undercoordinated_atom_index += excess_atom_coord == unique_coord[i]
        else:
            coord_combinatorial = unique_coord[i]
            break
    
    undercoordinated_atoms = excess_atom_index[undercoordinated_atom_index]
    # This implies we remove all the undercoordinated atoms first and then when we get to the group
    # from which we need to choose which ones to remove only a part
    if remove_undercoord_first == True:
        index_combinatorial = excess_atom_index[excess_atom_coord == coord_combinatorial]
        undercoordinated_atoms = excess_atom_index[undercoordinated_atom_index]
    else:
        index_combinatorial = excess_atom_index[excess_atom_coord <= coord_combinatorial]
        undercoordinated_atoms = []


    #print(index_combinatorial)
    n_configurations = comb(len(index_combinatorial),n_excess_atom)
    print("There are %s possible configurations."%f"{int(n_configurations):e}")
    
    return undercoordinated_atoms,index_combinatorial


def make_best_np(np0, undercoordinated_atoms, index_combinatorial, initial_structure,optimisation='random',strategy='com',charges={}, maxiter=1000,
                       N_np=100):
    
    """
    Optimizes the nanoparticle structure.

    Args:
        np0 (pymatgen.Structure): Initial nanoparticle structure.
        undercoordinated_atoms (array): List of undercoordinated atoms.
        index_combinatorial (array): Indices of atoms to be removed.
        initial_structure (pymatgen.Structure): Initial atomic structure.
        optimisation (str, optional): Optimisation method. Defaults to 'random'.
        strategy (str, optional): Optimisation strategy. Defaults to 'com'.
        charges (dict, optional): Atomic charges. Defaults to {}.
        maxiter (int, optional): Maximum iterations. Defaults to 1000.
        N_np (int, optional): Number of nanoparticles. Defaults to 100.

    Returns:
        tuple: Sites to remove and quality index of all configurations.
    """
    
    if strategy == 'dipole' and len(charges) == 0:
        print('ERROR: please select charges to use the dipole strategy')
    
    nano_initial = copy.deepcopy(np0)
    nano_initial.translate_sites(np.arange(nano_initial.num_sites),
                                 -np.mean(nano_initial.frac_coords,axis=0),to_unit_cell=False)
    
    nano_test = copy.deepcopy(np0)
    remove_atoms = []
    
    # Keep only the external layer atoms
    for i in np.arange(nano_test.num_sites):
        if i not in index_combinatorial:
            remove_atoms.append(i)
    
    inner_indeces = np.delete(np.arange(nano_initial.num_sites), index_combinatorial)
    
    # Center the atoms to remove
    nano_test.remove_sites(remove_atoms)
    nano_test.translate_sites(np.arange(nano_test.num_sites),-np.mean(nano_test.frac_coords,axis=0),to_unit_cell=False)
    n_atoms_to_remove = np.max(get_excess_atoms(np0,initial_structure=initial_structure))-len(undercoordinated_atoms)

    sites_to_remove_all = []
    quality_index_all = np.empty(0)
    
    # OPTIMISATION ALGORITHM
    
    cart_coords = nano_initial.cart_coords
    
#     if optimisation == 'exhaustive search':
#         maxiter = len
    
    sites_to_remove_list = optimisation_algorithm(index_combinatorial,n_atoms_to_remove, 
                                                  optimisation=optimisation, maxiter=maxiter)
        
    
    for i,sites_to_remove in enumerate(sites_to_remove_list):
                
        #sites_to_remove = optimisation_algorithm(index_combinatorial,n_atoms_to_remove, optimisation='random')
        
        
    # QUALITY INDEX    
        
        quality_index = objective_function(undercoordinated_atoms,sites_to_remove,nano_initial,strategy=strategy,charges=charges)
        
        # SAVE THE NPs
        
        #if there are no elements yet, add the first one
        if len(quality_index_all) == 0:
            quality_index_all = np.append(quality_index_all, quality_index)

            sites_to_remove_all.append(sites_to_remove)
            
        # if the list already has the desired number of NPs and the value is larger than the last one, pass
        elif len(quality_index_all) == N_np and np.round(quality_index,5) > np.round(quality_index_all[-1],5):
            pass
        else:
            #THIS MIGHT BE REDUNDANT
            if len(quality_index_all) < N_np or len(quality_index_all) > N_np and quality_index < quality_index_all[-1]:
                
                # Find the index where the element should be inserted
                index = np.searchsorted(quality_index_all, quality_index)

                # check if that configuration is already there
                
                # if the index is last, add to the list (this works because it they were the same it would be in 
                #potision len(...))
                if index > len(quality_index_all):
                    # Insert the element at the calculated index
                    quality_index_all = np.insert(quality_index_all, index, quality_index)

                    sites_to_remove_all.insert(index,sites_to_remove)

                # if the index is not last, check if it's already there    
                elif index < len(quality_index_all):
                    
                    # check if it's already there
                    if np.all(sites_to_remove_all[index] == sites_to_remove):
                        pass
                    
                    elif np.round(quality_index_all[index],5) == np.round(quality_index,5):
                        
                        # singularisation (check descriptor)
                        np_test_1 = copy.deepcopy(np0)
                        np_test_2 = copy.deepcopy(np0)
                        
                        cm_1 = get_coulomb_matrix(np_test_1)
                        cm_2 = get_coulomb_matrix(np_test_2)
                        
                        if np.all(cm_1 == cm_2) == False:

                            # Insert the element at the calculated index
                            quality_index_all = np.insert(quality_index_all, index, quality_index)

                            sites_to_remove_all.insert(index,sites_to_remove)
#                         elif np.all(cm_1 == cm_2) == True:
#                             print(sites_to_remove,sites_to_remove_all[index])
#                             print(np.sum(cm_1 == cm_2))
                    else:
                         #Insert the element at the calculated index
                        quality_index_all = np.insert(quality_index_all, index, quality_index)

                        sites_to_remove_all.insert(index,sites_to_remove)

                if len(quality_index_all) > N_np:

                    quality_index_all = quality_index_all[:-1]
                    sites_to_remove_all = sites_to_remove_all[:-1]

#         if quality_index <0.1:
#             print(quality_index,np.mean(nano_test.cart_coords[sites_random],axis=0))
#             nano_final = copy.deepcopy(nano)
#             nano_final.remove_sites(index_combinatorial[sites_to_remove])
#             #vview(nano_final)
    sites_to_remove_all = np.array(sites_to_remove_all)
    
    vector_repeated = np.tile(undercoordinated_atoms, (sites_to_remove_all.shape[0], 1))

    # Stack the vector and array horizontally
    sites_to_remove_all = np.hstack((vector_repeated, sites_to_remove_all))
    return sites_to_remove_all,quality_index_all


def build_stoich_np(np0,sites_to_remove):
    """
    Constructs stoichiometric nanoparticles by removing specified sites.

    Args:
        np0 (pymatgen.Structure): Initial nanoparticle structure.
        sites_to_remove (list): List of sites to be removed.

    Returns:
        list: List of pymatgen.Structure objects representing the stoichiometric nanoparticles.
    """
    nps = []
    for sites in sites_to_remove:
        np = copy.deepcopy(np0)
        np.remove_sites(sites)
        nps.append(np)
    return nps


def optimisation_algorithm(index_combinatorial,n_items_to_select, optimisation, maxiter):
    
    """
    Generates a list of sites to remove based on the optimisation strategy.

    Args:
        index_combinatorial (array): Indices of atoms to be removed.
        n_items_to_select (int): Number of items to select.
        optimisation (str): Optimisation method ('random' or 'exhaustive search').
        maxiter (int): Maximum number of iterations.

    Returns:
        list: List of sites to remove based on the optimisation strategy.
    """
    
    if optimisation=='random':
        
        sites_random = []
        for _ in range(maxiter):
            sites_random.append(np.random.choice(index_combinatorial,n_items_to_select,replace=False))
    
        return sites_random
    
    elif optimisation == 'exhaustive search':
        
        from itertools import combinations
        
        sites_es = np.array(list(combinations(index_combinatorial, n_items_to_select)))
        
        return sites_es


def objective_function(undercoordinated_atoms,sites_to_remove,nano_initial,strategy='com',charges={}):
    
    """
    Computes the quality index for a given configuration of sites to remove.

    Args:
        undercoordinated_atoms (array): List of undercoordinated atoms.
        sites_to_remove (array): List of sites to be removed.
        nano_initial (pymatgen.Structure): Initial nanoparticle structure.
        strategy (str, optional): Optimisation strategy ('com', 'dipole', 'max_distance'). Defaults to 'com'.
        charges (dict, optional): Atomic charges. Defaults to {}.

    Returns:
        float: Quality index of the configuration.
    """
    sites = np.append(undercoordinated_atoms,sites_to_remove)
    sites_to_keep = np.delete(np.arange(nano_initial.num_sites), sites)
    cart_coords = nano_initial.cart_coords[sites_to_keep]
    
    if strategy=='com':
        
        quality_index = np.sum(np.mean(cart_coords,axis=0)**2)
        
        return quality_index
    
    elif strategy=='dipole':

        atom_types = np.array(nano_initial.atomic_numbers)[sites_to_keep]
        charge_vector = [charges.get(atom_type, 0) for atom_type in atom_types]

        quality_index = np.sum(np.sum(np.array(charge_vector)[:, np.newaxis] * 
                                      cart_coords, axis=0)**2)
        return quality_index
        #print(quality_index)
        
    elif strategy=='max_distance':

        # Initialize total distance
        total_distance = 0.0

        # Calculate the total distance
        for i in range(1, len(cart_coords)):
            distance = np.linalg.norm(cart_coords[i] - cart_coords[i - 1])  # Euclidean distance between consecutive points
            total_distance += distance

        # If the path should loop back to the first point, add the distance from the last to the first point
        if len(cart_coords) > 2:
            distance = np.linalg.norm(cart_coords[0] - cart_coords[-1])
            total_distance += distance

        quality_index = -total_distance
        
        return quality_index
    
    elif strategy=='global_coordination':
        pass


def get_coulomb_matrix(nano):
    """
    Computes the Coulomb matrix of a given nanoparticle.

    Args:
        nano (pymatgen.Structure): Nanoparticle structure.

    Returns:
        np.ndarray: Coulomb matrix of the nanoparticle.
    """

    atomic_numbers = np.array(nano.atomic_numbers)
    atomic_numbers_outer = np.outer(atomic_numbers,atomic_numbers)

    distance_matrix = nano.distance_matrix
    np.fill_diagonal(distance_matrix,1)
    distance_matrix_inv = 1/nano.distance_matrix
    np.fill_diagonal(distance_matrix_inv,0)

    cm = atomic_numbers_outer*distance_matrix_inv
    
    return np.round(cm,5)
############### THE END? ##############################
def remove_undercoordinated_atoms(nano,species,min_coord):
    """
    Function that removes the unbounded atoms
    Args:
        symbol(Atoms): crystal structure
        atoms(Atoms): Bulk-cut like nanoparticle
    Return:
        atoms(Atoms): nanoparticle without unbounded atoms
    """
    sites = np.where(np.array(nano.atomic_numbers) == species)
    
    coord_vector = get_coordination(nano)

    undercoordinated_atoms = np.where(coord_vector[sites] < min_coord)[0]
    
    nano.remove_sites(undercoordinated_atoms)

    return nano


def get_coordination(nano):
    """
    Function that calculates the coordination of the sites in the nanoparticle
    using the distance matrix.

    Args:
        nano (pymatgen.core.structure.Structure): The pymatgen Structure object.
    Returns:
        numpy.ndarray: A numpy array containing the coordination number for each site.
    """

    # Position of atoms types in the structure
    atoms_type_site = []
    for atom in np.unique(nano.atomic_numbers):
        atoms_type_site.append(np.where(nano.atomic_numbers == atom)[0])
    atomic_numbers = nano.atomic_numbers

    distance_matrix_sorted = np.round(np.sort(nano.distance_matrix, axis=1), 5)

    first_shell = []  # Defined as a jump of 0.5 Angstrom
    second_shell = []
    for atom in range(nano.num_sites):
        first_shell_index = np.where(distance_matrix_sorted[atom, 1:] - distance_matrix_sorted[atom, :-1] > 0.5)[0][1]
        first_shell.append(distance_matrix_sorted[atom, first_shell_index])

    # second_shell_index = np.where(distance_matrix_sorted[atom, 1:] - distance_matrix_sorted[atom, :-1] > 0.5)[0]
    # second_shell.append(distance_matrix_sorted[atom, second_shell_index])

    first_shell_index = np.array(first_shell_index)

    coord_vector = np.zeros(nano.num_sites, dtype=int)

    for i in range(nano.num_sites):
        coord_vector[i] = np.sum(distance_matrix_sorted[i, 1:10] <= first_shell[i])

    return coord_vector


def is_stoichiometric(structure,initial_structure):

    excess_atoms = get_excess_atoms(structure,initial_structure)
    
    if np.all(excess_atoms == 0) == True:
        return True
    else:
        return False


def get_excess_atoms(nano, initial_structure):
    """
    Calculate the number of excess atoms to be removed from a nanoparticle structure compared to an initial bulk structure.

    Args:
        nano (pymatgen.core.structure.Structure): The nanoparticle structure.
        initial_structure (pymatgen.core.structure.Structure): The initial bulk structure.

    Returns:
        numpy.ndarray: An array containing the number of atoms to be removed.
    """
    
    import numpy as np
    from math import gcd

    # Get the atomic numbers of the nanoparticle structure
    atomic_numbers_np = np.array(nano.atomic_numbers)

    # Get the unique atom types present in the initial structure
    unique_atom_types = np.unique(initial_structure.atomic_numbers)

    # Calculate the composition of the nanoparticle structure and initial structure
    nano_composition = []
    initial_composition = []

    for atom_type in unique_atom_types:
        # Count the number of atoms of each type in the nanoparticle structure
        nano_count = len(np.where(atomic_numbers_np == atom_type)[0])
        nano_composition.append(nano_count)

        # Count the number of atoms of each type in the initial structure
        initial_count = len(np.where(initial_structure.atomic_numbers == atom_type)[0])
        initial_composition.append(initial_count)

    nano_composition = np.array(nano_composition)
    initial_composition = np.array(initial_composition)

    # Round the values to the nearest integer
    initial_composition = np.round(initial_composition)

    # Divide all elements by their greatest common divisor (GCD)
    initial_composition_gcd = int(np.abs(gcd(*initial_composition)))
    initial_composition //= initial_composition_gcd

    t = np.argmin((nano_composition / np.amin(nano_composition)) / initial_composition)
    ideal_cl = nano_composition[t] // initial_composition[t] * initial_composition

    excess = nano_composition - ideal_cl
    ideal_cl = nano_composition[t] // initial_composition[t] * initial_composition
    
    return nano_composition - ideal_cl


def remove_excess_atoms(np0,nano_remove_array):
   
    # Read the 
    from pymatgen.core.periodic_table import Element

    atom_symbols = list(np0.composition.reduced_composition.as_dict().keys())
    atomic_number = np.array([Element(atom_symbol).number for atom_symbol in atom_symbols])

    atoms_type_site = []
    for atom in atomic_number:
        atoms_type_site.append(np.where(np0.atomic_numbers == atom)[0])    
        coord_vector = get_coordination(np0)

    for i,atoms in enumerate(nano_remove_array):
        if atoms > 0:     
            coord_sorted = np.sort(coord_vector[atoms_type_site[i]])
            unique_coord, count_unique = np.unique(coord_sorted, return_counts=True)
            # This is the beginning of the combinatorial analysis
            '''if count_unique[0] < atoms:
                index = atoms
                for j in count_unique:
                    if j < index:
                        atoms_to_remove = np.argsort(coord_vector[atoms_type_site[i]])[0:count_unique]
            elif count_unique[0] > atoms:'''
                # remove the top ones in the array
            atoms_to_remove = np.argsort(coord_vector[atoms_type_site[i]])[0:atoms]
                # do all the combinations
                #import itertools
                #X = np.array(list(itertools.product([0, 1], repeat=18)))
                #np.where(np.sum(X,axis=1) == 2)
                # This easily becomes too large to explore
                
                # add symmetry analysis
    np0.remove_sites(atoms_to_remove)

    return np0


def get_less_abundant_atom(initial_structure):
    
    from pymatgen.core.periodic_table import Element

    composition = initial_structure.composition.get_el_amt_dict()
    less_abundand_element = min(composition, key=lambda k: composition[k])

    atomic_number = Element(less_abundand_element).number
    less_abundand_element = atomic_number

    return atomic_number

def get_more_abundant_atom(initial_structure):
    
    from pymatgen.core.periodic_table import Element

    composition = initial_structure.composition.get_el_amt_dict()
    less_abundand_element = max(composition, key=lambda k: composition[k])

    atomic_number = Element(less_abundand_element).number
    less_abundand_element = atomic_number

    return atomic_number

def get_surface_atoms(np0,initial_structure):
    
    nano = copy.deepcopy(np0)
    atomic_numbers = np.array(nano.atomic_numbers)

    #less abundant atoms

    less_abundand_element = get_less_abundant_atom(initial_structure)
    less_abundand_element_index = np.where(np.array(nano.atomic_numbers) == less_abundand_element)[0]
    max_coord = np.amax(get_coordination(nano)[less_abundand_element_index])
    surface_less_abundant = np.where(get_coordination(nano)[less_abundand_element_index] < max_coord)[0]
    less_abundand_element_surface_1 = less_abundand_element_index[surface_less_abundant]


    #more aboundant atoms
    more_abundand_element = get_more_abundant_atom(initial_structure)
    more_abundand_element_index = np.where(np.array(nano.atomic_numbers) == more_abundand_element)[0]
    max_coord = np.amax(get_coordination(nano)[more_abundand_element_index])
    surface_more_abundant = np.where(get_coordination(nano)[more_abundand_element_index] < max_coord)[0]
    more_abundand_element_surface = more_abundand_element_index[surface_more_abundant]


    #less abundant atoms linked to surface more aboundant atoms
    filter_1 = atomic_numbers[np.argsort(nano.distance_matrix[more_abundand_element_surface])] == less_abundand_element
    filter_2 = np.sort(nano.distance_matrix[more_abundand_element_surface]) < 2
    distance_matrix_filter = np.argsort(nano.distance_matrix[more_abundand_element_surface]) * filter_1 * filter_2

    less_abundand_element_surface_2 = np.unique(distance_matrix_filter.flatten())[1:]

    surface_atoms = np.concatenate([less_abundand_element_surface_1,more_abundand_element_surface,less_abundand_element_surface_2])

    surface_atoms = np.unique(surface_atoms)
    
    return surface_atoms

def build_np0_old(structure,surfaces,energies,size,rounding='closest'):
# This function returns the non stoichiometric np0 with size closest to the requested one

    from pymatgen.analysis.wulff import WulffShape
    from scipy.spatial import Delaunay, ConvexHull

    wulffshape = WulffShape(structure.lattice, surfaces, energies)

    atom_density = structure.num_sites/structure.volume
    wulffshape.volume

    ###Add rounding

    if type(size) == int:
        new_volume = size/atom_density
        ratio = new_volume**(1/3)/wulffshape.volume**(1/3)

        if debug == True:
            shape = ConvexHull((wulffshape.wulff_convex.points/(wulffshape.volume**(1/3)))*(new_volume**(1/3)))        
            print('NP0 size = ',shape.volume*atom_density)

        shape = Delaunay(wulffshape.wulff_convex.points*(ratio))
        sites_to_remove = []
        
        # Generate the supercell
        supercell_matrix = np.identity(3)*10 ###Make this better
        structure.make_supercell(supercell_matrix)
        
        # Center around the center of mass
        center_of_mass = np.average(structure.cart_coords,axis=0)
        structure.translate_sites(np.arange(structure.num_sites),
                                  -center_of_mass,frac_coords=False,
                                  to_unit_cell= False)
        
        # Generate the list of atoms to be removed
        for i,coord in enumerate(structure.cart_coords):
            if (shape.find_simplex(coord) >= 0) == False:
                sites_to_remove.append(i)
        
        structure.remove_sites(sites_to_remove)

        # Translate the sites back to their orginal position
        structure.translate_sites(np.arange(structure.num_sites),
                                  center_of_mass,frac_coords=False,
                                  to_unit_cell= False)

        np0 = structure
        
        return np0

############################################################################
    

    """
    Function that return the nature termination in a face
    
    Args:
        symbol(Atoms):crystal strucuture
        atoms(Atoms):nanoparticle
        surfaceIndexez([]): miller indexes 
    Return:
        termFaceNatur(str): metalRich,nonMetalRich,stoich
    """
    # print(surfaceIndexes)
    # Get the cell stoichiometry
    listOfChemicalSymbols=symbol.get_chemical_symbols()
    
    #Count and divide by the greatest common divisor
    chemicalElements=list(set(listOfChemicalSymbols))

    # put the stuff in order, always metals first
    if chemicalElements[0] in nonMetals:
        chemicalElements.reverse()

    counterCell=[]
    for e in chemicalElements:
        counterCell.append(listOfChemicalSymbols.count(e))

    gcd=np.gcd.reduce(counterCell)

    cellStoichiometry=counterCell/gcd
    crystalRatio=cellStoichiometry[0]/cellStoichiometry[1]

    terminationElements=terminations(symbol,atoms,[surfaceIndexes])
    if terminationElements=='non equivalents':
        return None
        
    # get the stoichiometry 
    orientationProp=[]
    for e in chemicalElements:
        orientationProp.append(terminationElements.count(e))
    # print(orientationProp)
    # if orientation prop just have one index 
    if len(list(set(terminationElements)))==1:
        if list(set(terminationElements))[0] in nonMetals:
            termFaceNatur='nonMetalRich'
        else:
            termFaceNatur='metalRich'
    #if orientation has more than one element
    else:
        gcd=np.gcd.reduce(orientationProp)
        orientationStoichiometry=orientationProp/gcd
        orientRatio=orientationStoichiometry[0]/orientationStoichiometry[1]
        # print(chemicalElements) 
        # print(orientationStoichiometry)
        # print(orientRatio,crystalRatio)
        # # # exit(1)
        # print(chemicalElements[0])
        # exit(1)
        if orientRatio==crystalRatio:
            termFaceNatur='stoich'
        
        elif orientRatio>crystalRatio:
            if chemicalElements[0] in nonMetals:
                termFaceNatur='nonMetalRich'
            else:
                termFaceNatur='metalRich'
        elif orientRatio<crystalRatio:
            if chemicalElements[1] in nonMetals:
                termFaceNatur='nonMetalRich'
            else:
                termFaceNatur='metalRich'
    # print(termFaceNatur)
    return termFaceNatur