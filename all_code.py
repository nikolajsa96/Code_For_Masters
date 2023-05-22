from ase.visualize import view
import ase
from ase import Atoms
from ase.cluster.cubic import FaceCenteredCubic
import random as ran
import numpy as np
from numpy import *
from ase.io import *
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.units import kB
import pylab as plt
from ase.eos import EquationOfState
from ase.build.attach import attach
import heapq
import os
from itertools import combinations

#code to keep only "H" atoms
def chell_builder(atoms):
    chell = atoms.copy()
    new_len = len(chell)
    for k in range(3):
        for i in range(len(chell)-1,0-1,-1):
            symbol = chell.get_chemical_symbols()[i]
            if symbol != "H":
                chell.pop(i)
                new_len = len(chell)
    return chell

#code to remove "H" atoms
def inner_layer_builder(atoms):
    inner = atoms.copy()
    new_len = len(inner)
    for k in range(3):
        for i in range(len(inner)-1,0-1,-1):
            symbol = inner.get_chemical_symbols()[i]
            if symbol == "H":
                inner.pop(i)
                new_len = len(inner)
    return inner


#code to make Str. 1
def sim_ran(surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
                layers = [7, 8, 7],
                symbols_list = ["Cu","Al","Au","Pt"],
                filled = 300):
    atoms = FaceCenteredCubic("Pt", surfaces, layers)
    change_list = []
    size = len(list(atoms.symbols))
    inx = list(range(0, size))
    chell = chell_builder(atoms)
    for i in range(size):
        atoms.symbols[i] = "H"
    for i in range(len(symbols_list)):
        g = ran.sample(inx,int(filled/len(symbols_list)))
        for j in g:
            inx.remove(j)
        change_list.append(g)
    for i in range(len(symbols_list)):
        for j in change_list[i]:
            atoms.symbols[j] = symbols_list[i]
    inner = inner_layer_builder(atoms)
    count_fix = (filled/len(symbols_list)-int(filled/len(symbols_list)))*len(symbols_list)
    if count_fix >= 1:
        chell = chell_builder(atoms)
        inx = list(range(0,len(chell)))
        g = ran.sample(inx,int(count_fix))
        for j,i in zip(g, range(1,len(g)+1)):
            chell.symbols[j] = symbols_list[-i]
    atoms = chell.copy()
    atoms += inner
    return atoms


#code to make Str.2 and the start of Str.3
def sim_t(surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
            layers = [4, 5, 4],
            H_surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
            H_layers = [6, 7, 6], symbols_list = ["Cu","Al","Au","Pt"]):
    atoms = FaceCenteredCubic("Pt", surfaces, layers)
    filled = len(atoms)
    change_list = []
    size = len(list(atoms.symbols))
    inx = list(range(0, size))
    chell = chell_builder(atoms)
    for i in range(size):
        atoms.symbols[i] = "H"
    for i in range(len(symbols_list)):
        g = ran.sample(inx,int(filled/len(symbols_list)))
        for j in g:
            inx.remove(j)
        change_list.append(g)
    for i in range(len(symbols_list)):
        for j in change_list[i]:
            atoms.symbols[j] = symbols_list[i]
    inner = inner_layer_builder(atoms)
    count_fix = (filled/len(symbols_list)-int(filled/len(symbols_list)))*len(symbols_list)
    if count_fix >= 1:
        chell = chell_builder(atoms)
        inx = list(range(0,len(chell)))
        g = ran.sample(inx,int(count_fix))
        for j,i in zip(g, range(1,len(g)+1)):
            chell.symbols[j] = symbols_list[-i]
    atoms = chell.copy()
    atoms += inner
    H_atoms = FaceCenteredCubic("Pt", H_surfaces, H_layers)
    H_size = len(H_atoms)
    for i in range(H_size):
        H_atoms.symbols[i] = "H"
    atoms += H_atoms
    size = len(atoms)
    for i in range(size-1,0-1,-1):
        size = len(atoms)
        r = list(range(0,size))
        r.pop(i)
        dist = atoms.get_distances(i,r)
        symbol = atoms.get_chemical_symbols()[i]
        if dist.min() < 1 and symbol == "H":
            atoms.pop(i)
    return atoms

#code to swap n non "H" atoms with "H" atom, used to go from Str. 2 to Str. 3
def ran_move(atoms, n_move = 5):
    atoms = atoms.copy()
    H_list = []
    not_H_list = []
    atoms_list = list(range(0,len(atoms)))
    for p in atoms_list:
        symbol = atoms.get_chemical_symbols()[p]
        if symbol == "H":
            H_list.append(p)
        else:
            not_H_list.append(p)

    H_move = ran.sample(H_list, n_move)
    Not_h_move = ran.sample(not_H_list, n_move)

    for i,j in zip(H_move, Not_h_move):
        symbol_0 = atoms.get_chemical_symbols()[i]
        symbol_1 = atoms.get_chemical_symbols()[j]
        atoms.symbols[i] = symbol_1
        atoms.symbols[j] = symbol_0

    return atoms



#code for the neighbor Free Space Move method
def nabo_tom(atoms, T, steps = 1000, cutoff_nabo = 1.05):
    atoms = atoms.copy()
    count_temp = 0
    im = []
    inner_im= []
    pot_save = []
    tom_H = atoms
    chell = chell_builder(tom_H)
    inner = inner_layer_builder(tom_H)
    inner.set_calculator(EMT())
    int_pot = inner.get_potential_energy()
    pot_save.append(int_pot)
    while len(pot_save) < steps + 1:
        inner.set_calculator(EMT())
        start_pot = inner.get_potential_energy()
        inner_save = inner.copy()
        chell_save = chell.copy()
        atoms2 = chell.copy()
        atoms2 += inner
        im.append(atoms2.copy())
        free_spot_H = []
        free_spot_not_H = []
        atoms_list = list(range(len(atoms2)))
        random_atom = ran.randrange(0,len(atoms2))
        dist_list = atoms2.get_distances(random_atom,atoms_list)
        s = heapq.nsmallest(2, dist_list)[-1]
        neighbors = [j for j in zip(dist_list, atoms_list) if 0!= j[0] < s*cutoff_nabo]
        not_used, new_neighbor = map(list,zip(*neighbors))
        for p in new_neighbor:
            symbol = atoms2.get_chemical_symbols()[p]
            if symbol == "H":
                free_spot_H.append(p)
            else:
                free_spot_not_H.append(p)
        if atoms2.get_chemical_symbols()[random_atom] != "H":
            free_spot = free_spot_H
        else:
            free_spot = free_spot_not_H
        if len(free_spot) > 0:
            random_int = ran.sample(free_spot,1)
            symbol_0 = atoms2.get_chemical_symbols()[random_int[0]]
            symbol_1 = atoms2.get_chemical_symbols()[random_atom]
            atoms2.symbols[random_int] = symbol_1
            atoms2.symbols[random_atom] = symbol_0

            chell = chell_builder(atoms2)
            inner = inner_layer_builder(atoms2)
            inner.set_calculator(EMT())
            slut_pot = inner.get_potential_energy()
            if np.exp(-((slut_pot-start_pot))/(kB*T[count_temp])) < np.random.rand():
                inner  = inner_save
                chell = chell_save
            count_temp = count_temp+ 1
            inner.set_calculator(EMT())
            end_pot = inner.get_potential_energy()
            pot_save.append(end_pot)
            inner_im.append(inner.copy())
    return pot_save, im, inner_im

#code for the global Free Space Move method
def global_tom(atoms, T, steps = 1000):
    atoms = atoms.copy()
    count_temp = 0
    im = []
    inner_im= []
    pot_save = []
    tom_H = atoms
    chell = chell_builder(tom_H)
    inner = inner_layer_builder(tom_H)
    inner.set_calculator(EMT())
    int_pot = inner.get_potential_energy()
    pot_save.append(int_pot)
    while len(pot_save) < steps + 1:
        inner.set_calculator(EMT())
        start_pot = inner.get_potential_energy()
        inner_save = inner.copy()
        chell_save = chell.copy()
        atoms2 = chell.copy()
        atoms2 += inner
        im.append(atoms2.copy())
        free_spot = []
        atoms_list = list(range(len(atoms2)))
        random_int = ran.sample(range(0,len(atoms)),2)
        if (atoms2.get_chemical_symbols()[random_int[0]] != "H" and atoms2.get_chemical_symbols()[random_int[1]] == "H") or (atoms2.get_chemical_symbols()[random_int[1]] != "H" and atoms2.get_chemical_symbols()[random_int[0]] == "H"):
            symbol_0 = atoms2.get_chemical_symbols()[random_int[0]]
            symbol_1 = atoms2.get_chemical_symbols()[random_int[1]]
            atoms2.symbols[random_int[0]] = symbol_1
            atoms2.symbols[random_int[1]] = symbol_0
            chell = chell_builder(atoms2)
            inner = inner_layer_builder(atoms2)
            inner.set_calculator(EMT())
            slut_pot = inner.get_potential_energy()
            if np.exp(-((slut_pot-start_pot))/(kB*T[count_temp])) < np.random.rand():
                inner  = inner_save
                chell = chell_save
            count_temp = conut_temp + 1
            inner.set_calculator(EMT())
            end_pot = inner.get_potential_energy()
            pot_save.append(end_pot)
            inner_im.append(inner.copy())
    return pot_save, im, inner_im

#code for the global All Move method
def global_ran(atoms, T, steps = 1000):
    atoms = atoms.copy()
    count_temp = 0
    im = []
    inner_im= []
    pot_save = []
    tom_H = atoms
    chell = chell_builder(tom_H)
    inner = inner_layer_builder(tom_H)
    inner.set_calculator(EMT())
    int_pot = inner.get_potential_energy()
    pot_save.append(int_pot)
    while len(pot_save) < steps + 1:
        inner.set_calculator(EMT())
        start_pot = inner.get_potential_energy()
        inner_save = inner.copy()
        chell_save = chell.copy()
        atoms2 = chell.copy()
        atoms2 += inner
        im.append(atoms2.copy())
        free_spot = []
        atoms_list = list(range(len(atoms2)))
        random_int = ran.sample(range(0,len(atoms)),2)
        symbol_0 = atoms2.get_chemical_symbols()[random_int[0]]
        symbol_1 = atoms2.get_chemical_symbols()[random_int[1]]
        if symbol_0 != "H" or symbol_1 != "H":
            atoms2.symbols[random_int[0]] = symbol_1
            atoms2.symbols[random_int[1]] = symbol_0
            chell = chell_builder(atoms2)
            inner = inner_layer_builder(atoms2)
            inner.set_calculator(EMT())
            slut_pot = inner.get_potential_energy()
            if np.exp(-((slut_pot-start_pot))/(kB*T[count_temp])) < np.random.rand():
                inner  = inner_save
                chell = chell_save
            count_temp = count_temp + 1
            inner.set_calculator(EMT())
            end_pot = inner.get_potential_energy()
            pot_save.append(end_pot)
            inner_im.append(inner.copy())
    return pot_save, im, inner_im

#code for the neighbor All Move method
def nabo_ran(atoms, T, steps = 1000, cutoff_nabo = 1.05):
    atoms = atoms.copy()
    count_temp = 0
    im = []
    inner_im= []
    pot_save = []
    tom_H = atoms
    chell = chell_builder(tom_H)
    inner = inner_layer_builder(tom_H)
    inner.set_calculator(EMT())
    int_pot = inner.get_potential_energy()
    pot_save.append(int_pot)
    while len(pot_save) < steps + 1:
        inner.set_calculator(EMT())
        start_pot = inner.get_potential_energy()
        inner_save = inner.copy()
        chell_save = chell.copy()
        atoms2 = chell.copy()
        atoms2 += inner
        im.append(atoms2.copy())
        free_spot = []
        atoms_list = list(range(len(atoms2)))
        random_atom = ran.randrange(0,len(atoms2))
        dist_list = atoms2.get_distances(random_atom,atoms_list)
        s = heapq.nsmallest(2, dist_list)[-1]
        neighbors = [j for j in zip(dist_list, atoms_list) if 0!= j[0] < s*cutoff_nabo]
        not_used, new_neighbor = map(list,zip(*neighbors))
        random_int = ran.sample(new_neighbor,1)
        symbol_0 = atoms2.get_chemical_symbols()[random_int[0]]
        symbol_1 = atoms2.get_chemical_symbols()[random_atom]
        if symbol_0 != "H" or symbol_1 != "H":
            atoms2.symbols[random_int] = symbol_1
            atoms2.symbols[random_atom] = symbol_0
            chell = chell_builder(atoms2)
            inner = inner_layer_builder(atoms2)
            inner.set_calculator(EMT())
            slut_pot = inner.get_potential_energy()
            if np.exp(-((slut_pot-start_pot))/(kB*T[count_temp])) < np.random.rand():
                inner  = inner_save
                chell = chell_save
            count_temp = count_temp + 1
            inner.set_calculator(EMT())
            end_pot = inner.get_potential_energy()
            pot_save.append(end_pot)
            inner_im.append(inner.copy())
    return pot_save, im, inner_im

#code used to make a temperature range, which did not just change at a chosen step
def temp_len(start=1000,slut=200,lang=8000,end=200,end_len=2000):
    len_t = list(range(start,slut,-1))
    len_t = np.repeat(len_t,lang/len(len_t)).tolist()
    print(len(len_t),len_t[-1])
    len_t.extend([end]*end_len)
    return len_t

#code used for the neighbor Free Shuffle method
def nabo_fs(atoms, T, steps = 1000, cutoff_nabo = 1.01):
    atoms = atoms.copy()
    count_temp = 0
    im = []
    inner_im= []
    pot_save = []
    chell = chell_builder(atoms)
    inner = inner_layer_builder(atoms)
    inner.set_calculator(EMT())
    int_pot = inner.get_potential_energy()
    pot_save.append(int_pot)
    while len(pot_save) < steps+1:
        inner.set_calculator(EMT())
        start_pot = inner.get_potential_energy()
        inner_save = inner.copy()
        chell_save = chell.copy()
        atoms2 = chell.copy()
        atoms2 += inner
        im.append(atoms2.copy())
        free_spot_H = []
        free_spot_not_H = []
        allowed_list = []
        atoms_list = list(range(len(atoms2)))
        random_atom = ran.randrange(0,len(atoms2))
        dist_list = atoms2.get_distances(random_atom,atoms_list)
        s = heapq.nsmallest(2, dist_list)[-1]
        neighbors = [j for j in zip(dist_list, atoms_list) if 0!= j[0] <= s*cutoff_nabo]
        not_used, new_neighbor = map(list,zip(*neighbors))
        for p in new_neighbor:
            symbol = atoms2.get_chemical_symbols()[p]
            if symbol == "H":
                free_spot_H.append(p)
            else:
                free_spot_not_H.append(p)
        if atoms2.get_chemical_symbols()[random_atom] != "H":
            if len(free_spot_H) == 0:
                continue
            if len(free_spot_not_H) == 0:
                #continue
                allowed_list.extend(free_spot_H) #add # to the shape from changing and remove from continue
            if len(free_spot_H) >= 1 and len(free_spot_not_H) >= 1:
                allowed_list.extend(free_spot_H) #add # to stop the shape from changing
                for n in free_spot_not_H:
                    dist_list_nabo = atoms2.get_distances(n, atoms_list)
                    neighbors_nabo = [j for j in zip(dist_list_nabo, atoms_list) if 0 != j[0] <= s*cutoff_nabo]
                    not_used, new_neighbor_nabo = map(list, zip(*neighbors_nabo))
                    for p in new_neighbor_nabo:
                        symbol = atoms2.get_chemical_symbols()[p]
                        if symbol != "H" and p in free_spot_not_H:
                            allowed_list.append([n,p])
        if len(allowed_list) > 0:
            random_int = ran.sample(allowed_list,1)[0]
            if type(random_int) is int:
                symbol_0 = atoms2.get_chemical_symbols()[random_int]
                symbol_1 = atoms2.get_chemical_symbols()[random_atom]
                atoms2.symbols[random_int] = symbol_1
                atoms2.symbols[random_atom] = symbol_0
            else:
                symbol_0 = atoms2.get_chemical_symbols()[random_int[0]]
                symbol_1 = atoms2.get_chemical_symbols()[random_int[1]]
                atoms2.symbols[random_int][0] = symbol_1
                atoms2.symbols[random_int][1] = symbol_0
            chell = chell_builder(atoms2)
            inner = inner_layer_builder(atoms2)
            inner.set_calculator(EMT())
            slut_pot = inner.get_potential_energy()
            if np.exp(-((slut_pot-start_pot))/(kB*T[count_temp])) < np.random.rand():
                inner  = inner_save
                chell = chell_save
            count_temp = count_temp + 1
            inner.set_calculator(EMT())
            end_pot = inner.get_potential_energy()
            pot_save.append(end_pot)
            inner_im.append(inner.copy())
    return pot_save, im, inner_im

#code for the modified neighbor Free Shuffle (only moves atoms on the facets)  
def nabo_fs_mod(atoms, T, steps = 1000, cutoff_nabo = 1.01):
    atoms = atoms.copy()
    count_temp = 0
    im = []
    inner_im= []
    pot_save = []
    chell = chell_builder(atoms)
    inner = inner_layer_builder(atoms)
    inner.set_calculator(EMT())
    int_pot = inner.get_potential_energy()
    pot_save.append(int_pot)
    may_move = []
    atoms_list = list(range(len(atoms)))
    for i in atoms_list:
        may_move_coord = []
        if atoms.get_chemical_symbols()[i] != "H":
            dist_list = atoms.get_distances(i, atoms_list)
            s = heapq.nsmallest(2, dist_list)[-1]
            neighbors = [j for j in zip(dist_list, atoms_list) if 0 != j[0] < s*cutoff_nabo]
            not_used, new_neighbors = map(list, zip(*neighbors))
            for j in new_neighbors:
                if atoms.get_chemical_symbols()[j] != "H":
                    may_move_coord.append(j)
            if len(may_move_coord) == 9 or len(may_move_coord) == 8:
                may_move.append(i)
    while len(pot_save) < steps+1:
        inner.set_calculator(EMT())
        start_pot = inner.get_potential_energy()
        inner_save = inner.copy()
        chell_save = chell.copy()
        atoms2 = inner.copy()
        atoms2 += chell
        im.append(atoms2.copy())
        free_spot_H = []
        free_spot_not_H = []
        allowed_list = []
        random_atom = ran.choice(may_move)
        nabo_symbol = []
        self_symbol = atoms2.get_chemical_symbols()[random_atom]
        dist_list = atoms.get_distances(random_atom, atoms_list)
        s = heapq.nsmallest(2, dist_list)[-1]
        neighbors = [j for j in zip(dist_list, atoms_list) if 0!= j[0] <= s*cutoff_nabo]
        not_used, new_neighbor = map(list,zip(*neighbors))
        for p in new_neighbor:
            symbol = atoms2.get_chemical_symbols()[p]
            if symbol == "H":
                free_spot_H.append(p)
            else:
                free_spot_not_H.append(p)
        if self_symbol != "H":
            if len(free_spot_H) == 0:
                continue
            if len(free_spot_not_H) == 0:
                continue
                #allowed_list.extend(free_spot_H)
            if len(free_spot_H) >= 1 and len(free_spot_not_H) >= 1:
                #allowed_list.extend(free_spot_H)
                for n in free_spot_not_H:
                    dist_list = atoms2.get_distances(n, atoms_list)
                    neighbors = [j for j in zip(dist_list, atoms_list) if 0 != j[0] <= s*cutoff_nabo]
                    not_used, new_neighbor = map(list,zip(*neighbors))
                    nabo_nabo_H = []
                    coord_test = []
                    for p in new_neighbor:
                        symbol = atoms2.get_chemical_symbols()[p]
                        if symbol == "H":
                            nabo_nabo_H.append(p)
                        else:
                            coord_test.append(p)
                    if any(x in nabo_nabo_H for x in free_spot_H) and len(nabo_nabo_H)>0:
                        if len(coord_test) == 9 or len(coord_test) == 8:
                            allowed_list.extend([n]*2)
        if len(allowed_list) > 0:
            random_int = ran.sample(allowed_list,1)[0]
            if type(random_int) is int:
                symbol_0 = atoms2.get_chemical_symbols()[random_int]
                symbol_1 = atoms2.get_chemical_symbols()[random_atom]
                atoms2.symbols[random_int] = symbol_1
                atoms2.symbols[random_atom] = symbol_0
            else:
                symbol_0 = atoms2.get_chemical_symbols()[random_int[0]]
                symbol_1 = atoms2.get_chemical_symbols()[random_int[1]]
                atoms2.symbols[random_int][0] = symbol_1
                atoms2.symbols[random_int][1] = symbol_0
            chell = chell_builder(atoms2)
            inner = inner_layer_builder(atoms2)
            inner.set_calculator(EMT())
            slut_pot = inner.get_potential_energy()
            if np.exp(-((slut_pot-start_pot))/(kB*T[count_temp])) < np.random.rand():
                inner = inner_save
                chell = chell_save
            count_temp = count_temp + 1
            inner.set_calculator(EMT())
            end_pot = inner.get_potential_energy()
            pot_save.append(end_pot)
            inner_im.append(inner.copy())
        else:
            print("No allowed moves")
    return pot_save, im, inner_im

#code to find coordination
def cond(data, cutoff_nabo = 1.05):
        cond = []
        atoms_total_len = list(range(len(data)))
        uni_sym, total_count = np.unique(data.get_chemical_symbols(), return_counts = True)
        dist_all = data.get_distances(1, atoms_total_len)
        dist_min = heapq.nsmallest(2, dist_all)[-1]
        for i in range(len(data)):
                nabo_symbol = []
                atoms_list = list(range(len(data)))
                self_symbol = data.get_chemical_symbols()[i]
                if self_symbol != "H":
                        dist_list = data.get_distances(i, atoms_list)
                        neighbors = [j for j in zip(dist_list, atoms_list) if 0 != j[0] < dist_min*cutoff_nabo]
                        not_used, new_neighbors = map(list, zip(*neighbors))
                        for j in new_neighbors:
                                if data.get_chemical_symbols()[j] != "H":
                                        nabo_symbol.append(j)
                        cond.append([self_symbol, len(nabo_symbol)])
        con_sym, con_count = np.unique(cond, return_counts = True, axis = 0)
        procent_list = []
        for idx, i in enumerate(con_sym):
                procent = con_count[idx]/total_count[int(np.where(uni_sym == i[0])[0])]
                procent_list.append(round(procent,2))
        cond_data = np.hstack((con_sym, con_count[:,None]))
        #cond_data = np.hstack((cond_data, np.array(procent_list)[:,None]))
        return cond_data


