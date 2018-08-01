'''
Author: Pranav Khade(pranavk@iastate.edu)
'''
import argparse
import numpy

from Bio.PDB import PDBList
from Bio.PDB import PDBParser
from Bio.PDB import Atom

##########----------##########----------##########----------##########----------##########----------
'''I/O'''
def IO():
    '''
    INFO: Argument parser to the program for now.
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument('filename',metavar='PDB')
    args=parser.parse_args()
    return args

##########----------##########----------##########----------##########----------##########----------
'''Selecting atoms of interests'''
def SelectCAlpha(molecule):
    '''
    INFO: Select the C-alpha atoms from the amino acid chain.
    '''
    ca_atoms=[atom for atom in molecule.get_atoms() if atom.name=='CA' and atom.get_parent().id[0]==' ']
    return ca_atoms


def SelectTipofAA(molecule):
    '''
    INFO: Returns the tips of the atoms (In case of ALA nad GLY, returns the C-Alpha atom)
    '''
    calpha=SelectCAlpha(molecule)
    TipsofAA=[]
    for CA in calpha:
        resname=CA.get_parent().get_resname()
        max_distance=0
        tip=CA
        if(resname=='ALA' or resname=='GLY'):
            TipsofAA.append(tip)
        else:
            for i in  CA.get_parent().get_atoms():
                temp_dist=numpy.linalg.norm(numpy.array(i.coord)-numpy.array(CA.coord))
                if(temp_dist>max_distance):
                    max_distance=temp_dist
                    tip=i
            TipsofAA.append(tip)
    return TipsofAA


def SelectAACenterofMass(molecule):
    '''
    INFO: This returns a center of mass of each amino acid in the protein.
    '''
    calpha=SelectCAlpha(molecule)
    residues=[i.get_parent() for i in calpha]
    AtomsofCenterofMass=[]
    for residue in residues:
        XYZ_M=[0,0,0]
        MassofAA=0
        SumBfactor=0
        TotalAtoms=0
        for AtomsNO,atom in enumerate(residue.get_atoms()):
            SumBfactor+=atom.get_bfactor()
            temp_coords=atom.coord
            temp_atommass=atom._assign_atom_mass()
            XYZ_M[0]+=temp_coords[0]*temp_atommass
            XYZ_M[1]+=temp_coords[1]*temp_atommass
            XYZ_M[2]+=temp_coords[2]*temp_atommass
            MassofAA+=atom._assign_atom_mass()
            TotalAtoms=AtomsNO
        CenterofMass=Atom.Atom(name='CenterofMass',coord=[i/MassofAA for i in XYZ_M],bfactor=SumBfactor/TotalAtoms,occupancy=1,altloc=None,fullname="CenterofMass",serial_number=None,element=None)
        residue.add(CenterofMass)
        AtomsofCenterofMass.append(CenterofMass)
    return AtomsofCenterofMass

def SelectAllAtoms(molecule):
    '''
    '''
    return molecule.get_atoms()

def SelectHeteroAtoms():
    '''
    NOTE: Will be written if necessary
    '''
    return True


##########----------##########----------##########----------##########----------##########----------

def main():
    filename=IO().filename
    ListOfFiles=PDBList().retrieve_pdb_file(filename,file_format='pdb',pdir='PDB')
    Structure=PDBParser(PERMISSIVE=False,QUIET=True).get_structure(filename,ListOfFiles)

    print SelectCAlpha(Structure)
    print SelectTipofAA(Structure)
    print SelectAACenterofMass(Structure)
    #print SelectAllAtoms()
    

    
    return True

if(__name__=='__main__'):
    main()