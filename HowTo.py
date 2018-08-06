import Parser

Protein=Parser.LoadPDB('5mti.pdb')

'''NOTE: ADD print in front of anything to see the result'''

#BASICS:

#To get all the atoms in a molecule
Protein[0].GetAllAtoms()

#To get all the amino acids in a molecule
Protein[0].GetAllResidues()

#To get specific atoms from specific amino acid:
Protein[0]['A'][2].GetAtoms()


#Stuff we discussed:

#To get all C-Alpha atoms (Later it will be added as a function in a protein object):
CAlphas=[i.GetCAlpha() for i in Protein[0].GetAllResidues()]

#To get co ordinates of all CAlphas
[i.Coordinates for i in CAlphas]

#To get all tips of the amino acid:
Tips=[i.GetTipofAA for i in Protein[0].GetAllResidues()]

#To get co ordinates of all CAlphas
[i.Coordinates for i in CAlphas]