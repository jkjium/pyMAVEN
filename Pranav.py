'''
Author: Pranav Khade(pranavk@iastate.edu)
SourceofInformation: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
'''
import numpy

class Atom():
    def __init__(self,id):
        self.id=id
        self.AtomName=None
        self.AlternateLocationIndicator=None #remove it later
        self.ResidueName=None #get parent
        self.Chain=None
        self.Coordinates=None
        self.Occupancy=None
        self.bfactors=None
        self.SegmentIdentifier=None
        self.Element=None

class AminoAcid():
    def __init__(self):
        self.id=None
        self.name=None
        self.Chain=None

class Chain():
    def __init__(self):
        self.id=None
        self.name=None

class Model():
    def __init__(self):
        print "hello world"

class Protein():
    def __init__(self):
        print "Hello world"

def main():
    fh=open('5mti.pdb')
    for _ in fh:
        if(_[0:4]=='ATOM'):
            Serial=int(_[6:11])
            AtomName=_[12:16].strip()
            AlternateLocationIndicator=_[16]
            ResidueName=_[17:20]
            ChainID=_[21]
            ResidueNumber=_[22:26].strip()
            CodeForInsertions=_[26]
            Coordinates=numpy.array([float(_[30:38]),float(_[38:46]),float(_[46:54])])
            Occupancy=float(_[54:60])
            bfactors=float(_[60:66])
            SegmentIdentifier=_[72:76]
            Element=_[76:78].strip()
            Charge=_[78:80]
            
    return True

if(__name__=='__main__'):
    main()