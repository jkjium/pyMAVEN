'''
Author: Pranav Khade(pranavk@iastate.edu)
SourceofInformation: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
'''
import numpy
import argparse

class Atom():
    def __init__(self,id,AtomName,Coordinates,Occupancy,bfactor,Element,Charge,ResidueNumber):
        self.id=id
        self.AtomName=AtomName
        self.AlternateLocationIndicator=None #remove it later
        self.ResidueNumber=ResidueNumber
        self.Chain=None
        self.Coordinates=Coordinates
        self.Occupancy=Occupancy
        self.bfactor=bfactor
        self.SegmentIdentifier=None
        self.Element=Element

class Residue():
    def __init__(self,id,name,ChainID):
        self.id=id
        self.name=name
        self.ChainID=ChainID
        self.Atoms=None
    
    def __setitem__(self,id,Atom):
        try:
            self.Atoms[id]=Atom
        except:
            self.Atoms={}
            self.Atoms[id]=Atom
    
    def GetAtoms(self):
        return self.Atoms.values()

class Chain():
    def __init__(self,id):
        self.id=id
        self.Residues=None
    
    def __setitem__(self,ResidueNumber,Residue):
        try:
            self.Residues[ResidueNumber]=Residue
        except:
            self.Residues={}
            self.Residues[ResidueNumber]=Residue

    def __getitem__(self,ResidueNumber):
        return self.Residues[ResidueNumber]
    
    def GetResidues(self):
        return self.Residues.values()

class Model():
    def __init__(self,id,AllAtoms,AllResidues,AllChains):
        self.__id=id
        self.__AllAtoms=AllAtoms
        self.__AllResidues=AllResidues
        self.__AllChains=AllChains

    def __getitem__(self,ChainID):
        return self.__AllChains[ChainID]
    
    def GetAllChains(self):
        return self.__AllChains.values()

    def GetAllResidues(self):
        return self.__AllResidues.values()
    
    def GetAllAtoms(self):
        return self.__AllAtoms.values()
    
    def GetChain(self,ChainID):
        return self.__AllChains[ChainID]
        
class Protein():
    def __init__(self,id,name,Models):
        self.id=id
        self.name=name
        self.Models=Models
    
    def __getitem__(self,ModelNumber):
        return self.Models[ModelNumber]

def LoadPDB(filename):
    Models=[]
    start=0
    fh=open(filename).read()
    frames=fh.split('\nMODEL')

    if(len(frames)>1):
        start=1
    else:
        start=0
    for FrameNumber,frame in enumerate(frames[start:]):
        #Map
        AllAtoms={}
        AllResidues={}
        AllChains={}
        AtomsResidueMap={}
        ResidueChainMap={}

        lines=frame.split('\n')   
        for _ in lines:
            if(_[0:4]=='ATOM'):
                #NOTE: MAPS CAN BE REMOVED SAFELY
                #Chain Defined
                ChainID=_[21]
                if(ChainID not in AllChains.keys()):AllChains[ChainID]=Chain(ChainID)

                #Residue Defined
                ResidueNumber=int(_[22:26].strip())
                ResidueName=_[17:20]
                if(ResidueNumber not in AllResidues.keys()):AllResidues[ResidueNumber]=Residue(ResidueNumber,ResidueName,ChainID)

                #Residue Added to the chain
                AllChains[ChainID].__setitem__(ResidueNumber,AllResidues[ResidueNumber])
                try:
                    if(ResidueNumber not in ResidueChainMap[ChainID]):ResidueChainMap[ChainID].append(ResidueNumber)
                except:
                    ResidueChainMap[ChainID]=[]
                    ResidueChainMap[ChainID].append(ResidueNumber)

                #Atom Defined
                id=int(_[6:11])
                AtomName=_[12:16].strip()
                Coordinates=numpy.array([float(_[30:38]),float(_[38:46]),float(_[46:54])])
                Occupancy=float(_[54:60])
                bfactor=float(_[60:66])
                Element=_[76:78].strip()
                Charge=_[78:80]

                AllAtoms[id]=Atom(id,AtomName,Coordinates,Occupancy,bfactor,Element,Charge,ResidueNumber)
                #Atom added to the residue
                AllResidues[ResidueNumber].__setitem__(id,AllAtoms[id])
                try:
                    AtomsResidueMap[ResidueNumber].append(id)
                except:
                    AtomsResidueMap[ResidueNumber]=[]
                    AtomsResidueMap[ResidueNumber].append(id)

                #What to do with these?
                AlternateLocationIndicator=_[16]
                CodeForInsertions=_[26]
                SegmentIdentifier=_[72:76]
        #print
        Models.append(Model(FrameNumber,AllAtoms,AllResidues,AllChains))
    return Protein(filename,None,Models)

#####-----#####-----#####-----#####-----#####-----

def DownloadPDB(filename):
    '''
    INFO: This class is used to download a PDB stucture from RCSB PDB
    '''
    import mechanize
    br = mechanize.Browser()
    response=br.open("https://files.rcsb.org/view/"+filename)
    folderandfile='pdb_files/'+filename
    open(folderandfile,'w').write(response.read())
    return True

def IO():
    '''
    INFO: Argument parser to the program for now.
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument('filename',metavar='PDBID')
    args=parser.parse_args()
    return args


#'5mti.pdb'
def main():
    filename=IO().filename
    mol=LoadPDB('5mti.pdb')
    #mol=LoadPDB('1ov9.pdb')

    #Following will select 0th frame from NMR, will select chain A from it and will select 2nd Amino acid from it.
    print mol[0]['A'][2].GetAtoms()
    return True

if(__name__=='__main__'):
    main()