'''
Author: Pranav Khade(pranavk@iastate.edu)
SourceofInformation: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
'''
import numpy
import argparse

class Atom():
    def __init__(self,id,AtomName,Coordinates,Occupancy,bfactor,Element,Charge,parent):
        self.id=id
        self.AtomName=AtomName
        self.AlternateLocationIndicator=None #remove it later
        self.__parent=parent
        self.Chain=None
        self.Coordinates=Coordinates
        self.Occupancy=Occupancy
        self.bfactor=bfactor
        self.SegmentIdentifier=None
        self.Element=Element

    def GetParent(self):
        return self.__parent

    def CalcDist(self,another_atom):
        return numpy.linalg.norm(self.Coordinates-another_atom.Coordinates)

    

class Residue():
    def __init__(self,id,name,parent):
        self.__id=id
        self.__name=name
        self.__parent=parent
        self.__Atoms=None
    
    def __setitem__(self,id,Atom):
        try:
            self.__Atoms[id]=Atom
        except:
            self.__Atoms={}
            self.__Atoms[id]=Atom
    
    def GetID(self):
        return self.__id
    
    def GetName(self):
        return self.__name

    def GetParent(self):
        return self.__parent
    
    def GetAtoms(self):
        for i in sorted(self.__Atoms.keys()):yield self.__Atoms[i]
    
    def GetCAlpha(self):
        return [i for i in self.GetAtoms() if i.AtomName=='CA'][0]
    
    def GetCenterofGravity(self):
        '''
        NOTE: Yet to add the atomic masses.
        '''
        atoms=self.GetAtoms()
        AtomicMass=1
        XYZ_M=[0,0,0]
        MassofAA=0
        for i in atoms:
            XYZ_M[0]+=i.Coordinates[0]*AtomicMass
            XYZ_M[1]+=i.Coordinates[1]*AtomicMass
            XYZ_M[2]+=i.Coordinates[2]*AtomicMass
            MassofAA=MassofAA+AtomicMass
        return numpy.array([i/MassofAA for i in XYZ_M])
    
    def GetTipofAA(self):
        CAlpha=self.GetCAlpha()
        resname=self.GetName()
        TipofAA=None
        if(resname=='ALA' or resname=='GLY'):
            TipofAA=CAlpha
        else:
            MaxDistance=0
            for i in self.GetAtoms():
                tempdistance=CAlpha.CalcDist(i)
                if(tempdistance>MaxDistance):
                    MaxDistance=tempdistance
                    TipofAA=i
        return TipofAA


class Chain():
    def __init__(self,id):
        self.__id=id
        self.__Residues=None
    
    def __setitem__(self,ResidueNumber,Residue):
        try:
            self.__Residues[ResidueNumber]=Residue
        except:
            self.__Residues={}
            self.__Residues[ResidueNumber]=Residue

    def __getitem__(self,ResidueNumber):
        return self.__Residues[ResidueNumber]
    
    def GetID(self):
        return self.__id
    
    def GetResidues(self):
        for i in sorted(self.__Residues.keys()):yield self.__Residues[i]

class Model():
    def __init__(self,id,AllAtoms,AllResidues,AllChains):
        self.__id=id
        self.__AllAtoms=AllAtoms
        self.__AllResidues=AllResidues
        self.__AllChains=AllChains

    def __getitem__(self,ChainID):
        return self.__AllChains[ChainID]
    
    def GetAllChains(self):
        for i in sorted(self.__AllChains.keys()):yield self.__AllChains[i]

    def GetAllResidues(self):
        for i in sorted(self.__AllResidues.keys()):yield self.__AllResidues[i]
    
    def GetAllAtoms(self):
        for i in sorted(self.__AllAtoms.keys()):yield self.__AllAtoms[i]
    
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
                if(ResidueNumber not in AllResidues.keys()):AllResidues[ResidueNumber]=Residue(ResidueNumber,ResidueName,AllChains[ChainID])

                #Residue Added to the chain
                AllChains[ChainID].__setitem__(ResidueNumber,AllResidues[ResidueNumber])

                #Atom Defined
                id=int(_[6:11])
                AtomName=_[12:16].strip()
                Coordinates=numpy.array([float(_[30:38]),float(_[38:46]),float(_[46:54])])
                Occupancy=float(_[54:60])
                bfactor=float(_[60:66])
                Element=_[76:78].strip()
                Charge=_[78:80]
                AllAtoms[id]=Atom(id,AtomName,Coordinates,Occupancy,bfactor,Element,Charge,AllResidues[ResidueNumber])
                
                #Atom added to the residue
                AllResidues[ResidueNumber].__setitem__(id,AllAtoms[id])

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
    #print mol[0]['A'][2].GetCenterofGravity()
    print mol[0]['A'][2].GetTipofAA().id,mol[0]['A'][2].GetCAlpha().id
    return True

if(__name__=='__main__'):
    main()
