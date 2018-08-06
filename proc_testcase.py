import Parser
import commp as cp

def writesgcpdb(arglist):
	pass

def writecutoffcontact(arglist):
	pass

def writepairwisedist(arglist):
	pass

# need chain selector implementation
def writecapdb(arglist):
	'''
	write ca atoms in pdb format file
	'''
	if len(arglist) < 2:
		cp._er('Usag: python writecapdb pdbfile chain')

	infile = arglist[0]
	chainid = arglist[1]
	outfile = '%s.%s.ca' % (infile, chainid)

	p = Parser.LoadPDB(infile)
	#outstr = ''.join([r.GetCAlpha() for r in p[0].GetChain(chainid).GetResidues()])


def writeseq(arglist):
	'''
	write sequence for input pdb to a file
	input: 1t3r.pdb
	output: 1t3r.pdb.seq
	'''
	if len(arglist) < 2:
		cp._err('Usage: python writeseq pdbfile chain')

	infile = arglist[0]
	chainid = arglist[1]
	outfile = '%s.%s.seq' % (infile, chainid)

	p = Parser.LoadPDB(infile)
	#ca =[r.GetCAlpha() for r in p.GetAllResidues()]
	#outstr =''.join([cp.aa2a[r.GetName()] for r in p[0].GetAllResidues()])
	outstr =''.join([cp.aa2a[r.GetName()] for r in p[0].GetChain(chainid).GetResidues()])

	with open(outfile, 'w') as fout:
		fout.write('%s\n' % outstr)

if __name__ == '__main__':
	cp.dispatch(__name__)
	'''
	python proc_testcase.py writecapdb 1t3r.pdb B
	'''