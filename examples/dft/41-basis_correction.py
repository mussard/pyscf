#!/usr/bin/env python
from pyscf.dft import basis_correction as bc

'''
Examples of runs for the basis-correction tool
(see pyscf/dft/basis_correction.py).

The class takes as mandatory argument an SCF object `scf_obj`.

(Note that inheriting RHF/RKS and ROHF/ROKS objects was tested a lot ; UHF/UKS less so.)

The following optional arguments can be given:
  - dm          by default, the `scf_obj.make_rdm1()` is used
                but the user can give another RDM1 as a `numpy.ndarray`
                or the path to a file of the form "i j dm(i,j)" to
                import any RDM1.

  - grid_file   by default, the grid from `scf_obf.grids` or
                from `dft.gen_grid.Grids` is used, but the user can
                give the path to a file of the form "x y z weight"
                to import any grid

  - mu          by default, `mu_of_r` is computed from scratch
                but the user can impose an homogeneous value on the grid
                by giving a float here, or give the path to a file
                of the form "x y z weight mu" to import any mu(r)

Other optional arguments are:
  - root        a string attached to every file outputed
  - grid_level  an integer define the quality of the grid when it is built
  - verbose     verbosity trigger
  - plot        trigger plots
  - origin      origin point in case of plots
  - direction   direction vector for y=f(x) on a line in case of plots
  - normal      normal vector for z=f(x,y) on a plane in case of plots
'''
print('\n\n')

from pyscf import gto
mol = gto.Mole()
mol.verbose  = 1
mol.atom     = [['He', (0.,0.,0.)],]
mol.basis    = 'aug-cc-pvtz'
mol.spin     = 0
mol.symmetry = 1
mol.build()

from pyscf import scf
hf = scf.RHF(mol)
e=hf.scf()
print('::HF::   E = %13.6f'%e)

from pyscf import dft
ks = dft.RKS(mol)
ks.xc = 'PBE'
e=ks.kernel()
print('::PBE::  E = %13.6f'%e)

'''
Runs with different `dm` argument (default,make_rdm1,string)
'''
print('\n'+'-'*23+' ALL DEFAULTS '+'-'*24)
b=bc.basis_correction(ks)
b.kernel()

print('\n'+'-'*26+' HF RDM '+'-'*27)
b=bc.basis_correction(ks,dm=hf.make_rdm1())
b.kernel()

print('\n'+'-'*23+' RDM FROM FILE '+'-'*23)
b=bc.basis_correction(ks,dm='dm.data')
b.kernel()

'''
Run with the `grid_file` argument (string)
'''
print('\n'+'-'*23+' GRID FROM FILE '+'-'*23)
b=bc.basis_correction(ks,grid_file='grid.data')
b.kernel()

'''
Runs with different `mu` argument (float,string)
'''
print('\n'+'-'*21+' CONSTANT MU = 0.5 '+'-'*21)
b=bc.basis_correction(ks,mu=0.5)
b.kernel()

print('\n'+'-'*23+' MU FROM FILE '+'-'*23)
b=bc.basis_correction(ks,mu='mu_of_r.data')
b.kernel()

'''
Run with verbosity
'''
print('\n'+'-'*25+' VERBOSITY '+'-'*25)
b=bc.basis_correction(ks,verbose=2)
b.kernel()

'''
Run with plots
This will trigger the drawing of:
    - an histogram of the radial distribution of points around the origin
    - the molecule and the grid points
    - plots of `mu_of_r`, `eps_PBE`, `eps_PBE_sr` and `eps_PBE-eps_PBE_sr`
      on a line defined by (origin,direction)
      and on a plane defined by (origin,normal)
      (The grid points that are selected for the line
       and the plane are also plotted, for reference)
'''
print('\n'+'-'*27+' PLOTS '+'-'*27)
b=bc.basis_correction(ks,plot=True,root='plot_he/',
                         origin=[0,0,0],direction=[0,0,1],normal=[1,0,0])
b.kernel()

'''
Run with N2
'''
print('\n'+'-'*23+' N2 WITH PLOTS '+'-'*23)
from pyscf import gto
mol = gto.Mole()
mol.verbose  = 0
mol.atom     = [['N', (0.,0.,-1.0975/2.)],
                ['N', (0.,0.,+1.0975/2.)],]
mol.basis    = 'cc-pvtz'
mol.spin     = 0
mol.symmetry = 1
mol.build()

from pyscf import scf
hf = scf.RHF(mol)
e=hf.scf()
print('::HF::   E = %13.6f'%e)

from pyscf import dft
ks = dft.RKS(mol)
ks.xc = 'PBE'
e=ks.kernel()
print('::PBE::  E = %13.6f'%e)

b=bc.basis_correction(ks,plot=True,root='plot_n2/',grid_level=0,
                         origin=[0,0,0],direction=[0,0,1],normal=[1,0,0])
b.kernel()

