#!/usr/bin/env python

from pyscf import lib,dft,ao2mo,scf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math,time,sys,os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*converting a masked element.*')
warnings.filterwarnings('ignore', message=".*The 'normed' kwarg is deprecated.*")
warnings.filterwarnings('ignore', message=".*Input data contains duplicate x,y points.*")

###global
thr_zero   = 1.e-5
thr_strict = 1.e-12
thr_large  = 1.e2
v_level    = None
dmrest     = None
morest     = None
nopen      = None

class basis_correction(lib.StreamObject):
  '''
  Basis-taylored DFT correction, implemented from the following papers:

    (1) 'Curing basis-set convergence of wave-function theory using
         density-functional theory: A systematically improvable approach'
         E. Giner, B. Pradines, A. Ferte, R. Assaraf, A. Savin, J. Toulouse
         10.1063/1.5052714
    (2) 'A Density-Based Basis-Set Correction For Wave Function Theory'
         P.-F. Loos, B. Pradines, A. Scemama, J. Toulouse, E. Giner
         10.1021/acs.jpclett.9b01176

  Equations in the following comments will refer to paper (2)

  The program prints out energies calculated with
    - a constant mu=0.0 over the grid (equals the native PBE calculated with this density)
    - a constant mu=0.5 over the grid (the commonly used value of mu)
    - a constant mu=avg over the grid (to show the differential effect of a mu(r))
    - the basis-taylored correction value, using mu(r)

  See `examples/dft/41-basis_correction.py`

  Input:
      obj            .a PySCF HF or DFT object
      dm             .a density matrix
      root           .a string attached to every file outputed
      mu             .will impose a constant value of mu over the grid
      grid_file      .a file containing a grid i,x,y,z,w
      grid_level     .an integer define the quality of the grid when it is built
      verbose        .verbosity trigger
      plot           .trigger plots
      origin         .origin point in case of plots
      direction      .direction vector for y=f(x) on a line in case of plots
      normal         .normal vector for z=f(x,y) on a plane in case of plots

  Attributes:
      run            .is the run RESTRICTED/UNRESTRICTED/RESTRICED_OPEN
      nmo            .the number of MO                        (inherited from 'obj')
      nelec          .the number of electrons                 (inherited from 'obj')
      nelec_alpha    .the number of alpha electrons           (inherited from 'obj')
      nelec_beta     .the number of beta electrons            (inherited from 'obj')
      ngrid          .the number of grid points               (inherited from 'obj')
      grid_coords    .the grid points coordinates   [N,3]     (inherited from 'obj')
      grid_weights   .the grid points weights       [N]       (inherited from 'obj')
      mo             .the MO coefficient            [nmo,nmo] (inherited from 'obj')
      v              .the integrals, in MOs         [nmo,nmo,nocc,nocc]
      aos_in_r       .the AOs                       [N,nmo]            (on the grid)
      mos_in_r       .the MOs                       [N,nmo]            (on the grid)
      rho_alpha      .the alpha-density             [N]                (on the grid)
      rho_beta       .the beta-density              [N]                (on the grid)
      rho            .the density                   [N]                (on the grid)
      eps_PBE        .epsilon_PBE                   [N]                (on the grid)
      eps_PBE_sr     .epsilon_PBE_short-range       [N]                (on the grid)
      mu_of_r        .mu(r)                         [N]                (on the grid)
      phi_square     .\sum_i i^2(r)                 [N]                (on the grid)
      v_phi_phi      .\sum_qj v[pq,ij]q(r)j(r)      [nmo,nval_beta,N]  (on the grid)
      n              .\sum_ij i(1)i(1) j(1)j(1)                [N]     (on the grid)
      f              .\sum_pq \sum_ij p(1)q(1)V[pq,ij]i(1)j(1) [N]     (on the grid)
      mu_average     .average value of mu_of_r over the grid
      max            .maximum distance for the grid points where mu_of_r<thr_large
      xyz            .the xyz coordinates transformed by `mol`
      bonds          .the bonds between atoms as [[i,j],[i,k],...]

  Output:
      e              .the basis-taylored correction

  Example:
  >>> from pyscf import gto
  >>> mol = gto.Mole()
  >>> mol.atom     = [['He', (0.,0.,0.)],]
  >>> mol.basis    = 'aug-cc-pvtz'
  >>> mol.build()
  >>>
  >>> from pyscf import dft
  >>> mks = dft.RKS(mol)
  >>> mks.kernel()
  >>>
  >>> bc=basis_correction(mks)
  >>> e_correction=bc.kernel()
  '''

  #  --------  #
  #    INIT    #
  #  --------  #
  def __init__(self,obj,
               dm=None,root='',mu=None,
               grid_file=None,grid_level=3,
               verbose=None,plot=False,
               origin=[0,0,0],direction=[0,0,1],normal=[1,0,0]):
      ###arguments
      self.obj         = obj
      self.dm          = dm
      self.root        = root
      self.mu          = mu
      self.grid_file   = grid_file
      self.grid_level  = grid_level
      self.plot        = plot
      self.origin      = origin
      self.direction   = direction
      self.normal      = normal

      ###verbosity
      global v_level
      if(hasattr(self.obj,'verbose')):
        v_level=self.obj.verbose
      if(verbose is not None):
        v_level= verbose
      else:
        v_level= 0

      ###cover
      printv0('\n\n')
      printv0('           ____            _       ____       _')
      printv0('          | __ )  __ _ ___(_)___  / ___|  ___| |_')
      printv0('          |  _ \ / _` / __| / __| \___ \ / _ \ __|')
      printv0('          | |_) | (_| \__ \ \__ \  ___) |  __/ |_')
      printv0('       ___|____/ \__,_|___/_|___/ |____/_\___|\__|')
      printv0('      / ___|___  _ __ _ __ ___  ___| |_(_) ___  _ __')
      printv0('     | |   / _ \| \'__| \'__/ _ \/ __| __| |/ _ \| \'_ \ ')
      printv0('     | |__| (_) | |  | | |  __/ (__| |_| | (_) | | | |')
      printv0('      \____\___/|_|  |_|  \___|\___|\__|_|\___/|_| |_|')
      printv0('                                                      ')
      printv0('\n\n')

      ###sanity of arguments
      self.sanity()

      ###REST?
      global morest
      if(len(self.obj.mo_coeff)==len(self.obj.mo_coeff[0])):
        morest           = True
        self.mo = [0,0]
        self.mo[0]       = self.obj.mo_coeff
        self.mo[1]       = self.obj.mo_coeff
        self.nmo         = self.mo[0].shape[0]
      else:
        morest           = False
        self.mo          = self.obj.mo_coeff
        self.nmo         = self.mo[0].shape[0]
      ###OPEN?
      global nopen
      if(len(self.obj.mol.nelec)==2):
        nopen            = True
        self.nelec       = sum(self.obj.mol.nelec)
        self.nelec_alpha = self.obj.mol.nelec[0]
        self.nelec_beta  = self.obj.mol.nelec[1]
      else:
        nopen            = False
        self.nelec       = sum(self.obj.mol.nelec)
        self.nelec_alpha = self.nelec/2.
        self.nelec_beta  = self.nelec/2.

      ###TODO get core,valence
      self.nval          = int(self.nelec)
      self.nval_alpha    = int(self.nelec_alpha)
      self.nval_beta     = int(self.nelec_beta)

      ###misc
      self.previous_origin    = None
      self.previous_direction = None
      self.previous_normal    = None
      self.previous_pts_line  = None
      self.previous_pts_plane = None
      self.labels = ['get_integrals',
                     'get_dm',
                     'get_grid',
                     'get_mos_in_r',
                     'get_rhos',
                     'get_eps_PBE',
                     'compute_phi_square',
                     'compute_v_phi_phi',
                     'compute_f_and_n',
                     'compute_mu_of_r',
                     'compute_average_mu_of_r',
                     'compute_eps_PBE_sr',
                     'energies',
                     'plot_data',
                     'TOTAL']
      self.times = [0.]*len(self.labels)

  #  ------------------  #
  #    MAIN FUNCTIONS    #
  #  ------------------  #
  def sanity(self):
      '''
      Check various fatal mistakes
      '''
      self.breakout=False
      ###obj
      if(not(hasattr(self.obj,'mo_coeff'))):
        print('>>%-37s  %20s'%('> DO NOT KNOW HOW TO mo_coeff',type(self.obj)))
        self.breakout=True
      if(not(isinstance(self.obj,scf.rohf.ROHF)
          or isinstance(self.obj,scf.hf.RHF)
          or isinstance(self.obj,scf.uhf.UHF))):
        print('>>%-37s  %20s'%('> DO NOT RECOGNIZE obj',type(self.obj)))
        self.breakout=True

      ###dm
      if(not(self.dm is None
          or isinstance(self.dm,str)
          or isinstance(self.dm,tuple)
          or isinstance(self.dm,np.ndarray))):
        print('>>%-37s  %20s'%('> DO NOT RECOGNIZE dm',type(self.dm)))
        self.breakout=True
      if((self.dm is None)and(not hasattr(self.obj,'make_rdm1'))):
        print('>>%-37s  %20s'%('> DO NOT KNOW HOW TO make_rdm1',type(self.obj)))
        self.breakout=True
      #if((isinstance(self.dm,np.ndarray))and(self.dm.shape!=(nmo,nmo))):
      #  print('>>> NOT THE RIGHT DIMENSIONS',self.dm.shape)
        self.breakout=True
      if(isinstance(self.dm,str)
         and not os.path.isfile(self.dm)):
        print('>>%-37s  %20s'%('> FILE DOES NOT EXIST dm',self.dm))
        self.breakout=True

      ###grid_file
      if(not(self.grid_file is None
          or isinstance(self.grid_file,str))):
        print('>>%-37s  %20s'%('> DO NOT RECOGNIZE grid',type(self.grid_file)))
        self.breakout=True
      if((self.grid_file is None)and
         (not(hasattr(self.obj,'grids')
           or hasattr(self.obj,'mol')))):
        print('>>%-37s  %20s'%('> DO NOT KNOW HOW TO build grid',type(self.obj)))
        self.breakout=True
      if(isinstance(self.grid_file,str)
         and not os.path.isfile(self.grid_file)):
        print('>>%-37s  %20s'%('> FILE DOES NOT EXIST grid_file',self.grid_file))
        self.breakout=True

      ###mu
      if(not(self.mu is None
          or isinstance(self.mu,float)
          or isinstance(self.mu,str))):
        print('>>%-37s  %20s'%('> DO NOT RECOGNIZE mu',type(self.mu)))
        self.breakout=True
      if(isinstance(self.mu,str)
         and not os.path.isfile(self.mu)):
        print('>>%-37s  %20s'%('> FILE DOES NOT EXIST mu',self.mu))
        self.breakout=True

      ###format
      if(not isinstance(self.grid_level,int)
         or self.grid_level<0 or self.grid_level>10):
        print('>>%-37s  %20s'%('> WRONG FORMAT grid_level',self.grid_level))
        self.breakout=True
      if(not isinstance(v_level,int)):
        print('>>%-37s  %20s'%('> WRONG FORMAT verbose',v_level))
        self.breakout=True
      if(not isinstance(self.plot,bool)):
        print('>>%-37s  %20s'%('> WRONG FORMAT plot',self.plot))
        self.breakout=True
      if(not isinstance(self.origin,(list,float))
        or len(self.origin)!=3):
        print('>>%-37s  %20s'%('> WRONG FORMAT origin',self.origin))
        self.breakout=True
      if(not isinstance(self.direction,(list,float))
        or len(self.direction)!=3):
        print('>>%-37s  %20s'%('> WRONG FORMAT direction',self.direction))
        self.breakout=True
      if(not isinstance(self.normal,(list,float))
        or len(self.normal)!=3):
        print('>>%-37s  %20s'%('> WRONG FORMAT normal',self.normal))
        self.breakout=True

  def initialization(self):
      '''
      Gather basic ingredients
      (density matrix, MOs on the grid,
       bi-electronic integrals, ...)
      and print out messages
      '''
      ###intro
      header('initialization')
      printv0('  %-37s'%('> initialization'))
      printv0('    %-35s  %20s'%('basis = ',self.obj.mol.basis))
      printv0('    %-35s  %20i'%('nmo = '  ,self.nmo))
      printv0('    %-35s  %20i'%('nelec = ',self.nelec))
      if(len(self.root)!=0 and self.root[-1]!='_' and self.root[-1]!='/'): self.root+='_'
      if('/' in self.root):
        if not os.path.exists(os.path.dirname(self.root)):
          os.makedirs(os.path.dirname(self.root))

      ###gather basic ingredients
      self.get_dm()
      self.get_integrals()
      self.get_grid()
      self.get_mos_in_r()
      self.get_rhos()
      self.get_eps_PBE()

  def get_integrals(self):
      '''
      Get the two-electron integrals involved,
      in MO and in physics notation.
      Only `v[all,all,nocc,nocc]` needed.

      Uses: ao2mo.outcore.general_iofree
      '''
      start=time.time()
      header('get_integrals')
      self.v = ao2mo.outcore.general_iofree(self.obj.mol,
              (self.mo[0], self.mo[0][:,:self.nval_alpha],
               self.mo[1], self.mo[1][:,:self.nval_alpha]), compact=False)
      self.v.shape=(self.nmo, self.nval_alpha, self.nmo, self.nval_alpha)
      self.v=self.v.transpose(0,2,1,3)
      dev_check_coulomb(self)
      end=time.time()
      self.times[0]+=end-start

  def get_dm(self):
      '''
      Get or import
      the density matrix

      Uses: make_rdm1
      '''
      start=time.time()

      ###produce the dm
      if(self.dm is None):
        printv0('  %-37s'%('> dm produced from obj'))
        self.dm=self.obj.make_rdm1()

      ###import the dm
      elif(isinstance(self.dm,str)):
        printv0('  %-37s  %20s'%('> dm imported from',self.dm))
        import_dm(self)

      ###export the dm
      export_dm(self)

      ###DM_REST?
      global dmrest
      if(len(self.dm)==len(self.dm[0])):
        tmp=self.dm
        self.dm=[tmp*0.5,tmp*0.5]
        dmrest=True
        del tmp
      else:
        dmrest=False
      dev_check_mos_dm(self)
      end=time.time()
      self.times[1]+=end-start

  def get_grid(self):
      '''
      Get or produce the DFT grid from PySCF
      or import it from a file

      Uses: dft.gen_grid.Grids
      '''
      start=time.time()
      header('get_grid')

      ###produce the grid
      if(self.grid_file==None):
        printv0('  %-37s'%('> grid produced from obj'))
        if(hasattr(self.obj,'grids')):
          self.obj.grids.level=self.grid_level
          self.obj.grids.build()
          self.grid_coords  = self.obj.grids.coords
          self.grid_weights = self.obj.grids.weights
          self.ngrid        = self.obj.grids.weights.shape[0]
        elif(hasattr(self.obj,'mol')):
          grids=dft.gen_grid.Grids(self.obj.mol)
          grids.level=self.grid_level
          grids.build()
          self.grid_coords  = grids.coords
          self.grid_weights = grids.weights
          self.ngrid        = grids.weights.shape[0]
          del grids

      ###import the grid
      elif(isinstance(self.grid_file,str)):
        printv0('  %-37s  %20s'%('> grid imported from',self.grid_file))
        import_grid(self)

      ###export the grid
      export_grid(self)
      printv0('    %-35s  %20i'%('ngrid =',self.ngrid))
      end=time.time()
      self.times[2]+=end-start

  def get_mos_in_r(self):
      '''
      Get the AOs and MOs
      over the grid

      Uses: dft.numint.eval_ao
      '''
      start=time.time()
      header('get_mos_in_r')
      ###aos
      self.aos_in_r = dft.numint.eval_ao(self.obj.mol,self.grid_coords)

      ###mos
      printv0('  %-37s'%('> convert aos to mos'))
      if(morest):
        self.mos_in_r = [0,0]
        self.mos_in_r[0] = np.dot(self.aos_in_r,self.mo[0])
        self.mos_in_r[1] = self.mos_in_r[0]
      else:
        self.mos_in_r = [0,0]
        self.mos_in_r[0] = np.dot(self.aos_in_r,self.mo[0])
        self.mos_in_r[1] = np.dot(self.aos_in_r,self.mo[1])
      dev_check_orthos(self)
      end=time.time()
      self.times[3]+=end-start

  def get_rhos(self):
      '''
      Get the density
      on the grid

      Uses: dft.numint.eval_rho
      '''
      start=time.time()
      header('get_rhos')
      if(dmrest):
        self.rho_alpha = dft.numint.eval_rho(self.obj.mol, self.aos_in_r, self.dm[0])
        self.rho_beta  = self.rho_alpha
        self.rho       = self.rho_alpha+self.rho_beta
      else:
        self.rho_alpha = dft.numint.eval_rho(self.obj.mol, self.aos_in_r, self.dm[0])
        self.rho_beta  = dft.numint.eval_rho(self.obj.mol, self.aos_in_r, self.dm[1])
        self.rho       = self.rho_alpha+self.rho_beta
      dev_check_rho_nelec(self)
      dev_check_mos_rho(self)
      end=time.time()
      self.times[4]+=end-start

  def get_eps_PBE(self):
      '''
      Get `epsilon_PBE`
      on the grid

      Note that this is such that
        E_c^PBE = \int eps_PBE(r) dr
      whereas what's given by the PySCF tool is so that
        E_c^PBE = \int eps_PBE(r) n(r) dr

      Uses: dft.numint.eval_ao
            dft.numint.eval_rho
            dft.libxc.eval_xc

      todo_later: find a way to get eps_PBE_value directly (no array)
      todo_later: looks difficult, would have to check lib.libxc in C
      todo_later: [edit] I don't even think that'd be so useful

      Note: could do a `dmrest` case, but not that costly anyway
      '''
      start=time.time()
      header('get_eps_PBE')
      printv0('  %-37s'%'> native PBE functional')
      aos = dft.numint.eval_ao(self.obj.mol,self.grid_coords,deriv=1)
      rho_alpha = dft.numint.eval_rho(self.obj.mol, aos, self.dm[0],xctype='GGA')
      rho_beta  = dft.numint.eval_rho(self.obj.mol, aos, self.dm[1],xctype='GGA')

      ###native PBE
      self.eps_PBE, = dft.libxc.eval_xc('PBE,PBE',(rho_alpha,rho_beta),1)[:1]
      self.eps_PBE=self.eps_PBE*self.rho
      printv0('    %-35s  %20.8f'%('Exc =',integrate(self,self.eps_PBE)))

      ###correlation only
      self.eps_PBE, = dft.libxc.eval_xc('0*HF,PBE',(rho_alpha,rho_beta),1)[:1]
      self.eps_PBE=self.eps_PBE*self.rho
      printv0('    %-35s  %20.8f'%('Ecorr =',integrate(self,self.eps_PBE)))
      del aos,rho_alpha,rho_beta
      end=time.time()
      self.times[5]+=end-start

  def compute_intermediaries(self):
      '''
      Compute or import `mu_of_r`
      '''
      ###path to mu_of_r
      if(self.mu==None):
        header('compute_intermediary')
        printv0('  %-37s'%('> compute intermediary and mu_of_r'))
        self.compute_phi_square()
        self.compute_v_phi_phi()
        self.compute_f_and_n()
        self.compute_mu_of_r()
        export_mu_of_r(self)
      elif(isinstance(self.mu,float)):
        printv0('  %-37s  %20.8f'%('> mu_of_r imposed',self.mu))
        self.mu_of_r = np.array([self.mu for grid_A in range(self.ngrid)])
      elif(isinstance(self.mu,str)):
        printv0('  %-25s  %32s'%('> mu_of_r imported from',self.mu))
        import_mu_of_r(self)

      ###average mu_of_r
      self.compute_average_mu_of_r()

  def compute_phi_square(self):
      '''
      Compute \sum_ij \phi^2_i(r)
      on the grid, as an intermediary to `n2(r,r)`

      (This is of course just `rho_alpha` and `rho_beta`,
      but might not be anymore when doing core/val)
      '''
      start=time.time()
      self.phi_square=[np.zeros(self.ngrid),np.zeros(self.ngrid)]
      for i in range(self.nval_alpha):
          self.phi_square[0]+=self.mos_in_r[0][:,i]*self.mos_in_r[0][:,i]
      for i in range(self.nval_beta):
          self.phi_square[1]+=self.mos_in_r[1][:,i]*self.mos_in_r[1][:,i]
      #dev for grid_A in range(self.ngrid):
      #dev   print 'phi_square %6i %13.8f'%(grid_A,self.phi_square[0][grid_A])
      end=time.time()
      self.times[6]+=end-start

  def compute_v_phi_phi(self):
      '''
      Compute \sum_pq V[q,p,j,i]\phi_p(r)\phi_q(r)
      on the grid, as an intermediary to `f(r,r)`
      '''
      start=time.time()
      self.v_phi_phi=np.zeros((self.nmo,self.nval_beta,self.ngrid))
      for p in range(self.nmo):
       for i in range(self.nval_beta):
        for j in range(self.nval_alpha):
         for q in range(self.nmo):
              self.v_phi_phi[p,i,:]+=self.v[p,q,i,j]\
                                    *self.mos_in_r[0][:,q]\
                                    *self.mos_in_r[0][:,j]
        progress('compute_v_phi_phi',p*self.nval_beta+i+1,self.nval_beta*self.nmo)
      printv0("")
      #dev for p in range(self.nmo):
      #dev  for i in range(self.nval_beta):
      #dev   for grid_A in range(self.ngrid):
      #dev    print 'v_phi_phi %6i %6i %6i %13.8f'%(p,i,grid_A,self.v_phi_phi[p,i,grid_A])
      end=time.time()
      self.times[7]+=end-start

  def compute_f_and_n(self):
      '''
      Compute `f(r,r)` and `n2(r,r)`
      on the grid

      (Both equations adapted here for the case of an HF wavefunction)
      Eq.(17.a) f(r1,r2) = \sum_pq \sum_ij \phi_p(r1)\phi_q(r2) V(pq,ij) \phi_i(r1)\phi_j(r2)
      Eq.(17.b) n(r1,r2) = \sum_ij \phi_i(r1)\phi_i(r1) \phi_j(r2)\phi_j(r2)
      '''
      start=time.time()
      self.n=self.phi_square[0]*self.phi_square[1]
      self.f=np.zeros(self.ngrid)
      for grid_A in range(self.ngrid):
        for p in range(self.nmo):
          for i in range(self.nval_beta):
            self.f[grid_A] += self.v_phi_phi[p,i,grid_A]\
                             *self.mos_in_r[1][grid_A,p]\
                             *self.mos_in_r[1][grid_A,i]
        progress('compute_f_and_n',grid_A+1,self.ngrid)
      printv0("")
      #dev for grid_A in range(self.ngrid):
      #dev  print 'n %6i %13.8f'%(grid_A,self.n[grid_A])
      #dev for grid_A in range(self.ngrid):
      #dev  print 'f %6i %13.8f'%(grid_A,self.f[grid_A])
      dev_check_f(self)
      end=time.time()
      self.times[8]+=end-start

  def compute_mu_of_r(self):
      '''
      Compute `mu(r)`
      on the grid

      Eq.(9)  mu(r) = \sqrt(pi)/2 W(r,r)
      Eq.(16) W(r1,r2) = f(r1,r2)/n2(r1,r2)
      '''
      start=time.time()
      self.mu_of_r = np.zeros(self.ngrid)
      got_in_A = 0
      got_in_B = 0
      got_in_C = 0
      for grid_A in range(self.ngrid):
          f_value=self.f[grid_A]
          n_value=self.n[grid_A]
          if(n_value<thr_strict):
              got_in_A+=1
              W_value = 1.e+10
          elif(f_value<-thr_zero):
              got_in_B+=1
              W_value = 1.e+10
          elif(f_value*n_value<-thr_zero):
              got_in_C+=1
              W_value = 1.e+10
          else:
              W_value = f_value/n_value
          self.mu_of_r[grid_A] = W_value*math.sqrt(math.pi)*0.5
          progress('compute_mu_of_r',grid_A+1,self.ngrid)
      printv0('')
      if(got_in_A!=0): printv0('    %-35s  %20i'%('warning: at some grid points, n~0'  ,got_in_A))
      if(got_in_B!=0): printv0('    %-35s  %20i'%('warning: at some grid points, f<0'  ,got_in_B))
      if(got_in_C!=0): printv0('    %-35s  %20i'%('warning: at some grid points, f.n<0',got_in_C))
      del f_value,n_value,W_value,got_in_A,got_in_B,got_in_C
      end=time.time()
      self.times[9]+=end-start

  def compute_average_mu_of_r(self):
      '''
      Compute the average of `mu_of_r`, \int mu(r) n(r) dr
      via integration over the grid
      '''
      start=time.time()
      den=self.grid_weights*self.rho
      mu,den=mask_over(thr_large,[self.mu_of_r,den])
      self.mu_average = np.ma.dot(den,mu,strict=True)/self.nelec
      printv0('    %-35s  %20.8f '%('average_mu',self.mu_average))
      end=time.time()
      self.times[10]+=end-start
      del den,mu

  def compute_eps_PBE_sr_value(self,mu_value,eps_PBE_value,rho_alpha_value,rho_beta_value):
      '''
      Compute `epsilon_PBE_short-range`
      **on a point of the grid**

      Note that this is such that
        E_c^srPBE = \int eps_PBE_sr(r) dr

      Eq(15.a) eps_PBE_sr = eps_PBE_value / (1+beta.mu^3)
      Eq(15.b) beta = 3*eps_PBE_value / (denom)

      Note that: n2_UEG = n^2.(1-eta^2).g0 = 4.rho_alpha.rho_beta.g0,
                 which is what's implemented here.

      This uses the work of
          Toulouse,Gori-Giorgi,Savin TCA 2005
      and Gori-Giorgi,Savin PRA 2006
      '''
      if(mu_value==0.):
          eps_PBE_sr = eps_PBE_value
      else:
          denom = 2.*math.sqrt(math.pi)*(1.-math.sqrt(2.))\
                * 4.*rho_alpha_value*rho_beta_value*g0_UEG(rho_alpha_value+rho_beta_value,thr_strict)
          if(abs(denom) > thr_strict):
              beta       = 3.*eps_PBE_value/denom
              eps_PBE_sr = eps_PBE_value/(1.+beta*mu_value**3)
          else:
              eps_PBE_sr = 0.
      return eps_PBE_sr

  def compute_eps_PBE_sr(self,mu_of_r):
      '''
      Put `epsilon_PBE_short-range`
      in an array of the grid
      '''
      start=time.time()
      eps_PBE_sr = np.zeros(self.ngrid)
      for grid_A in range(self.ngrid):
        mu_value           = mu_of_r[grid_A]
        eps_PBE_value      = self.eps_PBE[grid_A]
        rho_alpha_value    = self.rho_alpha[grid_A]
        rho_beta_value     = self.rho_beta[grid_A]
        eps_PBE_sr[grid_A] = self.compute_eps_PBE_sr_value(mu_value,eps_PBE_value,\
                                                           rho_alpha_value,rho_beta_value)
      end=time.time()
      self.times[11]+=end-start
      del mu_value,eps_PBE_value,rho_alpha_value,rho_beta_value
      return eps_PBE_sr

  def energies(self):
      '''
      Compute the following energies:
         E(mu=0.0 everywhere)
         E(mu=0.5 everywhere)
         E(mu=avg everywhere)
         E(mu(r))
      '''
      start=time.time()
      header('energies')
      printv0('  > energies ')
      eps=self.compute_eps_PBE_sr(np.array([0.0]*self.ngrid))
      print('    %-35s  %20.8f'%('E(mu = 0.0) =',integrate(self,eps)))

      eps=self.compute_eps_PBE_sr(np.array([0.5]*self.ngrid))
      print('    %-35s  %20.8f'%('E(mu = 0.5) =',integrate(self,eps)))

      eps=self.compute_eps_PBE_sr(np.array([self.mu_average]*self.ngrid))
      print('    %-35s  %20.8f'%('E(mu = avg) =',integrate(self,eps)))

      self.eps_PBE_sr=self.compute_eps_PBE_sr(self.mu_of_r)
      e=integrate(self,self.eps_PBE_sr)
      print('    %-35s  %20.8f'%('E_basis-set =',e))
      end=time.time()
      self.times[12]+=end-start
      del eps
      return e

  def plot_data(self):
      '''
      Plot (mostly) `mu_of_r`,
      `eps_PBE` and `eps_PBE_sr`
      '''
      start=time.time()
      if(self.plot):
        print('  %-37s'%('> prepare plots'))
        draw_mol_and_grid(self)
        plot_histo_grid(self)
        plot_line_plane(self,self.mu_of_r   ,'mu_of_r')
        plot_line_plane(self,self.eps_PBE_sr,'eps_sr',type='under')
        plot_line_plane(self,self.eps_PBE   ,'eps'   ,type='under')
        plot_line_plane(self,self.eps_PBE-self.eps_PBE_sr,'delta')
        #dev plot_of_r(self,self.mu_of_r,'mu_of_r_0.1',dim=0.1)
        #dev plot_of_r(self,self.mu_of_r,'mu_of_r_0.4',dim=0.4)
        #dev plot_of_r(self,self.mu_of_r,'mu_of_r_0.8',dim=0.8)
        #dev plot_of_r(self,self.mu_of_r,'mu_of_r_1.2',dim=1.2)
        #dev plot_of_r(self,self.mu_of_r,'mu_of_r_max',dim=self.max)
      end=time.time()
      self.times[13]+=end-start

  def kernel(self):
      '''
      Put everything in place
      to compute the correction

      Eq.(14) E = \int epsilon_PBE_short-range(r) dr
      '''
      start=time.time()

      if(self.breakout): return
      self.initialization()
      self.compute_intermediaries()
      e=self.energies()
      self.plot_data()

      end=time.time()
      self.times[14]+=end-start

      timings(self)
      return e

#  -----------------  #
#    TOOLS TO HELP    #
#  -----------------  #
def header(string):
    '''
    Formatting
    '''
    if(v_level>1):
      print('\n-'+string)

def printv0(*args):
    if(v_level>0):
      print(' '.join(str(i) for i in args))

def log(message,number):
    '''
    Print log lines if `v_level`
    '''
    if(v_level>1):
      if(abs(number)>thr_zero):
          print('..  %-35s  %20.8f  ****'%(message,number))
      else:
          print('..  %-35s  %20.8f'%(message,number))

def progress(message,value,endvalue):
    '''
    Print a dynamic progress bar, for example:
    '          1044/3740   [-------->                      ]  28%'
    '''
    if(v_level>0):
      bar_length = 17
      percent = float(value)/endvalue
      arrow   = '-'*int(round(percent * bar_length)-1)+'>'
      spaces  = ' '*(bar_length - len(arrow))
      line='\r        %-20s%6i/%-6i [%s] %3i'\
           %(message,value,endvalue,arrow+spaces,int(round(percent * 100)))
      sys.stdout.write(line+'%')
      sys.stdout.flush()
      del bar_length,percent,arrow,spaces,line

def timings(self):
    '''
    Print out the timings
    for all major routines
    '''
    if(v_level>0):
      header('timings')
      print('  %-37s'%('> timings'))
      for i in range(len(self.times)):
        print('    %-35s  %16.2f sec'%(self.labels[i],self.times[i]))
      print('')

def integrate(self,array):
    '''
    Compute \int array(r) dr
    over the grid
    '''
    return np.dot(array,self.grid_weights)

def to_mo(i):
    '''
    Translate an electron index
    into an MO index
    '''
    if(morest):
      return int(i/2.)
    else:
      print('Should not call `to_mo` from UNREST')
      exit(0)

def g0_UEG(rho_value,thr_strict):
    '''
    Parametrization
    needed for `epsilon_PBE_short-range`
    '''
    ahd = -0.36583
    d2  =  0.7524
    B   = -2.*ahd-d2
    C   =  0.08193
    D   = -0.01277
    E   =  0.001859
    if(abs(rho_value)>thr_strict):
        rs = (3./(4.*math.pi*rho_value))**(1./3.)
        x  = -d2*rs
        g0=0.5*(1. - B*rs + C*rs**2 + D*rs**3 + E*rs**4)*math.exp(x)
    else:
        g0=0.0
    del ahd,d2,B,C,D,E
    return g0

def g0_UEG_mu(mu_value,rho_value):
    '''
    Parametrization
    needed for `epsilon_PBE_short-range`
    '''
    alpha = (4./(9.*math.pi))**(1./3.)
    ahd   = -0.36583
    d2    =  0.7524
    B     = -2.*ahd-d2
    C     =  0.08193
    D     = -0.01277
    E     =  0.001859
    rs    = (3./(4.*math.pi*rho_value))**(1./3.)
    kf    = (alpha*rs)**(-1.)
    zeta  = mu_value/kf
    x     = -d2*rs*h(zeta)/ahd
    g0=(math.exp(x)/2.) * (1. - B*((h(zeta)**1.)/(ahd**1.))*(rs**1.)
                              + C*((h(zeta)**2.)/(ahd**2.))*(rs**2.)
                              + D*((h(zeta)**3.)/(ahd**3.))*(rs**3.)
                              + E*((h(zeta)**4.)/(ahd**4.))*(rs**4.))
    del alpha,ahd,d2,B,C,D,E,rs,kf,zeta,x
    return g0

def h(zeta):
    '''
    Parametrization
    needed for `epsilon_PBE_short-range`
    '''
    ahd    = -0.36583
    alpha  = (4./(9.*math.pi))**(1./3.)
    a1     = -(6.*alpha/math.pi)*(1.-math.log(2.))
    b1     = 1.4919
    b3     = 1.91528
    a2     = ahd*b3
    b2     = (a1-(b3*alpha/math.sqrt(math.pi)))/ahd
    h_zeta = (               a1*zeta**2. + a2*zeta**3.)\
           / (1. + b1*zeta + b2*zeta**2. + b3*zeta**3.)
    del ahd,alpha,a1,b1,b3,a2,b2
    return h_zeta

def import_dm(self):
    '''
    Import a density matrix from `self.dm` file,
    with format 'i j dm(i,j)'
             or 'a/b i j dm[a/b](i,j)'
    '''
    file = open(self.dm,'r')
    f = file.read().split('\n')
    file.close()
    tmp = np.zeros((self.nmo,self.nmo))
    tmpa = None
    for line in f:
      data=line.split()
      if len(data)==0:
          continue
      if len(data)==3:
        tmp[int(data[0]),int(data[1])]=float(data[2])
      elif len(data)==4:
        current=int(data[0])
        if(current==1 and tmpa is None):
          tmpa=tmp
          tmp = np.zeros((self.nmo,self.nmo))
        tmp[int(data[1]),int(data[2])]=float(data[3])
      else:
        print('>>>Expecting "i j dm(i,j)"')
        print('>>>       or "a/b i j dm[a/b](i,j)"')
        print(line,data)
        exit(0)
    if(tmpa is None):
      self.dm=tmp
    else:
      self.dm=[tmpa,tmp]
    del file,f,data

def import_grid(self):
    '''
    Import a grid from `self.grid_file` file,
    with format 'x y z weight'
    '''
    file = open(self.grid_file,'r')
    f = file.read().split('\n')
    file.close()
    self.grid_coords  = []
    self.grid_weights = []
    for line in f:
      data=line.split()
      if len(data)==0:
          continue
      if len(data)!=4:
          print('>>>Expecting "x y z weight"')
          print(line,data)
          exit(0)
      self.grid_coords.append([data[0],data[1],data[2]])
      self.grid_weights.append(data[3])
    self.grid_coords =np.asarray(self.grid_coords ,dtype=np.float32)
    self.grid_weights=np.asarray(self.grid_weights,dtype=np.float32)
    self.ngrid       =self.grid_weights.shape[0]
    del file,f,data

def import_mu_of_r(self):
    '''
    Import `mu_of_r` from a file
    with format 'x y z weight mu'
    '''
    file = open(self.mu,'r')
    f = file.read().split('\n')
    file.close()
    self.mu_of_r=[]
    grid_A=0
    for line in f:
        data=line.split()
        if len(data)==0:
            continue
        if len(data)!=5:
            print('>>>Expecting "x y z weight mu"')
            print(line,data)
            exit(0)
        if((abs(self.grid_coords[grid_A,0]-float(data[0]))>thr_zero)
         or(abs(self.grid_coords[grid_A,1]-float(data[1]))>thr_zero)
         or(abs(self.grid_coords[grid_A,2]-float(data[2]))>thr_zero)
         or(abs(self.grid_weights[grid_A] -float(data[3]))>thr_zero)):
           print('>>>The mu_of_r file does not correspond to the current grid.')
           exit(0)
        self.mu_of_r.append(data[4])
        grid_A+=1
    self.mu_of_r=np.asarray(self.mu_of_r,dtype=np.float32)
    if(self.mu_of_r.shape[0]!=self.ngrid):
      print('>>>The mu_of_r file does not correspond to the current grid.')
      exit(0)
    del file,f,data,grid_A

def export_dm(self):
    '''
    Export the density matrix to a file
    '''
    filename=self.root+'dm.data'
    if(os.path.isfile(filename)):
        printv0('    %-25s  %30s'%('no export, file exists',filename))
    else:
      f=open(filename,'w')
      for ab in range(2):
        for i in range(self.nmo):
          for j in range(self.nmo):
            f.write('%1i %5i %5i %15.8f\n'%(ab,i,j,self.dm[ab][i,j]))
      f.close()
      printv0('    %-35s  %20s'%('export dm',filename))
      del f
    del filename

def export_grid(self):
    '''
    Export the grid to a file
    '''
    filename=self.root+'grid.data'
    if(os.path.isfile(filename)):
        printv0('    %-25s  %30s'%('no export, file exists',filename))
    else:
      f=open(filename,'w')
      for grid_A in range(self.ngrid):
        f.write('%15.8f %15.8f %15.8f %15.8f\n'
              %(self.grid_coords[grid_A,0],
                self.grid_coords[grid_A,1],
                self.grid_coords[grid_A,2],
                self.grid_weights[grid_A]))
      f.close()
      printv0('    %-35s  %20s'%('export grid',filename))
      del f
    del filename
    return

def export_mu_of_r(self):
    '''
    Export `mu_of_r` to a file
    '''
    filename=self.root+'mu_of_r.data'
    if(os.path.isfile(filename)):
        printv0('    %-25s  %30s'%('no export, file exists',filename))
    else:
      f=open(filename,'w')
      for grid_A in range(self.ngrid):
        f.write('%15.8f %15.8f %15.8f %15.8f %25.8f\n'
              %(self.grid_coords[grid_A,0],
                self.grid_coords[grid_A,1],
                self.grid_coords[grid_A,2],
                self.grid_weights[grid_A],
                self.mu_of_r[grid_A]))
      f.close()
      printv0('    %-35s  %20s'%('export mu_of_r',filename))
      del f
    del filename

def mask_outside(dim,list):
    '''
    Mask first array in list
    based of values of the other arrays
    Typically: mask `f` based on `x,y,z` in [f,x,y,z]
    '''
    output=list[0]
    for i in range(len(list)-1):
      output=np.ma.masked_where(abs(list[i+1])>dim,output,copy=True)
    return output

def mask_over(value,list):
    '''
    Mask all arrays in `list`
    based on values of the first
    Typically: mask `f,x,y,z` based on `f` in [f,x,y,z]
    '''
    output=[0]*len(list)
    for i in range(len(list)):
      output[i]=np.ma.masked_where(abs(list[0])>value,list[i],copy=True)
    return output

def mask_under(value,list):
    '''
    Mask all arrays in `list`
    based on values of the first
    Typically: mask `f,x,y,z` based on `f` in [f,x,y,z]
    '''
    output=[0]*len(list)
    for i in range(len(list)):
      output[i]=np.ma.masked_where(abs(list[0])<value,list[i],copy=True)
    return output

def plot_histo_grid(self):
    '''
    Plot a histogram of
    the distance of grid points to zero.
    '''
    d=np.sqrt(np.einsum('ij,ij->i',self.grid_coords,self.grid_coords))
    ###plot
    filename=self.root+'histo.png'
    plt.clf()
    plt.hist(d)
    plt.savefig(filename)
    plt.close()
    printv0('    %-22s  %33s'%('histogram of the grid',filename))
    ###get `max`
    mu,d=mask_over(10,[self.mu_of_r,d])
    self.max=np.ma.max(d.compressed())
    del d,filename,mu

def plot_line_plane(self,f,name,type='over'):
    '''
    Plot data in the form of:
      - f(x) on a line
      - f(x,y) as a contour in 2d
      - f(x,y) as a contour in 3d

    Note: other contour ideas are for example:
      - contours=ax.tricontour(tri,z, 10, colors='black')
    '''
    import matplotlib.tri as mtri
    ###z=f(x)
    x,pts_line=select_line(self,self.direction,self.origin)
    z=f[pts_line]
    if(type=='over'):
      z,x=mask_over(1e1,[z,x])
    elif(type=='under'):
      z,x=mask_under(1e-4,[z,x])
    z=z.compressed()
    x=x.compressed()
    ###plot
    filename1=self.root+name+'_line.png'
    plt.clf()
    plt.scatter(x,z)
    plt.savefig(filename1)
    plt.close()
    printv0('    %-10s  %45s'%('on a line',filename1))

    ###z=f(x,y)
    x,y,pts_plane=select_plane(self,self.normal,self.origin)
    z=f[pts_plane]
    if(type=='over'):
      z,x,y=mask_over(1e1,[z,x,y])
    elif(type=='under'):
      z,x,y=mask_under(1e-4,[z,x,y])
    z=z.compressed()
    x=x.compressed()
    y=y.compressed()
    tri=mtri.Triangulation(x,y)
    ###plot
    filename2=self.root+name+'_2d.png'
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    contours=ax.tricontour(tri,z, 10, cmap=plt.cm.get_cmap('Blues'))
    plt.clabel(contours, inline=True, fontsize=8)
    plt.savefig(filename2)
    plt.close()
    printv0('    %-10s  %45s'%('contour2d',filename2))
    ###plot
    filename3=self.root+name+'_3d.png'
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.plot_trisurf(tri,z,cmap=plt.cm.get_cmap('Blues'))
    ax.scatter(x,y,z, marker='.', s=10, c="gray", alpha=0.5)
    plt.savefig(filename3)
    plt.close()
    printv0('    %-10s  %45s'%('contour3d',filename3))
    del x,y,z,tri,filename1,filename2,filename3,\
        contours,fig,ax,pts_line,pts_plane

def plot_of_r(self,f,name,dim=2,x=None,y=None,z=None):
    '''
    Plot any 3d function `f(x,y,z)`

    Note: other cmap are for example:
      - Reds,Greens: meh,
      - cubehelix: ok,
      - RdGy_r,RdBu: good
    '''
    if(x is None): x=self.grid_coords[:,0]
    if(y is None): y=self.grid_coords[:,1]
    if(z is None): z=self.grid_coords[:,2]
    f0, = mask_over(thr_large,[f])
    f1 = mask_outside(dim,[f0,x,y,z])
    ###plot
    filename=self.root+name+'.png'
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x, y, z, c=f1,alpha=0.7,s=50,lw=2,
                     cmap=plt.cm.get_cmap('Blues'),vmin=f0.min(),vmax=f0.max())
    fig.colorbar(img)
    plt.savefig(filename)
    plt.close()
    printv0('    %-25s  %30s'%(name,filename))
    del ax,f0,f1,img,filename,fig

def qualifies(vec,direction,thresh):
    '''
    Check if `vec = current_pt - origin_pt`
    is on the line with directive `direction`
    i.e. check that
      vec[0]/direction[0]=vec[1]/direction[1]=vec[2]/direction[2]
    or that
      vec[i]=0 if direction[i]=0
    '''
    qualifies=False
    if  (direction[0]==0. and direction[1]==0.):
      if(abs(vec[0])<thresh and abs(vec[1])<thresh):qualifies=True
    elif(direction[0]==0. and direction[2]==0.):
      if(abs(vec[0])<thresh and abs(vec[2])<thresh):qualifies=True
    elif(direction[1]==0. and direction[2]==0.):
      if(abs(vec[1])<thresh and abs(vec[2])<thresh):qualifies=True
    elif(direction[0]==0.):
      if(abs(vec[0])<thresh and abs(vec[1]/direction[1]-vec[2]/direction[2])<thresh):qualifies=True
    elif(direction[1]==0.):
      if(abs(vec[1])<thresh and abs(vec[0]/direction[0]-vec[2]/direction[2])<thresh):qualifies=True
    elif(direction[2]==0.):
      if(abs(vec[2])<thresh and abs(vec[0]/direction[0]-vec[1]/direction[1])<thresh):qualifies=True
    else:
      if(abs(vec[0]/direction[0]-vec[1]/direction[1])<thresh\
     and abs(vec[0]/direction[0]-vec[2]/direction[2])<thresh):qualifies=True
    return qualifies

def select_line(self,direction,origin):
    '''
    Pick up points of the grid
    on a line defined
    by it's directive vector
    and an origin point

    Returns coordinates rotated
    and translated to a corresponding basis
    and the list of selected points
    '''
    ###not redo if already went there!
    if(not(self.previous_direction is None
        or self.previous_origin is None)
       and(direction==self.previous_direction
          and origin==self.previous_origin)):
      matrix=rotate_coords(direction)
      new_coords=np.einsum('ij,Aj->Ai',matrix,self.grid_coords)-origin
      del matrix
      return new_coords[self.previous_pts_line,2],self.previous_pts_line

    ###select pts of the line
    self.previous_direction=direction
    self.previous_origin=origin
    direction=[i/np.linalg.norm(direction) for i in direction]
    pts_line=[]
    for grid_A in range(self.ngrid):
      vec=[self.grid_coords[grid_A,i]-origin[i] for i in range(3)]
      if(qualifies(vec,direction,1e-1)):
        pts_line.append(grid_A)
    self.previous_pts_line=pts_line
    matrix=rotate_coords(direction)
    new_coords=np.einsum('ij,Aj->Ai',matrix,self.grid_coords)-origin

    ###testing
    if(True):
      dim=2.0
      plt.clf()
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.set_aspect('equal')
      ax.set_xlim(-dim,dim)
      ax.set_ylim(-dim,dim)
      ax.set_zlim(-dim,dim)
      ###plot atoms
      new_xyz=np.einsum('ij,Aj->Ai',matrix,self.xyz)-origin
      newx=new_xyz[:,0]
      newy=new_xyz[:,1]
      newz=new_xyz[:,2]
      ax.scatter(newx,newy,newz, c='Blue',s=500,alpha=1.)
      ###plot bonds
      for b in range(len(self.bonds)):
        x1=[newx[i] for i in self.bonds[b]]
        y1=[newy[i] for i in self.bonds[b]]
        z1=[newz[i] for i in self.bonds[b]]
        ax.plot(x1,y1,z1,'b-',lw=5)
      ###plot selected grid points
      x=new_coords[pts_line,0]
      y=new_coords[pts_line,1]
      z=new_coords[pts_line,2]
      x=mask_outside(dim,[x,x,y,z])
      y=mask_outside(dim,[y,x,y,z])
      z=mask_outside(dim,[z,x,y,z])
      ax.scatter(x,y,z, c='Gray',alpha=0.5,s=25)
      plt.savefig(self.root+'pts_of_line.png')
      plt.close()
      printv0('    %-15s  %40s'%('selected line',self.root+'pts_of_line.png'))
      del dim,fig,ax,new_xyz,newx,newy,newz,x,y,z
    del vec,matrix
    return new_coords[pts_line,2],pts_line

def select_plane(self,normal,origin):
    '''
    Pick up points of the grid
    on a plane defined
    by it's normal vector
    and an origin point

    Returns coordinates rotated
    and translated to a corresponding basis
    and the list of selected points
    '''
    ###not redo if already went there!
    if(not(self.previous_normal is None
        or self.previous_origin is None)
       and(normal==self.previous_normal
       and origin==self.previous_origin)):
      matrix=rotate_coords(normal)
      new_coords=np.einsum('ij,Aj->Ai',matrix,self.grid_coords)-origin
      del matrix
      return new_coords[self.previous_pts_plane,0],\
             new_coords[self.previous_pts_plane,1],\
             self.previous_pts_plane

    ###select pts of the line
    self.previous_normal=normal
    self.previous_origin=origin
    normal=[i/np.linalg.norm(normal) for i in normal]
    d=np.dot(normal,origin)
    pts_plane=[]
    for grid_A in range(self.ngrid):
      abc=np.dot(normal,self.grid_coords[grid_A])
      if(abs(abc-d)<1e-1): pts_plane.append(grid_A)
    self.previous_pts_plane=pts_plane
    matrix=rotate_coords(normal)
    new_coords=np.einsum('ij,Aj->Ai',matrix,self.grid_coords)-origin

    ###testing
    if(True):
      dim=2.0
      plt.clf()
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.set_aspect('equal')
      ax.set_xlim(-dim,dim)
      ax.set_ylim(-dim,dim)
      ax.set_zlim(-dim,dim)
      ###plot atoms
      new_xyz=np.einsum('ij,Aj->Ai',matrix,self.xyz)-origin
      newx=new_xyz[:,0]
      newy=new_xyz[:,1]
      newz=new_xyz[:,2]
      ax.scatter(newx,newy,newz, c='Blue',s=500,alpha=1.)
      ###plot bonds
      for b in range(len(self.bonds)):
        x1=[newx[i] for i in self.bonds[b]]
        y1=[newy[i] for i in self.bonds[b]]
        z1=[newz[i] for i in self.bonds[b]]
        ax.plot(x1,y1,z1,'b-',lw=5)
      ###plot selected grid points
      x=new_coords[pts_plane,0]
      y=new_coords[pts_plane,1]
      z=new_coords[pts_plane,2]
      x=mask_outside(dim,[x,x,y,z])
      y=mask_outside(dim,[y,x,y,z])
      z=mask_outside(dim,[z,x,y,z])
      ax.scatter(x,y,z, c='Gray',alpha=0.5,s=25)
      plt.savefig(self.root+'pts_of_plane.png')
      plt.close()
      printv0('    %-15s  %40s'%('selected plane',self.root+'pts_of_line.png'))
      del dim,fig,ax,new_xyz,newx,newy,newz,x,y,z
    del d,abc,matrix

    return new_coords[pts_plane,0],new_coords[pts_plane,1],pts_plane

def rotate_coords(vec):
    '''
    Provide the rotation
    matrix that will rotate
    anything to a basis formed
    around the single input vector `vec`
    '''
    vec=[i/np.linalg.norm(vec) for i in vec]
    if(vec[2]!=0):
      u1=[1,0,-vec[0]/vec[2]]
      u2=[0,1,-vec[1]/vec[2]]
      a=-vec[0]*vec[1]/(vec[0]**2+vec[2]**2)
    elif(vec[1]!=0):
      u1=[1,-vec[0]/vec[1],0]
      u2=[0,-vec[2]/vec[1],1]
      a=-vec[0]*vec[2]/(vec[0]**2+vec[1]**2)
    elif(vec[0]!=0):
      u1=[-vec[1]/vec[0],1,0]
      u2=[-vec[2]/vec[0],0,1]
      a=-vec[1]*vec[2]/(vec[1]**2+vec[0]**2)
    u2=[u2[i]+a*u1[i] for i in range(3)]
    w1=[i/np.linalg.norm(u1) for i in u1]
    w2=[i/np.linalg.norm(u2) for i in u2]
    del u1,u2,a
    return np.asarray([w1,w2,vec])

def draw_mol_and_grid(self):
    '''
    Draw the molecule
    and the grid points
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.set_aspect('equal')
    ###plot grid pts
    x=mask_outside(2.0,[self.grid_coords[:,0],self.grid_coords[:,0], self.grid_coords[:,1], self.grid_coords[:,2]])
    y=mask_outside(2.0,[self.grid_coords[:,1],self.grid_coords[:,0], self.grid_coords[:,1], self.grid_coords[:,2]])
    z=mask_outside(2.0,[self.grid_coords[:,2],self.grid_coords[:,0], self.grid_coords[:,1], self.grid_coords[:,2]])
    ax.scatter(x,y,z, c='Gray',alpha=0.08,s=25)
    ###get info about the molecule
    vdw={'H' :1.20,'He':1.40,
         'Li':1.82,'C' :1.70,'N' :1.55,'O' :1.52,'F' :1.47,'Ne':1.54,
         'Na':2.27,'Mg':1.73,'Si':2.10,'P' :1.80,'S' :1.80,'Cl':1.75,'Ar':1.88,
         'K' :2.75,'Ni':1.63,'Cu':1.40,'Zn':1.39,'Ga':1.87,'As':1.85,'Se':1.90,'Br':1.85,'Kr':2.02,
         'Pd':1.63,'Ag':1.72,'Cd':1.58,'In':1.93,'Sn':2.17,'Te':2.06,'I' :1.98,'Xe':2.16,
         'Pt':1.75,'Au':1.66,'Hg':1.55,'Tl':1.96,'Pb':2.02,
         'U' :1.86}
    self.xyz=[]
    elt=[]
    atomx=[]
    atomy=[]
    atomz=[]
    for ia,atom in enumerate(self.obj.mol._atom):
      self.xyz.append([i for i in atom[1]])
      elt.append(atom[0])
      atomx.append(atom[1][0])
      atomy.append(atom[1][1])
      atomz.append(atom[1][2])
    self.xyz=np.asarray(self.xyz)
    s=[700*vdw[i] for i in elt]
    ###get bonds
    self.bonds=[]
    for i in range(len(elt)):
     for j in range(i):
      dist=np.linalg.norm(self.xyz[i]-self.xyz[j])
      if(dist<(vdw[elt[i]]+vdw[elt[j]])):
        self.bonds.append([i,j])
    ###plot bonds
    for b in range(len(self.bonds)):
      x1=[atomx[i] for i in self.bonds[b]]
      y1=[atomy[i] for i in self.bonds[b]]
      z1=[atomz[i] for i in self.bonds[b]]
      ax.plot(x1,y1,z1,'b-',lw=5)
    ###plot atoms
    ax.scatter(atomx,atomy,atomz, s=s,alpha=1.,c='Blue')
    plt.savefig(self.root+'molecule_and_grid.png')
    plt.close()
    printv0('    %-10s  %45s'%('molecule',self.root+'molecule_and_grid.png'))
    ###info to help choose what to give as `direction` and `normal`
    if(False):
      for i in range(len(elt)):
        print '      %-3s [%13.8f %13.8f %13.8f]'%(elt[i],self.xyz[i,0],self.xyz[i,1],self.xyz[i,2])
      for b in range(len(self.bonds)):
        print '    %2s-%-2s [%13.8f %13.8f %13.8f]'\
        %(elt[self.bonds[b][0]],elt[self.bonds[b][1]],\
          self.xyz[self.bonds[b][0]][0]-self.xyz[self.bonds[b][1]][0],\
          self.xyz[self.bonds[b][0]][1]-self.xyz[self.bonds[b][1]][1],\
          self.xyz[self.bonds[b][0]][2]-self.xyz[self.bonds[b][1]][2])
    del fig,ax,x,y,z,vdw,elt,atomx,atomy,atomz,s

#  -----------------------------  #
#    SOME CHECKS FOR FUNCTIONS    #
#  -----------------------------  #
def dev_check_mos_dm(self):
  '''
  Check that DM=mo.occ.mo
  and tr(DM.S)=nelec
  and tr(CT.S.C)=1mat
  '''
  if(v_level>1):
    occ=[[1]*self.nelec_alpha+[0]*(self.nmo-self.nelec_alpha),
         [1]*self.nelec_beta +[0]*(self.nmo-self.nelec_beta )]
    a=np.einsum('mi,i,ni->mn',self.mo[0],occ[0],self.mo[0])
    log('norm(dm-dm) (alpha) (onlySR)',np.linalg.norm(self.dm[0]-a))
    a=np.einsum('mi,i,ni->mn',self.mo[1],occ[1],self.mo[1])
    log('norm(dm-dm) (beta) (onlySR)',np.linalg.norm(self.dm[1]-a))

    if(hasattr(self.obj,'get_ovlp')):
      a=np.einsum('ij,ij',self.dm[0],self.obj.get_ovlp()) - self.nelec_alpha
      log('tr(DM.S)=nelec',a)
      a=np.einsum('ij,ij',self.dm[1],self.obj.get_ovlp()) - self.nelec_beta
      log('tr(DM.S)=nelec',a)

      a=np.diag([1]*self.nmo)-np.einsum('ij,jk,kl->il',self.mo[0].T,self.obj.get_ovlp(),self.mo[0])
      log('norm(C.T.S.C-1mat)',np.linalg.norm(a))
      a=np.diag([1]*self.nmo)-np.einsum('ij,jk,kl->il',self.mo[1].T,self.obj.get_ovlp(),self.mo[1])
      log('norm(C.T.S.C-1mat)',np.linalg.norm(a))
    del a

def dev_check_orthos(self):
  '''
  Check that <AO|AO>=S_AO and <MO|MO>=1
  via integrations over the grid
  '''
  #TODO For some reason the <AO|AO> and
  #TODO <MO|MO> are not always right..!?!
  #TODO Everything else works though, so...
  if(v_level>1):
    if(hasattr(self.obj,'get_ovlp')):
      ref=self.obj.get_ovlp()
      a=np.zeros((self.nmo,self.nmo))
      for i in range(self.nmo):
       for j in range(self.nmo):
        a[i,j]=integrate(self,self.aos_in_r[:,i]*self.aos_in_r[:,j])
      log('norm(ovlp-ovlp)=0 (AO)',np.linalg.norm(a-ref))
      ###print <AO|AO>-ovlp
      #dev print(''.join(['%20i'%(i+1) for i in range(self.nmo)]))
      #dev for i in range(self.nmo):
      #dev   print('%8i'%(i+1)+''.join(['%20.8f'%(a[i,j]-ref[i,j]) for j in range(i+1)]))

    ref=np.diag([1]*self.nmo)
    a=np.zeros((self.nmo,self.nmo))
    for i in range(self.nmo):
     for j in range(self.nmo):
      a[i,j]=integrate(self,self.mos_in_r[0][:,i]*self.mos_in_r[0][:,j])
    log('norm(ovlp-ovlp)=0 (MOa)',np.linalg.norm(a-ref))
    ###print <MO|MO>-1mat
    #dev print(''.join(['%20i'%(i+1) for i in range(self.nmo)]))
    #dev for i in range(self.nmo):
    #dev   print('%8i'%(i+1)+''.join(['%20.8f'%(a[i,j]-ref[i,j]) for j in range(i+1)]))

    ref=np.diag([1]*self.nmo)
    a=np.zeros((self.nmo,self.nmo))
    for i in range(self.nmo):
     for j in range(self.nmo):
      a[i,j]=integrate(self,self.mos_in_r[1][:,i]*self.mos_in_r[1][:,j])
    log('norm(ovlp-ovlp)=0 (MOb)',np.linalg.norm(a-ref))
    ###print <MO|MO>-1mat
    #dev print(''.join(['%20i'%(i+1) for i in range(self.nmo)]))
    #dev for i in range(self.nmo):
    #dev   print('%8i'%(i+1)+''.join(['%20.8f'%(a[i,j]-ref[i,j]) for j in range(i+1)]))
    del a,ref

def dev_check_mos_rho(self):
  '''
  Check `rho` over the grid
  in several different ways
  '''
  if(v_level>1):
    rho=np.einsum('pi,ij,pj->p', self.aos_in_r, self.dm[0], self.aos_in_r)
    log('norm(rho-rho)=0 (dm)',np.linalg.norm(rho-self.rho_alpha))
    rho=np.einsum('pi,ij,pj->p', self.aos_in_r, self.dm[1], self.aos_in_r)
    log('norm(rho-rho)=0 (dm)',np.linalg.norm(rho-self.rho_beta))

    occ=[[1]*self.nelec_alpha+[0]*(self.nmo-self.nelec_alpha),
         [1]*self.nelec_beta +[0]*(self.nmo-self.nelec_beta )]
    rho=np.zeros(self.ngrid)
    for i in range(self.nmo):
        rho+=occ[0][i]*self.mos_in_r[0][:,i]**2
    log('norm(rho-rho)=0 (occ) (onlySR)',np.linalg.norm(rho-self.rho_alpha))

    rho=np.zeros(self.ngrid)
    for i in range(self.nmo):
        rho+=occ[1][i]*self.mos_in_r[1][:,i]**2
    log('norm(rho-rho)=0 (occ) (onlySR)',np.linalg.norm(rho-self.rho_beta))

    rho=np.zeros(self.ngrid)
    for i in range(self.nval_alpha):
        rho+=self.mos_in_r[0][:,i]**2
    log('norm(rho-rho)=0 (alpha) (onlySR)',np.linalg.norm(rho-self.rho_alpha))

    rho=np.zeros(self.ngrid)
    for i in range(self.nval_beta):
        rho+=self.mos_in_r[1][:,i]**2
    log('norm(rho-rho)=0 (beta) (onlySR)',np.linalg.norm(rho-self.rho_beta))
    del rho

def dev_check_coulomb(self):
  '''
  Check that
  1/2 sum_elec V_ij^ij = E_coul
  '''
  if(v_level>1):
    if(morest):
      a=0
      for elec_i in range(self.nval):
        for elec_j in range(self.nval):
          a+=0.5*self.v[to_mo(elec_i),to_mo(elec_j),
                        to_mo(elec_i),to_mo(elec_j)]
      log('sum_v=Ecoul', a)
    else:
      a=0
      for elec_i in range(self.nval_alpha):
        for elec_j in range(self.nval_alpha):
          a+=0.5*self.v[elec_i,elec_j,elec_i,elec_j]
      for elec_i in range(self.nval_beta):
        for elec_j in range(self.nval_beta):
          a+=0.5*self.v[elec_j,elec_i,elec_j,elec_i]
      for elec_i in range(self.nval_alpha):
        for elec_j in range(self.nval_beta):
          a+=0.5*self.v[elec_i,elec_j,elec_i,elec_j]
          a+=0.5*self.v[elec_j,elec_i,elec_j,elec_i]
      log('sum_v=Ecoul (if mo_alpha=mo_beta)', a)
    del a

def dev_check_rho_nelec(self):
  '''
  Check that \int rho(r) dr = nelec
  via integration over the grid
  '''
  if(v_level>1):
    log('int(rho)-nelec'      ,integrate(self,self.rho)      -self.nelec      )
    log('int(rho)-nelec_alpha',integrate(self,self.rho_alpha)-self.nelec_alpha)
    log('int(rho)-nelec_beta ',integrate(self,self.rho_beta )-self.nelec_beta )

def dev_check_f(self):
  '''
  Check \int f(r1,r2) dr1 dr2
  with analytical integration over r2
  and integration over the grid for r1
  '''
  if(v_level>1):
    header('some checks')
    int_f=np.zeros(self.ngrid)
    for j in range(self.nval_alpha):
      for p in range(self.nmo):
        prod=self.mos_in_r[0][:,j]*self.mos_in_r[0][:,p]
        for i in range(self.nval_beta):
          int_f+=prod*self.v[p,i,j,i]
    log('int(f)=Ecoul (if mo_alpha=mo_beta)',  2.0*integrate(self,int_f))
    del int_f,prod

