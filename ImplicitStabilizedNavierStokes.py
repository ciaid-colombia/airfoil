from dolfin import *
import mshr
import matplotlib.pyplot as plt
import numpy as np
import ctypes, ctypes.util, numpy, scipy.sparse, scipy.sparse.linalg, collections
from numpy import array, zeros, ones, any, arange, isnan
from numpy.linalg import eigh as pyeig
from itertools import combinations
import os

def build_space(N_circle, N_bulk, u_in):
    """Prepare data benchmark. Return function
    space, list of boundary conditions and surface measure
    on the cylinder."""

    # Define domain
    center = Point(0.2, 0.2)
    radius = 0.05
    L = 2.2
    W = 0.41
    geometry = mshr.Rectangle(Point(0.0, 0.0), Point(L, W)) \
             - mshr.Circle(center, radius, N_circle)

    # Build mesh
    mesh = mshr.generate_mesh(geometry, N_bulk)

    # Construct facet markers
    bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    for f in facets(mesh):
        mp = f.midpoint()
        if near(mp[0], 0.0):  # inflow
            bndry[f] = 1
        elif near(mp[0], L):  # outflow
            bndry[f] = 2
        elif near(mp[1], 0.0) or near(mp[1], W):  # walls
            bndry[f] = 3
        elif mp.distance(center) <= radius:  # cylinder
            bndry[f] = 5

    # Build function spaces (Taylor-Hood)
    P2 = VectorElement("P", mesh.ufl_cell(), 1)
    P1 = FiniteElement("P", mesh.ufl_cell(), 1)
    TH = MixedElement([P2, P1])
    W = FunctionSpace(mesh, TH)

    # Prepare Dirichlet boundary conditions
    bc_walls = DirichletBC(W.sub(0), (0, 0), bndry, 3)
    bc_cylinder = DirichletBC(W.sub(0), (0, 0), bndry, 5)
    bc_in = DirichletBC(W.sub(0), u_in, bndry, 1)
    bc_out = DirichletBC(W.sub(1), 0., bndry, 2)
    bcs = [bc_cylinder, bc_walls, bc_in, bc_out]

    # Prepare surface measure on cylinder
    ds_circle = Measure("ds", subdomain_data=bndry, subdomain_id=5)

    return W, bcs, ds_circle

class InitialCondition(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
    def value_shape(self):
        return (3,)

def stab_d_type(argument):
	stabtype = {
    		'SUPG': 0.0,
    		'GLS': 1.0,
    		'VMS': -1.0
		}
	return stabtype.get(argument)

def stab_p_type(argument):
	stabtype = {
    		'SUPG': 0.0,
    		'GLS': 1.0,
    		'VMS': -1.0
		}
	return stabtype.get(argument)

def sym2asym(HH):
    if HH.shape[0] == 3:
        return array([HH[0,:],HH[1,:],\
                      HH[1,:],HH[2,:]])
    else:
        return array([HH[0,:],HH[1,:],HH[3,:],\
                      HH[1,:],HH[2,:],HH[4,:],\
                      HH[3,:],HH[4,:],HH[5,:]])

def c_cell_dofs(mesh,V):
   if V.ufl_element().is_cellwise_constant():
    return arange(mesh.num_cells()*mesh.geometry().dim()**2)
   else:
    return arange(mesh.num_vertices()*mesh.geometry().dim()**2)

def mesh_metric(mesh):
        # this function calculates a mesh metric (or perhaps a square inverse of that, see mesh_metric2...)
        cell2dof = c_cell_dofs(mesh,TensorFunctionSpace(mesh, "DG", 0))
        cells = mesh.cells()
        coords = mesh.coordinates()
        p1 = coords[cells[:,0],:]
        p2 = coords[cells[:,1],:]
        p3 = coords[cells[:,2],:]
        r1 = p1-p2; r2 = p1-p3; r3 = p2-p3
        Nedg = 3
        if mesh.geometry().dim() == 3:
          Nedg = 6
          p4 = coords[cells[:,3],:]
          r4 = p1-p4; r5 = p2-p4; r6 = p3-p4
        rall = zeros([p1.shape[0],p1.shape[1],Nedg])
        rall[:,:,0] = r1; rall[:,:,1] = r2; rall[:,:,2] = r3
        if mesh.geometry().dim() == 3:
          rall[:,:,3] = r4; rall[:,:,4] = r5; rall[:,:,5] = r6
        All = zeros([p1.shape[0],Nedg**2])
        inds = arange(Nedg**2).reshape([Nedg,Nedg])
        for i in range(Nedg):
          All[:,inds[i,0]] = rall[:,0,i]**2; All[:,inds[i,1]] = 2.*rall[:,0,i]*rall[:,1,i]; All[:,inds[i,2]] = rall[:,1,i]**2
          if mesh.geometry().dim() == 3:
            All[:,inds[i,3]] = 2.*rall[:,0,i]*rall[:,2,i]; All[:,inds[i,4]] = 2.*rall[:,1,i]*rall[:,2,i]; All[:,inds[i,5]] = rall[:,2,i]**2
        Ain = zeros([Nedg*2-1,Nedg*p1.shape[0]])
        ndia = zeros(Nedg*2-1)
        for i in range(Nedg):
          for j in range(i,Nedg):
              iks1 = arange(j,Ain.shape[1],Nedg)
              if i==0:
                  Ain[i,iks1] = All[:,inds[j,j]]
              else:
                  iks2 = arange(j-i,Ain.shape[1],Nedg)
                  Ain[2*i-1,iks1] = All[:,inds[j-i,j]]
                  Ain[2*i,iks2]   = All[:,inds[j,j-i]]
                  ndia[2*i-1] = i
                  ndia[2*i]   = -i
        
        A = scipy.sparse.spdiags(Ain, ndia, Ain.shape[1], Ain.shape[1]).tocsr()
        b = ones(Ain.shape[1])
        X = scipy.sparse.linalg.spsolve(A,b)
        #set solution
        XX = sym2asym(X.reshape([mesh.num_cells(),Nedg]).transpose())
        M = Function(TensorFunctionSpace(mesh,"DG", 0))
        M.vector().set_local(XX.transpose().flatten()[cell2dof])
        return M

def advnorm(wk):
        (uk, pk) = wk.split(True)
        (ux, uy) = uk.split(True)     
        V = uk.function_space()
        mesh = uk.function_space().mesh()
        degree = V.ufl_element().degree()
        W1 = FunctionSpace(mesh, 'P', degree)
        norm = Function(W1)
        unorm2 = ux.vector() * ux.vector()  \
               + uy.vector() * uy.vector()        
        wnorm = np.sqrt(unorm2.get_local())        
	norm.vector().set_local(wnorm)                                             
	norm.vector().apply('')
        chale = sqrt(dot(dot(uk,mesh_metric(W.mesh())),uk)/(dot(uk,uk)+DOLFIN_EPS))
        invchale = conditional(lt(chale,1E-4),1.0E12,1.0/chale)
        return norm*invchale

def tau(wk,dt,mu,rho):
        norminvcha = advnorm(wk)
	h = CellDiameter(W.mesh())
        freq1 = Constant(C1)*mu/(h*h)                    # Stokes term (using h=vol^(1/3))
        freq2 = Constant(C2)*rho*norminvcha              # Convective term (using h= h_stream)
        freqt = Constant(C4)*rho/dt              # Convective term (using h= h_stream)
        freto = freq1 + freq2 + freqt                           # Total frequency
        timom = conditional(lt(freto,1E-4),1.0E12,1.0/freto)
        tidiv = C3*h*h/timom
        return as_matrix(((timom, 0, 0),(0,timom, 0),(0, 0, tidiv)))	

def Subscales(w,wk,wt,dt,mu,rho):
        tau_ = tau(wk,dt,mu,rho) #Must use a Picard's previous velocity field, otherwise, Tau is super-nonlinear
        return tau_*(-L(w, w, mu, rho, 1.0, 0))

def VarForm(w, wk, wt, w_, dt, mu, rho, stab='VMS'):
	(u, p) = (as_vector ((w[0], w[1])) , w[2])
	(v, q) = (as_vector ((w_[0], w_[1])), w_[2])
 	delta = 1.0
	if(stab=='False'):
		delta = 0.0
	F_G = (mu*inner(grad(u), grad(v)) + inner(grad(p) + rho*dot(grad(u),u), v) + div(u)*q)*dx 
	F_stab = delta*inner(Subscales(w,wk,wt,dt,mu,rho), -L(w_, w, mu, rho, 1.0, 0))*dx
	F = F_G + F_stab
	return F

def L(w, w2, mu, rho, st_p, st_d):
	(u, p) = (as_vector ((w[0], w[1])) , w[2])
	(u2 , p2) = (as_vector ((w2[0], w2[1])), w2[2])
	Au = rho*grad(u2)*u + Constant(st_p)*grad(p) - Constant(st_d)*(mu*div(grad(u)))  
        Ap = div(u)
	Aui = [Au[i] for i in range(0, 2)]
	return  as_vector(Aui + [Ap])

def solve_unsteady_navier_stokes(W, mu, rho, bcs, T, dt, theta):
    """Solver unsteady Navier-Stokes and write results
    to file"""

    # Current, Non-linearity and old solution
    w = Function(W)
    u, p = split(w)
    dw = TrialFunction(W)

    wk = Function(W)
    w0 = Function(W)

    w_old = Function(W)
    u_old, p_old = split(w_old)

    w_oold = Function(W)
    u_oold, p_oold = split(w_old)

    # Define variational forms
    z = TestFunction(W)
    (v, q) = split(z)

    Ft= Constant(1/dt)*Constant(theta[0])*rho*dot(u, v)*dx 
    Ftt= Constant(1/dt)*Constant(theta[1])*rho*dot(u_old, v)*dx 
    Fttt= Constant(1/dt)*Constant(theta[2])*rho*dot(u_oold, v)*dx 
    FL = VarForm(w, w0, w_old, z, dt, mu, rho, stab='VMS')

    F = Ft + Ftt + Fttt + FL 

    # Solve the problem
    J = derivative(F, w, dw)

    ufile = File("%s.results/velocity.pvd" % (fileName))
    pfile = File("%s.results/pressure.pvd" % (fileName))

    ic = InitialCondition(degree = 0)
    wk.assign(interpolate(ic,W))
    w.assign(interpolate(ic,W))
    w_old.assign(interpolate(ic,W))
    w_oold.assign(interpolate(ic,W))

    tol = 1e-20     	# tolerance
    maxiter = 5 	# iteration limit		
    w_inc = Function(W)
    
    # Perform time-stepping
    t = 0
    while t < T:
	print 'time=%g' % (t)
        w_oold.vector()[:] = w_old.vector()
        w_old.vector()[:] = w.vector()
        wk.vector()[:] = w.vector()

        Ftt= Constant(1/dt)*Constant(theta[1])*rho*dot(u_old, v)*dx 
        Fttt= Constant(1/dt)*Constant(theta[2])*rho*dot(u_oold, v)*dx 

        L2_error = 1.0  # error measure ||u-u_k||
        it = 0          # iteration counter
        while L2_error > tol and it < maxiter:
	
	    it += 1				
            Ft= Constant(1/dt)*Constant(theta[0])*rho*dot(u, v)*dx 
            FL = VarForm(w, wk, w_old, z, dt, mu, rho, stab='VMS')
            F = Ft + Ftt + Fttt + FL 
            A, b = assemble_system(J, -F, bcs)
            solve(A, w_inc.vector(), b)     # Determine step direction
	    L2_error = assemble(((w_inc-wk)**2)*dx)
	    print 'it=%d: L2-error=%g' % (it, L2_error)
            wk.assign(w_inc)
	
	    if it == maxiter: print 'Solver did not converge!'

        w.assign(w_inc)
        u, p = w.split()
        u.rename("v", "velocity") ; ufile << u
        p.rename("p", "pressure") ; pfile << p
        t += dt

if __name__ == "__main__":

    fileName = os.path.splitext(__file__)[0]

    #Stabilization constants    
    C1 = 2
    C2 = 4
    C3 = 1
    C4 = 0.5

    """Solve unsteady Navier-Stokes to resolve
    Karman vortex street and save to file"""

    # Problem data
    u_in = Expression(("4.0*U*x[1]*(0.41 - x[1])/(0.41*0.41)", "0.0"),
                      degree=2, U=1)
    rho = Constant(1.0)
    mu = Constant(0.001)
    T = 1

    # Discretization parameters
    N_circle = 100
    N_bulk = 200
    theta = [3/2,-2,1/2] 
    dt = 0.0005

    # Prepare function space, BCs and measure on circle
    W, bcs, ds_circle = build_space(N_circle, N_bulk, u_in)

    # Solve unsteady Navier-Stokes
    solve_unsteady_navier_stokes(W, mu, rho, bcs, T, dt, theta)
