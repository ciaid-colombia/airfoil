	

from __future__ import print_function
from ufl import *
from fenics import *
from mshr import *
import os
from os import listdir
import dolfin
import numpy as np
from numpy import array, zeros, ones, any, arange, isnan, cos, arccos
import ctypes, ctypes.util, numpy, scipy.sparse, scipy.sparse.linalg, collections
import matplotlib.pyplot as plt
import time
import sys
import meshio
import re

###############################################################################
###############################################################################
############################ Own functions ####################################

def mesh(lx,ly, Nx,Ny):
	 m = UnitSquareMesh(Nx, Ny)
	 x = m.coordinates()

	 #Refine near cylinder
	 x[:,0] = (x[:,0]-0.5)*2.
	 x[:,0] = (pi-arccos(x[:,0]))/pi   

	 #Refine on vertical centerline
	 x[:,1] = (x[:,1]-0.5)*2.
	 x[:,1] = (pi-arccos(x[:,1]))/pi   

	 #Scale
	 x[:,0] = x[:,0]*lx
	 x[:,1] = x[:,1]*ly

	 return m

def Refiner(p,Radius,Scale,m):

	# Mark cells for refinement
	cell_markers = MeshFunction("bool", m, 2)
	cell_markers.set_all(False)
	for c in cells(m):
		if c.midpoint().distance(p) < Radius*Scale:
			cell_markers[c] = True
	m = refine(m, cell_markers)

	return m

# def meshRectangle(lx,ly, N,Center_x,Center_y,Radius):
def meshRectangle(m,Center_x,Center_y,Radius):
	
	# domain = Rectangle(Point(0, 0), Point(lx, ly))

	# m = generate_mesh(domain, N)

	p = Point(Center_x, Center_y)

	m = Refiner(p,Radius,2.0,m)
	# m = Refiner(p,Radius,8.0,m)
	# m = Refiner(p,Radius,5.0,m)
	# m = Refiner(p,Radius,4.0,m)
	# m = Refiner(p,Radius,2.5,m)
	# m = Refiner(p,Radius,2.0,m)
	# m = Refiner(p,Radius,1.8,m)

	return m


def RectangleRefiner(xMin,xMax,yMin,yMax,m):
	
	# Mark cells for refinement
	cell_markers = MeshFunction("bool", m, 2)
	cell_markers.set_all(False)
	# Index
	xind = [0, 2, 4]
	yind = [1, 3, 5]
	for c in cells(m):
		CoordVertex = (c.get_vertex_coordinates())
		Coord = np.asarray(CoordVertex)
		MeanCoord = [np.mean(Coord[xind]),np.mean(Coord[yind])]
		if MeanCoord[0] > xMin and MeanCoord[0] < xMax and \
			MeanCoord[1] > yMin and MeanCoord[1] < yMax:
			cell_markers[c] = True
	m = refine(m, cell_markers)

	return m


def boundaryfilter(m):

	# Mark boundary adjacent cells
	boundary_cells = []
	for cell in cells(m):
		facetcells = facets(cell)
		for myFacet in facetcells:
			if myFacet.exterior() == True: 
				boundary_cells.append(cell)
	
	cell_domains = MeshFunction('size_t', m, m.topology().dim())
	cell_domains.set_all(0)
	for myCell in boundary_cells:
		cell_domains[myCell] = 1
	
	return Mesh(SubMesh(m, cell_domains, 0))

###############################################################################
###############################################################################



###############################################################################

#Computation time
start = time.time()
iterTime = start


MeshFile = 'NACA0012.dat.msh'

# get file name
fileName = os.path.splitext(__file__)[0]

folder_path = "%s.results/" % (fileName)

# VTK
vtkfile_u = File('%s.results/Velocity.pvd' % (fileName))
vtkfile_p = File('%s.results/Pressure.pvd' % (fileName))
vtkfile_s = File('%s.results/Solid.pvd' % (fileName))
vtkfile_Test = File('%s.results/Test.pvd' % (fileName))

TransientON = True
TimeMethod = 'BDF2'

T = 5.0          	# final time
dt = 0.05 	 	# time step size
t_steady = 2. 	        # Time at which the inlet speed of the fluid gets steady, in s

num_steps = int(T/dt) 	# number of time steps
if not TransientON: 
	num_steps = 1

# Lists to save for postprocessing of the data
time_list = [0.]
Lift_list = [0.]
Drag_list = [0.]

#Geometric parameters and mesh
BoxHeigth = 0.41
BoxLength = 2.2
Center_x = 0.2
Center_y = 0.2
Radius = 0.05

#Fluid's parameters


Uinf = 3.0
Alpha= 5.0
Alpha = Alpha*np.pi/180


Umean = Uinf
mu = 0.00001        	# dynamic viscosity
rho = 1.0         # density
Beta = 0.0			# 0 for Picard - 1 for Newton-Raphson 
f  = Constant((0, 0))
f3  = Constant(0.0)
k  = Constant(dt)
nu = Constant(mu/rho)
mu = Constant(mu)
rho = Constant(rho)
Beta = Constant(Beta)

chord = 1.0
Re = float(rho*Uinf*chord / mu)
print('Re:', "{:.2e}".format(Re))

#VMS
C1 = Constant(4.0)
C2 = Constant(2.0)
C3 = Constant(1.0)
C4 = Constant(1.0)

# Iterations
tol = 1.0E-5        # tolerance
maxiter = 10        # max no of iterations allowed


############################################################################
############################ Boundary definition ###########################

class OutBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary 
 
class WallsBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and (near(x[1], 0.0) or near(x[1], BoxHeigth))
	
class InflowBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and near(x[0], 0.0)

class OutflowBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and near(x[0], BoxLength)


############################ Initial conditions ############################

class InitialVelGradient(UserExpression):
	def eval(self, values, x):
		values[0] = 0.0
		values[1] = 0.0
		values[2] = 0.0
		values[3] = 0.0
	def value_shape(self):
		return (2,2)

class InitialPGradient(UserExpression):
	def eval(self, values, x):
		values[0] = 0.0
		values[1] = 0.0
	def value_shape(self):
		return (2,)

class InitialCondition(UserExpression):
	def eval(self, values, x):
		values[0] = Umean
		values[1] = 0.0
		values[2] = 0.0
	def value_shape(self):
		return (3,)

ic=InitialCondition(degree =2)
iug=InitialVelGradient(degree =2)
ipg=InitialPGradient(degree =2)


############################################################################
######################## Tesnor and Matrix operations ######################

# Define symmetric gradient
def epsilon(u):
	 return sym(nabla_grad(u))

def convective(u1,u2):
	return(dot(u1, nabla_grad(u2)))

def tensor_jump(v,n):
	return outer(v('+'),n('+')) + outer(v('-'),n('-'))

############################################################################
################################## Matrix ##################################

# Galerkin Termns: 
# 2 * mu * D(v) : D(u) 
# - ∇·v * p 
# + q * ∇·u
def MAT_Gh(u, v, p, q, mu):  
	return 2.0*mu*inner(epsilon(u), epsilon(v))*dx  \
				- p*div(v)*dx \
				+ q*div(u)*dx

# Stabilization Terms:
# τ1 * ( (un ∇)·v , (un ∇)·u )
# τ1 * ( ∇q, ∇p ) 
# τ2 * ( ∇·v , ∇·u )
def MAT_VMSh(u, u_n, v, p, q, rho, Tau1, Tau2):
	return  Tau1 * inner( rho * convective(u_n,v) + nabla_grad(q), 
								 rho * convective(u_n,u) + nabla_grad(p) )*dx \
			+ Tau2 * div(v)*div(u)*dx

# NonLinear Terms:
#  	(un ∇)·u  
# β * (u ∇)·un  
def MAT_NLh(u, u_n, v, Beta, rho):
	return Beta * inner( rho * convective(u,u_n), v )*dx \
					+ inner( rho * convective(u_n,u), v )*dx

# Transient Term:
# ρ * 1/dt * (u,v)
# ρ * 1/dt * ((un ∇)·v,u)
# 	   1/dt * (∇q     ,u)
def MAT_Th(u,u_n,v,rho,k,Tau1):
	MAT = rho / k *        inner( v                    , u )*dx \
		 + rho / k * Tau1 * inner( rho*convective(u_n,v), u )*dx \
		 + rho / k * Tau1 * inner( nabla_grad(q)        , u )*dx 
	return MAT


############################################################################
################################## RHS  ####################################

# Galerkin Termns: 
# v * f
def VECT_Gh(v, f):  
	return inner( v,f )*dx

# Stabilization Terms:
# τ1 * ( ρ(un ∇)·v , f )
# τ1 * ( ∇q, f ) 
def VECT_VMSh(u_n,v,q,Tau1,f):
	return Tau1 * rho * inner( convective(u_n,v)  , f)*dx \
			+ Tau1 * 		  inner( nabla_grad(q)      , f)*dx \
			+ Tau2 * div(v) * f3 *dx 

# NonLinear Terms:
# (β * (un ∇)·un ,v) 
def VECT_NLh(u_n, v, Beta, rho):
	return Beta*inner( rho * convective(u_n,u_n), v)*dx 

# Transient Term:
# ρ * 1/dt * (u0,v)
def VECT_Th(u_0,u_n,v,rho,k,Tau1):
	VECT = rho / k *        inner( v                    , u_0 )*dx \
			+ rho / k * Tau1 * inner( rho*convective(u_n,v), u_0 )*dx \
			+ rho / k * Tau1 * inner( nabla_grad(q)        , u_0 )*dx 
	return VECT

############################################################################
############################################################################


Length = float(chord*5.0)
Width = float(chord*5.0)
# h0 = float(1.0)

center = Point(-chord, Width/2.0)
center1 = Point(0, Width/2.0)

center2 = Point(-chord/2.0, Width/2.0)


# os.system('meshio-convert '+ mesh_from_file +' mesh.xdmf')
os.system('dolfin-convert '+ MeshFile +' mesh.xml')


mesh = Mesh('mesh.xml')
markers = MeshFunction("size_t", mesh, 'mesh_physical_region.xml')
bndry = MeshFunction('size_t', mesh, 'mesh_facet_region.xml')

File("%s.results/Mesh.pvd" % (fileName)) << mesh


print('Number of nodes:', mesh.num_vertices() )
print('Number of elements:', mesh.num_cells() )



############################################################################
########################### Build function space ###########################

P3 = TensorElement("P", mesh.ufl_cell(), 1)
P2 = VectorElement("P", mesh.ufl_cell(), 1)
P1 = FiniteElement("P", mesh.ufl_cell(), 1)
TH = MixedElement([P2, P1])
W = FunctionSpace(mesh, TH)

Fsig = FunctionSpace(mesh, P3)

######################################################################################
############################### Boundary conditions ##################################
######################################################################################

# Fluid Boundaries
UinfX = Uinf*np.cos(Alpha)
UinfY = Uinf*np.sin(Alpha)
inflow_profile = Expression(("Ux", "Uy"),
										degree=2, Ux=UinfX, Uy=UinfY)

# Prepare Dirichlet boundary conditions
bc_UpperWall = DirichletBC(W.sub(1), 0., bndry, 1009)
bc_LowerWall = DirichletBC(W.sub(0), inflow_profile, bndry, 1010)
bc_body = DirichletBC(W.sub(0), (0, 0), bndry, 1012)
bc_in = DirichletBC(W.sub(0), inflow_profile, bndry, 1008)
bc_out = DirichletBC(W.sub(1), 0., bndry, 1011)
bcs = [bc_body, bc_UpperWall, bc_LowerWall, bc_in, bc_out]

# Prepare surface measure on cylinder
# ds_body = Measure("ds", subdomain_data=bndry, subdomain_id=5)
ds_body = Measure("ds", subdomain_data=bndry, subdomain_id=1012)


File("%s.results/Mesh.pvd" % (fileName)) << bndry




######################################################################################

# Define functions for solutions at previous and current time steps

# Non-linear and old solutions
U_n = Function(W)
U_0 = Function(W)
U_00 = Function(W)
U_check = Function(W)

S0 = FunctionSpace(mesh, "Lagrange", 1)
s0 = Function(S0)

# Define trial and test functions
(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)

# Get sub-functions
u_n, p_n = split(U_n)
u_0, p_0 = split(U_0)
u_00, p_00 = split(U_00)

u_check, p_check = split(U_check)



WRITExdmffile_u = XDMFFile("%s.results/velocity.xdmf" % (fileName))
WRITExdmffile_p = XDMFFile("%s.results/pressure.xdmf" % (fileName))


V2 = W.sub(0).collapse()
V1 = FunctionSpace(mesh, P1) 

VxdmfFile = folder_path + 'velocity.xdmf'
PxdmfFile = folder_path + 'pressure.xdmf'


# for file_name in listdir(folder_path):
# 	if file_name.endswith('.xdmf') or file_name.endswith('.h5'):
# 		os.remove(folder_path + file_name)

stepCount = 0
if os.path.isfile(VxdmfFile):

	READxdmffile_u = XDMFFile("%s.results/velocity.xdmf" % (fileName))
	READxdmffile_p = XDMFFile("%s.results/pressure.xdmf" % (fileName))

	uf0 = Function(V2)
	uf00 = Function(V2)
	pf1 = Function(V1)

	with open("%s.results/velocity.xdmf" % (fileName)) as file:
		line_list = list(file)
		# line_list.reverse()

		for line in line_list:
				if line.find('Time') != -1:
					# do something
					x = re.findall('[0-9]+', line)
					stepCount = float(x[0])+1.0
					stepCount = float(stepCount)
					# print(stepCount)
					# stepCount = stepCount + 1
					# print(stepCount)

	READxdmffile_u.read_checkpoint(uf0,"velocity",-1) 
	READxdmffile_u.read_checkpoint(uf00,"velocity",-2) 

	assign(U_0.sub(0), uf0)
	assign(U_n.sub(0), uf0)
	assign(U_00.sub(0), uf00)


W = FunctionSpace(mesh, TH)

V0 = VectorFunctionSpace(mesh, "Lagrange", 1)

u0_n = Function(V0)
u0 = Function(V0)

# Define facet normal and mesh size
n  = FacetNormal(mesh)
h = 2.0*Circumradius(mesh)

# For linear system
U = Function(W)

############################################################################
############# Calculate mesh dependent algorithmic parameters ##############
############################################################################

vnorm = sqrt(dot(u_n, u_n))

# VMS Stabilization terms 
Tau1 = C1*mu/h**2.0 + C2*rho*vnorm/h
Tau1 = 1.0 / Tau1
Tau2 = C3*mu + C4*rho*vnorm*h

############################################################################
############################################################################

a = MAT_Gh(u, v, p, q, mu) \
	+ MAT_NLh(u, u_n, v, Beta, rho) \
	+ MAT_VMSh(u, u_n, v, p, q, rho, Tau1, Tau2)

L = VECT_Gh(v, f) \
	+ VECT_NLh(u_n, v, Beta, rho) \
	+ VECT_VMSh(u_n, v, q, Tau1, f) 

if TransientON:
	if TimeMethod == 'BDF':
		a = a + MAT_Th(u,u_n,v,rho,k,Tau1)
		L = L + VECT_Th(u_0,u_n,v,rho,k,Tau1)
	elif TimeMethod == 'BDF2':
		a = a + 3.0/2.0*MAT_Th(u,u_n,v,rho,k,Tau1) 
		L = L + 4.0/2.0*VECT_Th(u_0,u_n,v,rho,k,Tau1) \
					- 1.0/2.0*VECT_Th(u_00,u_n,v,rho,k,Tau1)	



lift = 0.
drag = 0.

t = 0.0

for i in range(num_steps):
	
	print('TimeStep = ', i, 'Time = ', t)
	t += dt 
	if t > t_steady: inflow_profile.time = t_steady
	else: inflow_profile.time = t


	# Update solution time step
	if TimeMethod=='BDF2' and i>1:
		U_00.assign(U_0)
	U_0.assign(U_n)
			
	L2_error = 1.0      # error measure ||u-u_k||
	it = 0            	# iteration counter


	while L2_error > tol and it < maxiter:
		it += 1

		iterTime0 = iterTime
		iterTime = time.time()
		deltaIterTime = iterTime - iterTime0
		TotalTime = iterTime - start

		# Assemble matrices and vector
		A = assemble(a)
		b = assemble(L)

		bc_UpperWall.apply(A,b)
		bc_LowerWall.apply(A,b)
		bc_body.apply(A,b)
		bc_in.apply(A,b)
		bc_out.apply(A,b)

		#Solve linear system 
		solve(A, U.vector(), b)


		# Calculate error
		u0_n.assign( project(U_n.sub(0),	 	V0))
		u0.assign( project(U.sub(0),	 	V0))
		s0.assign( project(U.sub(1),	 	S0))
		error = (((u0-u0_n)**2)*dx)
		L2_error = sqrt(abs(assemble(error)))
		print('Iteration = ', it, 'L2-norm = ', L2_error, 
					'IterTime =', deltaIterTime, 'TotalIterTime =', TotalTime)

		# Update solution nonlinear step
		U_n.assign(U)

		u, p = split(U) 



	# Compute fluid stress on current configuration
	d_ = 1./2.*(nabla_grad(u_n) + nabla_grad(u_n).T)
	I = Identity(mesh.geometry().dim())
	# d_ = 2.0*mu*sym(grad(u))

	tau_ =  -p_n*I + 2.*mu*d_
	# tau_ = -p*I + 2.0*nu*sym(grad(u))

	# sigma_f = project(-p*Identity(u.geometric_dimension())+2.*mu*d_, Fsig, solver_type="mumps", \
	# 	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"} )
	
	# Compute the traction vector over the solid's surface (Updated Lagrange Configuration)
	t_s_hat = dot(n,tau_)
	# t_s_hat = dot(n,sigma_f)
	
	#Dynamic pressure 
	qd = 0.5*rho*Uinf*Uinf*chord
	# print(float(qd))


	#Translate Solid's domain
	Drag =  t_s_hat[0]*np.cos(Alpha) + t_s_hat[1]*np.sin(Alpha)
	Lift = -t_s_hat[0]*np.sin(Alpha) + t_s_hat[1]*np.cos(Alpha)
	# Drag =  t_s_hat[0]
	# Lift =  t_s_hat[1]

	Cd = (Drag/qd) * ds_body
	Cl = (Lift/qd) * ds_body
	DragCoeff = assemble(Cd)
	LiftCoeff = assemble(Cl)

	# print(drag/dp1,lift/dp1)
	print('Cd:', "{:.4f}".format(DragCoeff))
	print('Cl:', "{:.4f}".format(LiftCoeff))
	# print(DragCoeff,LiftCoeff)

	time_list.append(t)
	Drag_list.append(DragCoeff)
	Lift_list.append(LiftCoeff)
	
	if (i % 5 == 0 ):
		#Output fields
		# u0.rename("velocity", "velocity") ;vtkfile_u << u0
		# s0.rename("pressure", "pressure") ;vtkfile_p << s0

		u11, p11 = U.split(True)

		j = i + stepCount

		WRITExdmffile_u.write_checkpoint(u11, "velocity", j, XDMFFile.Encoding.HDF5, True)  #appending to file
		WRITExdmffile_p.write_checkpoint(p11, "pressure", j, XDMFFile.Encoding.HDF5, True)  #appending to file



WRITExdmffile_u.close()
WRITExdmffile_p.close()

numpy.savetxt("%s.results/Forces.csv" % (fileName), numpy.c_[time_list, Drag_list, Lift_list], delimiter=",")
