##http://airfoiltools.com/airfoil/naca4digit


from dolfin import *
import ufl
import time
import os
import mshr

# get file name
fileName = os.path.splitext(__file__)[0]
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 8
#parameters["form_compiler"]["quadrature_rule"] = 'auto'

comm = mpi_comm_world()
rank = MPI.rank(comm)
set_log_level(INFO if rank==0 else INFO+1)
ufl.set_level(ufl.INFO if rank==0 else ufl.INFO+1)
parameters["std_out_all_processes"] = False;
info_blue(dolfin.__version__)

# Time stepping parameters
dt = 0.01    
t_end = 1.0 
theta=Constant(0.5)   # theta schema
k=Constant(1.0/dt)
g=Constant((0.0,-1.0))

## Create mesh
channel = mshr.Rectangle(Point(-1.0, -0.5),Point(2, 0.5)) 
# Create list of polygonal domain vertices for the car
domain_vertices = [Point(1, 0  ),
 Point( 1.000167,  0.001249 ),
 Point( 0.998653,  0.001668 ),
 Point( 0.994122,  0.002919 ),
 Point( 0.986596,  0.004976 ),
 Point( 0.976117,  0.007801 ),
 Point( 0.962742,  0.011341 ),
 Point( 0.946545,  0.015531 ),
 Point( 0.927615,  0.020294 ),
 Point( 0.906059,  0.025547 ),
 Point( 0.881998,  0.031197 ),
 Point( 0.855570,  0.037149 ),
 Point( 0.826928,  0.043305 ),
 Point( 0.796239,  0.049564 ),
 Point( 0.763684,  0.055826 ),
 Point( 0.729457,  0.061992 ),
 Point( 0.693763,  0.067967 ),
 Point( 0.656819,  0.073655 ),
 Point( 0.618851,  0.078967 ),
 Point( 0.580092,  0.083817 ),
 Point( 0.540785,  0.088125 ),
 Point( 0.501176,  0.091816 ),
 Point( 0.461516,  0.094825 ),
 Point( 0.422059,  0.097095 ),
 Point( 0.382787,  0.098537 ),
 Point( 0.343868,  0.098810 ),
 Point( 0.305921,  0.097852 ),
 Point( 0.269212,  0.095696 ),
 Point( 0.234002,  0.092400 ),
 Point( 0.200538,  0.088046 ),
 Point( 0.169056,  0.082736 ),
 Point( 0.139770,  0.076589 ),
 Point( 0.112880,  0.069743 ),
 Point( 0.088560,  0.062343 ),
 Point( 0.066964,  0.054540 ),
 Point( 0.048221,  0.046485 ),
 Point( 0.032437,  0.038325 ),
 Point( 0.019693,  0.030193 ),
 Point( 0.010051,  0.022209 ),
 Point( 0.003547,  0.014471 ),
 Point( 0.000198,  0.007052 ),
 Point( 0.000000,  0.000000 ),
 Point( 0.002885, -0.006437 ),
 Point( 0.008765, -0.012027 ),
 Point( 0.017579, -0.016779 ),
 Point( 0.029250, -0.020704 ),
 Point( 0.043684, -0.023825 ),
 Point( 0.060773, -0.026172 ),
 Point( 0.080396, -0.027782 ),
 Point( 0.102423, -0.028706 ),
 Point( 0.126714, -0.029000 ),
 Point( 0.153123, -0.028734 ),
 Point( 0.181496, -0.027986 ),
 Point( 0.211676, -0.026843 ),
 Point( 0.243500, -0.025401 ),
 Point( 0.276797, -0.023760 ),
 Point( 0.311396, -0.022023 ),
 Point( 0.347115, -0.020295 ),
 Point( 0.383767, -0.018677 ),
 Point( 0.421506, -0.017200 ),
 Point( 0.460025, -0.015646 ),
 Point( 0.498824, -0.014038 ),
 Point( 0.537674, -0.012432 ),
 Point( 0.576342, -0.010875 ),
 Point( 0.614595, -0.009404 ),
 Point( 0.652198, -0.008049 ),
 Point( 0.688920, -0.006829 ),
 Point( 0.724534, -0.005754 ),
 Point( 0.758815, -0.004826 ),
 Point( 0.791547, -0.004042 ),
 Point( 0.822520, -0.003392 ),
 Point( 0.851537, -0.002863 ),
 Point( 0.878408, -0.002440 ),
 Point( 0.902958, -0.002109 ),
 Point( 0.925025, -0.001853 ),
 Point( 0.944461, -0.001659 ),
 Point( 0.961137, -0.001514 ),
 Point( 0.974939, -0.001409 ),
 Point( 0.985774, -0.001335 ),
 Point( 0.993567, -0.001286 ),
 Point( 0.998264, -0.001258 ),
 Point( 0.999833, -0.001249 ),
 Point( 1 , 0 )]


blade = mshr.Polygon(domain_vertices);

domain = channel - blade
mesh = mshr.generate_mesh(domain, 50)



class InitialCondition(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
    def value_shape(self):
        return (3,)

ic=InitialCondition(degree = 2)

class Boundary_NACA(SubDomain):
   def inside(self, x, on_boundary):
      tol = 1E-14
      return on_boundary and  x[0]>-0.05 and x[0]<1.05 and x[1]>-0.1 and x[1]<0.1


boundary_N = Boundary_NACA()
domainBoundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
domainBoundaries.set_all(0)
ds = Measure("ds")[domainBoundaries]

nor = 3

for i in range(nor):
    edge_markers = MeshFunction("bool", mesh, mesh.topology().dim() - 1, False)
    boundary_N.mark(edge_markers, True)
    mesh = refine(mesh, edge_markers)


# Define function spaces
V = VectorElement("CG",mesh.ufl_cell(), 2)
P = FiniteElement("CG",mesh.ufl_cell(), 1)
VP = MixedElement([V, P])
W = FunctionSpace(mesh,VP)


# Define unknown and test function(s)
w = Function(W)
w0 = Function(W)
(v_, p_) = TestFunctions(W)
(v,p)=split(w)
(v0,p0)=split(w0)


bcs = list()
bcs.append( DirichletBC(W.sub(0), Constant((1.0, 0.0)), "near(x[0],-1.0)") )
bcs.append( DirichletBC(W.sub(0), Constant((1.0, 0.0)), "near(x[1],-0.5) || near(x[1],0.5)") )
bcs.append( DirichletBC(W.sub(1), Constant(0.0), "near(x[0],2.0)") )
bcs.append( DirichletBC(W.sub(0), Constant((0.0, 0.0)), "x[0]>-0.05 && x[0]<1.05  && x[1]>-0.1 && x[1]<0.1 && on_boundary") )


rho=1e1
mu=1e-3
def sigma(v,p):
    return(-p*I + mu*(grad(v)+grad(v).T))

def EQ(v,p,v_,p_):
    F =  rho*inner(grad(v)*v, v_)*dx - rho*inner(g,v_)*dx + inner(sigma(v,p),grad(v_))*dx 
    return(F)

n = FacetNormal(mesh)
I = Identity(V.cell().geometric_dimension())    # Identity tensor
h = CellDiameter(mesh)

F=k*0.5*(theta*rho)*inner(v-v0,v_)*dx + theta*EQ(v,p,v_,p_) + (Constant(1.0)-theta)*EQ(v0,p,v_,p_) + div(v)*p_*dx 

J = derivative(F, w)

#ffc_options = {"quadrature_degree": 4, "optimize": True, "eliminate_zeros": False}
ffc_options = {"quadrature_degree": 4, "optimize": True}
problem=NonlinearVariationalProblem(F,w,bcs,J,ffc_options)
solver=NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'umfpack'
prm['newton_solver']['lu_solver']['report'] = False
prm['newton_solver']['lu_solver']['same_nonzero_pattern']=True
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-8
prm['newton_solver']['maximum_iterations'] = 30
prm['newton_solver']['report'] = True
#prm['newton_solver']['error_on_nonconvergence'] = False


w.assign(interpolate(ic,W))
w0.assign(interpolate(ic,W))

(v,p) = w.split()
(v0,p0) = w0.split()


# Create files for storing solution
vfile = File("%s.results/velocity.pvd" % (fileName))
pfile = File("%s.results/pressure.pvd" % (fileName))

v.rename("v", "velocity") ; vfile << v
p.rename("p", "pressure") ; pfile << p

# Time-stepping
t = dt
while t < t_end:

   print("t =%d", t)

   begin("Solving transport...")
   solver.solve()
   end()

   (v,p)=w.split(True)
   v.rename("v", "velocity") ; vfile << v
   p.rename("p", "pressure") ; pfile << p

   w0.assign(w)
   t += dt  # t:=t+1

   # Report drag and lift
   force = dot(sigma(v,p), n)
   D = (force[0]/0.002)*ds(5)
   L = (force[1]/0.002)*ds(5)
   #drag = assemble(D)
   #lift = assemble(L)
   #info("drag= %e    lift= %e" % (drag , lift))
