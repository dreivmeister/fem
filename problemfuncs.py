from abc import ABC, abstractmethod


# needs to be reimplemented for each problem
class ProblemFunctions(ABC):
    # represents the rhs of pde
    @abstractmethod
    def f(self):
        pass
    
    # gives neumann boundary conditions
    @abstractmethod
    def g(self):
        pass
    
    # gives dirichlet boundary conditions
    def u_d(self):
        pass