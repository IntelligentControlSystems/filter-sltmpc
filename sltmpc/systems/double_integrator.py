from ampyc.systems import SystemBase

class DoubleIntegrator(SystemBase):
    '''
    Implements a double integrator of the form:
    .. math::
        x_{k+1} = A x_k + B u_k + w_k

    where :math:`x` is the state, :math:`u` is the input, and :math:`w` is a disturbance.
    '''
    
    def update_params(self, params):
        super().update_params(params)
        assert params.A.shape == (self.n, self.n), 'A must have shape (n,n)'
        assert params.B.shape == (self.n, self.m), 'B must have shape (n,m)'
        self.A = params.A
        self.B = params.B

    def f(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional
        return self.A @ x.reshape(self.n, 1) + self.B @ u.reshape(self.m, 1)

    def h(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        return x.reshape(self.n, 1)