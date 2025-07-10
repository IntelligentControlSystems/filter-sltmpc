from ampyc.systems import SystemBase

class UncertainDoubleIntegrator(SystemBase):
    '''
    Implements a double integrator of the form:
    .. math::
        x_{k+1} = (A + dA) x_k + (B + dB) u_k + w_k

    where :math:`x` is the state, :math:`u` is the input, :math:`w` is a disturbance,
    and :math:`dA` and :math:`dB` are model uncertainties.
    '''
    
    def update_params(self, params):
        super().update_params(params)
        assert params.A.shape == (self.n, self.n), 'A must have shape (n,n)'
        assert params.B.shape == (self.n, self.m), 'B must have shape (n,m)'
        self.A = params.A
        self.B = params.B

        self.Delta_A = params.Delta_A
        self.Delta_B = params.Delta_B

        self.dA_gen = params.Delta_A_gen
        self.dB_gen = params.Delta_B_gen

    def f(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        self._check_u_shape(u)  # make sure u is m dimensional

        dA = self.dA_gen.generate()
        dB = self.dB_gen.generate()

        return (self.A + dA) @ x.reshape(self.n, 1) + (self.B + dB) @ u.reshape(self.m, 1)

    def h(self, x, u):
        self._check_x_shape(x)  # make sure x is n dimensional
        return x.reshape(self.n, 1)