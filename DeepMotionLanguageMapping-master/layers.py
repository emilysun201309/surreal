from keras.layers import GRU
from keras import initializations
import keras.backend as K

class GRULN(GRU):
    '''Gated Recurrent Unit with Layer Normalization
    Current impelemtation only works with consume_less = 'gpu' which is already
    set.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        ...: see GRU documentation for all other arguments.

    # References
        -[Layer Normalization](https://arxiv.org/abs/1607.06450)
    '''
    def __init__(self, output_dim, **kwargs):
        if 'consume_less' in kwargs:
            del kwargs['consume_less']
        super(GRULN, self).__init__(output_dim, consume_less='gpu', **kwargs)
        self.gamma_init = initializations.get('one')
        self.beta_init = initializations.get('zero')
        self.epsilon = 1e-5

    def build(self, input_shape):
        super(GRULN, self).build(input_shape)
        shape = (self.output_dim,)
        shape1 = (2*self.output_dim,)
        # LN is applied in 4 inputs/outputs (fields) of the cell
        gammas = {}
        betas = {}
        # each location has its own BN
        for slc, shp in zip(['state_below', 'state_belowx', 'preact', 'preactx'], [shape1, shape, shape1, shape]):
            gammas[slc] = self.gamma_init(shp,
                                          name='{}_gamma_{}'.format(
                                              self.name, slc))
            betas[slc] = self.beta_init(shp,
                                        name='{}_beta_{}'.format(
                                            self.name, slc))

        self.gammas = gammas
        self.betas = betas

        self.trainable_weights += self.gammas.values() + self.betas.values()

    def ln(self, x, slc):
        # sample-wise normalization
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
        x_normed = (x - m) / (std + self.epsilon)
        x_normed = self.gammas[slc] * x_normed + self.betas[slc]
        return x_normed

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        matrix_x = K.dot(x * B_W[0], self.W) + self.b
        x_ = self.ln(matrix_x[:, : 2 * self.output_dim], 'state_below')
        xx_ = self.ln(matrix_x[:, 2 * self.output_dim:], 'state_belowx')
        matrix_inner = self.ln(K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim]), 'preact')

        x_z = x_[:, :self.output_dim]
        x_r = x_[:, self.output_dim: 2 * self.output_dim]
        inner_z = matrix_inner[:, :self.output_dim]
        inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

        z = self.inner_activation(x_z + inner_z)
        r = self.inner_activation(x_r + inner_r)

        x_h = xx_
        inner_h = r * self.ln(K.dot(h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:]), 'preactx')
        hh = self.activation(x_h + inner_h)

        h = z * h_tm1 + (1 - z) * hh
        return h, [h]
