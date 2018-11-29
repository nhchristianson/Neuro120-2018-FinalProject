import numpy as np
from sklearn.model_selection import train_test_split

# Set seed
SEED = 123
np.random.seed(123)

#-------------------------------------------------------------------------------
# Global parameters
#-------------------------------------------------------------------------------
beta     = 5.0
n_input  = 2
n_output = 1
n_hidden = 2
n_total  = n_input + n_hidden + n_output
activation = 'sigmoid'

#-------------------------------------------------------------------------------
# Main class
#-------------------------------------------------------------------------------

class EqProp():
    def __init__(self, 
                 beta       = beta, 
                 activation = activation,
                 n_input    = n_input, 
                 n_output   = n_output, 
                 n_hidden   = n_hidden,
                 W_init     = None,
                 b_init     = None):
        self.beta       = beta
        self.activation = activation
        self.n_input    = n_input
        self.n_output   = n_output
        self.n_hidden   = n_hidden
        self.n_total    = n_input + n_hidden + n_output


        # Randomly initialize weights
        if W_init == None:
            W_init = np.random.normal(size = (n_total, n_total))
            np.fill_diagonal(W_init, 0)
            W_init = W_init + W_init.T
        if b_init == None:
            b_init = 0*np.random.normal(size = (n_total, 1))

        self.W_init = W_init
        self.W  = W_init
        self.b  = b_init

    def activation_function(self, x):
        if activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        else:
            raise Exception('Activation ' + activation + ' not implemented.')

    def der_activation_function(self, x):
        """
        Derivative of activation function
        """
        if activation == 'sigmoid':
            aux = self.activation_function(x)
            return aux*(1.0-aux)
        else:
            raise Exception('Activation ' + activation + ' not implemented.')

    def energy_function(self, u, W, b, beta=0, y_label = None):
        """
        Total energy function.

        Args:
            u (np.array): values of hidden units, shape (n_total, 1)
            W (np.array): matrix of weights, shape (n_total, n_total)
            b (np.array): biases, shape (n_total, 1)   
            beta (float): clamping factor
            y_label (np.array): desired output, shape (n_output, 1)
        """
        aux    = self.activation_function(u)
        W_zero_diag = W.copy()
        np.fill_diagonal(W_zero_diag, 0)

        term_1 = 0.5*(u**2.0).sum()
        term_2 = 0.5* ((aux.T).dot(W_zero_diag)).dot(aux)
        term_3 = (b*aux).sum()
        E = term_1-term_2-term_3
        C = 0
        if y_label is not None:
            y = u[n_total-n_output:,0].reshape(-1,1)
            C = 0.5*( (y-y_label)**2.0 ).sum()
            C = C.squeeze()

        return E.squeeze().squeeze() + beta*C

    def der_energy_function(self, u, W, b, beta=0, y_label = None):
        """
        Derivative of total energy function wrt the units.

        Args:
            u (np.array): values of hidden units, shape (n_total, 1)
            W (np.array): matrix of weights, shape (n_total, n_total)
            b (np.array): biases, shape (n_total, 1)   
            beta (float): clamping factor
            y_label (np.array): desired output, shape (n_output, 1)
        """
        rho_u    = self.activation_function(u)
        d_rho_u  = self.der_activation_function(u)
        W_zero_diag = W.copy()
        np.fill_diagonal(W_zero_diag, 0)

        term_1 = u
        term_2 = 0.5*d_rho_u*((W_zero_diag+W_zero_diag.T).dot(rho_u))
        term_3 = d_rho_u*b

        dE = term_1-term_2-term_3 
        dC = 0
        if y_label is not None:
            y = u[n_total-n_output:,0].reshape(-1,1)
            dC = (y-y_label)
            dC = np.vstack((
                            np.zeros((n_input+n_hidden, 1)),
                            dC))
        return dE + beta*dC


    def get_fixed_point(self, u_init, W, b, beta=0, y_label = None):
        u_init           = u_init.copy()
        x                = u_init[:n_input, 0].copy()
        u_init[:n_input, 0] = 0
        tol = 1e-5
        step_size = 0.1

        while True:
            dF_du  = self.der_energy_function(u_init, W, b, beta, y_label)
            dF_du[:n_input,0] = 0
            u_next = u_init - step_size*dF_du

            # print("err = ", ((u_next-u_init)**2.0).mean())
            if ((u_next-u_init)**2.0).mean() < tol:
                break

            u_init = u_next

        u_next[:n_input, 0] = x
        return u_next

    def update_params(self, W, b, u_0, u_beta, beta):
        rho_u_0    = self.activation_function(u_0)
        rho_u_beta = self.activation_function(u_beta)

        M_0        = rho_u_0.dot(rho_u_0.T)
        M_beta     = rho_u_beta.dot(rho_u_beta.T)

        delta_W = 0.5*(1.0/beta)*(M_beta - M_0)
        delta_b  = (1.0/beta)*(rho_u_beta - rho_u_0)

        # make sure no new connections are created
        delta_W[W == 0.] = 0.

        # print(rho_u_beta)
        # print(M_beta)
        # print(delta_W)
        # print("~~~")

        return W + delta_W, b + delta_b


    def predict(self, X):
        n_input = self.n_input
        n_total = self.n_total
        n_output = self.n_output
        W = self.W
        b = self.b

        N = X.shape[0]
        y = np.zeros(N)
        u_0  = np.zeros((self.n_total, 1))

        for n in range(N):
            x = X[n, :]
            u_0[:n_input,0] = x
            u_0  = self.get_fixed_point(u_0, W, b, beta = 0, y_label = None )
            y[n] = np.argmax(u_0[n_total-n_output:n_total,0].squeeze())
        return y

    def train_model(self, X_train, y_train, n_epochs = 1):
        beta = self.beta
        n_input = self.n_input
        n_total = self.n_total

        u_0    = np.zeros((n_total, 1))
        u_beta = np.zeros((n_total, 1))

        print('=============')
        y_pred = self.predict(X_train)
        print( "n_errors = ", np.abs(y_train-y_pred).sum() )
        print( "accuracy = ", 1.0 - np.abs(y_train-y_pred).sum()/len(y_pred) )
        print('=============')


        for e in range(n_epochs):
            print('\n epoch = ', e)
            for n in range(len(y_train)):
                yn = np.array( [[y_train[n]]] )
                xn  = X_train[n, :]
                u_0[:n_input,0] = xn
                

                E1 = self.energy_function(u_0, self.W, self.b, beta = 0, y_label = None)

                u_0  = self.get_fixed_point(u_0, self.W, self.b, beta = 0, y_label = None )

                E2 = self.energy_function(u_0, self.W, self.b, beta = 0 , y_label = None)

                u_beta  = self.get_fixed_point(u_0, self.W, self.b, beta = beta, y_label = yn)

                if E2 - E1 > 0:
                    print(E2 - E1, (E2 - E1)/E1)


                self.W, self.b = self.update_params(self.W, self.b, u_0, u_beta, beta)

            y_pred = self.predict(X_train)
            print( "n_errors = ", np.abs(y_train-y_pred).sum() )
            print( "accuracy = ", 1.0 - np.abs(y_train-y_pred).sum()/len(y_pred) )
            self.y_train = y_train
            self.y_pred  = y_pred

        
#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------

def get_train_test_data(n_samples = 400, test_size = 0.25):
    """
    Returns X (n_samples, n_features) and y (n_features,)
    """
    assert n_input > 1
    mu_1 = np.zeros( (n_input, 1) )
    mu_2 = np.zeros( (n_input, 1) )
    mu_1[0, 0] = 1
    mu_2[0, 0] = -1
    sigma = 0.25*2

    n1 = n_samples//2
    n2 = n_samples - n1

    X1 = mu_1.dot(np.ones((1,n1))) + np.random.normal(0, sigma, (2, n1))
    y1 = np.zeros(n1)

    X2 = mu_2.dot(np.ones((1,n2))) + np.random.normal(0, sigma, (2, n2))
    y2 = np.ones(n2)

    X = np.hstack((X1, X2)).T
    y = np.hstack((y1, y2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot( X_train[:,0], X_train[:,1], 'ro' )
    plt.plot( X_test[:,0], X_test[:,1], 'bo' )
    plt.show()

    return X_train, X_test, y_train, y_test



#-------------------------------------------------------------------------------
# Debug
#-------------------------------------------------------------------------------
def check_derivative():
    W = np.random.normal(size = (n_total, n_total))
    b = np.random.normal(size = (n_total, 1))
    u = np.random.normal(size = (n_total, 1))
    y_label = None # 1*np.array([0.7]).reshape(-1,1)
    beta    = 0.0


    eqprop = EqProp()
    der_fval = eqprop.der_energy_function(u, W, b, beta = beta, y_label = y_label)
    print(der_fval)

    eps   = 0.000001
    u_vec = np.array([0,0,0,1,0]).reshape(-1,1)

    print("----")

    fval = eqprop.energy_function(u, W, b, beta = beta, y_label = y_label)
    fval_eps = eqprop.energy_function(u + eps*u_vec, 
                                    W, b, beta = beta, y_label = y_label)

    der = (fval_eps - fval)/eps
    print(der)


def debug():
    W = np.random.normal(size = (n_total, n_total))
    b = np.random.normal(size = (n_total, 1))
    u = np.random.normal(size = (n_total, 1))
    y_label = np.array([1]).reshape(-1,1)
    beta    = 0.5

    u_init = u
    u_next = get_fixed_point(u_init, W, b, beta=beta, y_label = y_label)
    print(u_init)
    print(u_next)



#-------------------------------------------------------------------------------
# Main script
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    # check_derivative()
    X_train, X_test, y_train, y_test = get_train_test_data()
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    eqprop = EqProp()
    eqprop.train_model(X_train, y_train)


# if __name__ == '__main__':
#     main()
#     # check_derivative()
