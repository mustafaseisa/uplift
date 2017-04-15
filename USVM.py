from numpy import ndarray, array, repeat, hsplit, unique, where, delete, diag, hstack, vstack, eye, zeros, vectorize, linspace, bincount, argsort, cumsum
from cvxopt.solvers import qp, options
from cvxopt import matrix
from math import ceil

class USVM:
    
    def __init__(self, treatment, control):
        
        assert [type(treatment), type(control)] == [ndarray, ndarray] # numpy arrays only
        
        yt, Xt = hsplit(treatment, [1])
        yc, Xc = hsplit(control, [1])
        
        self._nt, mt = Xt.shape
        self._nc, mc = Xc.shape
        self._n = self._nc + self._nt # n is number of datum across both groups
        
        assert min(mt, mc, self._nt, self._nc) >= 1 and self._n >= 3 # data shouldn't be trivial
        assert mt == mc # same number of features in treatment and control
        
        self._m = mt # store number of features
        
        assert unique(yt).all() in [-1,1] and unique(yc).all() in [-1,1] # labels are binary
        
        tPlusIndex = where(yt.flatten() == 1.0)[0] # index for positive in treatment
        self._ntplus = len(tPlusIndex) # number of such points (length of index)
        tMinusIndex = delete(range(self._nt), tPlusIndex) # index for negative in treatment
        self._ntminus = self._nt - self._ntplus # number of such points
        
        self._Dtplus = Xt[tPlusIndex] # positive treatment datum
        self._Dtminus = Xt[tMinusIndex] # negative treatment datum
        
        cPlusIndex = where(yc.flatten() == 1.0)[0] # index for positive in control
        self._ncplus = len(cPlusIndex) # number of such points (length of index)
        cMinusIndex = delete(range(self._nc), cPlusIndex) # index for negative in control
        self._ncminus = self._nc - self._ncplus # number of such points
        
        self._Dcplus = Xc[cPlusIndex] # positive treatment datum
        self._Dcminus = Xc[cMinusIndex] # negative treatment datum
        
        # model parameters
        
        self.__optimized = False # indicator for whether otpimization routine was performed
        options['show_progress'] = False # supress optimization output
        
        self.w = None # hyperplane slope
        self.b1 = None # treatment group intercept
        self.b2 = None # control group intercept
        self.threshold = None # thresholding predictor function
        
        print("Successfully initialized.")
        
    def optimize(self, C1=1, C2divC1=1, feedback=True):
        
        assert C1 >=0 and C2divC1 >= 1
        
        p = self._m + 2 + 2*self._n # dimension of decision vector: w + b1 + b2 + xi
        P = diag([1.]*self._m + [0.]*(p-self._m)) # truncated identity
        
        q = array([0.]*(self._m + 2) + [C1]*self._n + [C2divC1*C1]*self._n).reshape((p,1)) # linear term
        
        h = array([-1.]*2*self._n + [0.]*2*self._n) # on right side of inequality
        
        ### Components of G matrix ###
        
        # upper left block
        
        omega0 = vstack( (self._Dtplus, self._Dcminus) )
        omega1 = vstack( (self._Dtminus, self._Dcplus) )
        wBlock = vstack( (-omega0, omega1, omega1, -omega0) )
        
        # upper middle block
        
        bBlock = array( [(1.,  0.)]*(self._ntplus  + self._ncminus) +
                        [(0., -1.)]*(self._ntminus + self._ncplus ) +
                        [(-1., 0.)]*(self._ntminus + self._ncplus ) +
                        [(0.,  1.)]*(self._ntplus  + self._ncminus) )
        
        # upper right block
        
        xiBlock = -eye(2*self._n)
        
        # lower left
        
        zeroBlock = zeros( (2*self._n, self._m + 2) )
        
        ### End ###

        G = vstack( (hstack( (wBlock, bBlock, xiBlock) ),
                     hstack( (zeroBlock, xiBlock) ) ) )
        
        ### CVXOPT Quadratic Programming ###
        
        theta = array(
                qp(P=matrix(P), q=matrix(q), G=matrix(G), h=matrix(h)).get('x')
                )
        
        self.w, self.b1, self.b2 = ( theta[:self._m], theta[self._m][0], theta[self._m + 1][0] )
        
        ### Parametrize the thresholding function ###
        
        threshold = lambda double: self.__threshold(double, b1=self.b1, b2=self.b2)
        
        self.threshold = vectorize(threshold, otypes=[int])
        
        ### End ###
        
        self.__optimized = True
        
        if feedback==True:
            return print("Optimal parameters stored.")
        else:
            pass
     
    def __threshold(self, double, b1, b2):

        if double > b1:
            return 1
        elif double <= b2:
            return -1
        else:
            return 0
    
    def predict(self, X):
        
        if self.__optimized == False:
            
            return print("Must run .optimize method before performing prediction.")
        
        else:
            
            assert type(X) == ndarray
            n, m = X.shape
            assert n >= 1 and m == self._m
            
            return self.threshold(X.dot(self.w))
    
    def rates(self, fixedC1 = .5, rangeC2divC1 = linspace(1, 1.5, 10)):
        
        triplesList = []
        data = vstack( (self._Dtplus, self._Dcminus, self._Dtminus, self._Dcplus) )
        
        for ratio in rangeC2divC1:
            
            self.optimize(C1 = fixedC1, C2divC1 = ratio, feedback = False)
            labels = self.predict(X = data)
            counts = tuple(bincount(labels.flatten() + 1, minlength=3)/self._n)
            triplesList.append(counts)
        
        from matplotlib import pyplot as plt
        
        plt.style.use('ggplot')
        plt.figure(figsize=(12,5))
        plt.plot(rangeC2divC1, triplesList, linewidth=3)
        plt.legend(['negative', 'neutral', 'positive'])
        plt.xlim([min(rangeC2divC1),max(rangeC2divC1)])
        
        plt.title("CLASSIFICATION RATES")
        plt.xlabel(r"PENALTY RATIO $C_2/C_1$")
        plt.ylabel("PERCENT")
        plt.show()

    def hyperplanes(self, slope, b1, b2, data):

        from matplotlib import pyplot as plt
        from numpy import meshgrid, arange
        plt.style.use('ggplot')
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        
        fig = plt.figure(figsize=(15,10))
        ax = fig.gca(projection='3d')
        
        xmin = min([min(frame[:,0]) for frame in data])
        xmax = max([max(frame[:,0]) for frame in data])
        ymin = min([min(frame[:,1]) for frame in data])
        ymax = max([max(frame[:,1]) for frame in data])
        
        X = arange(xmin, xmax, 0.25)
        Y = arange(ymin, ymax, 0.25)
        X, Y = meshgrid(X, Y)
        Z1 = b1 + slope[0]*X + slope[1]*Y
        Z2 = Z1 - b1 + b2
        
        surf1 = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.BuGn,
                           linewidth=0, antialiased=True, alpha=.3)
        
        surf2 = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.OrRd,
                           linewidth=0, antialiased=True, alpha=.3)

        tplus = ax.scatter(xs = data[0][:,0], ys = data[0][:,1], zs = 1, c='g', marker = '+', s=50, label='treatment+')
        tminus = ax.scatter(xs = data[2][:,0], ys = data[2][:,1], zs = -1, c='g', marker='o', label='treatment-')
        cplus = ax.scatter(xs = data[3][:,0], ys = data[3][:,1], zs = 1, c='r', marker='+', s=50, label='control+')
        cminus = ax.scatter(xs = data[1][:,0], ys = data[1][:,1], zs = -1, c='r', marker='o', label='control-')

        plt.legend(loc='best')
        plt.show()

    def __recall(self, data, gridSize = 10):
        
        assert self.__optimized == True
        assert type(data) == ndarray
        assert min(data.shape) > 2 
        
        y, X = hsplit(data, [1])
        ny = len(y)
        
        assert gridSize < ny
        assert unique(y).all() in [-1,0,1]
        assert X.shape[1] == self._m
        
        from math import ceil
        
        grid = linspace(0, ny - 1, gridSize, True)
        
        orderedLabels = y[argsort(X.dot(self.w), axis=0).flatten()[::-1]] == 1
        proportions = cumsum(orderedLabels)/sum(orderedLabels)
        
        recall = list(map(lambda tick: proportions[ceil(tick)], grid))
        recall.insert(0, 0.)

        grid = list((grid+1)/ny)
        grid.insert(0, 0.)
        
        return (grid, recall)

    def upliftCurve(self, treatment, control, gridSize = 20):

        xt, yt = self.__recall(treatment, gridSize)
        xc, yc = self.__recall(control, gridSize)

        tBias = self._ntplus/self._nt
        cBias = self._ncplus/self._nc

        diff = list(map(lambda t: tBias*t[0]-cBias*t[1], zip(yt, yc)))

        from matplotlib import pyplot as plt

        plt.style.use('ggplot')
        plt.figure(figsize=(15,6))

        # CAP CURVE

        plt.subplot(1,2,1)
        plt.plot(xt, yt, linewidth=2, label='treatment')
        plt.plot(xc, yc, linewidth=2, label='control')
        plt.plot([0,1], [0,1], linewidth=.5, label='baseline')
        plt.legend(loc='best')
        plt.xlabel('PERCENT DATA OBSERVED')
        plt.ylabel('PERCENT OF POSITIVE CAPTURED')

        # COMPARATIVE GAIN

        plt.subplot(1,2,2)
        plt.fill_between(xt, diff, [(tBias-cBias)*c for c in xt], label='treatment - control')
        # plt.legend(loc='best')
        plt.xlabel('PERCENT DATA OBSERVED')
        plt.ylabel('RELATIVE GAIN: TREATMENT - CONTROL')
        plt.ylim([0.,1.2*max(diff)])
        plt.show()
