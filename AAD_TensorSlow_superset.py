import numpy as np
import time
import matplotlib.pyplot as plt

tape = []

class number():
    def __init__(self,value):
        global tape
        dico = {}
        dico["value"] = value
        dico["args"] = 0
        dico["idx"] = len(tape)
        tape += [dico]
        self.dic = dico
    def __str__(self):
        return str(self.dic["value"])
    def __add__(self,rhs):
        if isinstance(rhs,number):
            res = number(self.dic["value"] + rhs.dic["value"])
            res.dic["args"] = 2
            res.dic["id1"] = self.dic["idx"]
            res.dic["id2"] = rhs.dic["idx"]
            res.dic["d1"] = 1
            res.dic["d2"] = 1
            return res
        else:
            res = number(self.dic["value"] + rhs)
            res.dic["args"] = 1
            res.dic["id1"] = self.dic["idx"]
            res.dic["d1"] = 1
            return res
    def __radd__(self,lhs):
        res = number(self.dic["value"] + lhs)
        res.dic["args"] = 1
        res.dic["id1"] = self.dic["idx"]
        res.dic["d1"] = 1
        return res
    def __sub__(self,rhs):
        if isinstance(rhs,number):
            res = number(self.dic["value"] - rhs.dic["value"])
            res.dic["args"] = 2
            res.dic["id1"] = self.dic["idx"]
            res.dic["id2"] = rhs.dic["idx"]
            res.dic["d1"] = 1
            res.dic["d2"] = -1
            return res
        else:
            res = number(self.dic["value"] - rhs)
            res.dic["args"] = 1
            res.dic["id1"] = self.dic["idx"]
            res.dic["d1"] = 1
            return res
    def __rsub__(self,lhs):
        res = number(lhs - self.dic["value"])
        res.dic["args"] = 1
        res.dic["id1"] = self.dic["idx"]
        res.dic["d1"] = -1
        return res
    def __mul__(self,rhs):
        if isinstance(rhs,number):
            res = number(self.dic["value"] * rhs.dic["value"])
            res.dic["args"] = 2
            res.dic["id1"] = self.dic["idx"]
            res.dic["id2"] = rhs.dic["idx"]
            res.dic["d1"] = rhs.dic["value"]
            res.dic["d2"] = self.dic["value"]
            return res
        else:
            res = number(self.dic["value"] * rhs)
            res.dic["args"] = 1
            res.dic["id1"] = self.dic["idx"]
            res.dic["d1"] = rhs
            return res
    def __rmul__(self,lhs):
        res = number(self.dic["value"] * lhs)
        res.dic["args"] = 1
        res.dic["id1"] = self.dic["idx"]
        res.dic["d1"] = lhs
        return res
    def __truediv__(self,rhs):
        if isinstance(rhs,number):
            res = number(self.dic["value"] / rhs.dic["value"])
            res.dic["args"] = 2
            res.dic["id1"] = self.dic["idx"]
            res.dic["id2"] = rhs.dic["idx"]
            res.dic["d1"] = 1/rhs.dic["value"]
            res.dic["d2"] = -self.dic["value"]/(rhs.dic["value"]**2)
            return res
        else:
            res = number(self.dic["value"] / rhs)
            res.dic["args"] = 1
            res.dic["id1"] = self.dic["idx"]
            res.dic["d1"] = 1/rhs
            return res
    def __rtruediv__(self,lhs):
        res = number(lhs/self.dic["value"])
        res.dic["args"] = 1
        res.dic["id1"] = self.dic["idx"]
        res.dic["d1"] = -lhs/(self.dic["value"]**2)
        return res
    def __neg__(self):
        res = number(-self.dic["value"])
        res.dic["args"] = 1
        res.dic["id1"] = self.dic["idx"]
        res.dic["d1"] = -1
        return res
    def adjoints(self):
        global tape
        N = self.dic["idx"]
        res = np.zeros(N+1)
        res[-1] = 1
        for i in range(N,-1,-1):
            v = res[i]
            for j in range(tape[i]["args"]):
                id = tape[i]["id" + str(j+1)]
                d = tape[i]["d" + str(j+1)]
                res[id] += d*v
        return res

def exp(self):
    if isinstance(self,number):
        y = np.exp(self.dic["value"])
        res = number(y)
        res.dic["args"] = 1
        res.dic["id1"] = self.dic["idx"]
        res.dic["d1"] = y
    else:
        res = np.exp(self)
    return res

def log(self):
    if isinstance(self,number):
        y = np.log(self.dic["value"])
        res = number(y)
        res.dic["args"] = 1
        res.dic["id1"] = self.dic["idx"]
        res.dic["d1"] = 1/self.dic["value"]
    else:
        res = np.log(self)
    return res

def sqrt(self):
    if isinstance(self,number):
        y = np.sqrt(self.dic["value"])
        res = number(y)
        res.dic["args"] = 1
        res.dic["id1"] = self.dic["idx"]
        res.dic["d1"] = 0.5/y
    else:
        res = np.sqrt(self)
    return res

def sigmoid(self):
    return 1/(1+exp(-self))

def tanh(self):
    e = exp(-2*self)
    return (1-e)/(1+e)

class function():
    def __init__(self,fs):
        self.fs = fs
    def call_and_derivate(self,x,clean_tape = True):
        global tape
        n = len(x)
        i0 = len(tape)
        z = [number(xi) for xi in x]
        for f in self.fs:
            z = f(z)
        grad = np.array(z.adjoints()[i0:i0+n])
        if clean_tape:
            tape = []
            return z.dic["value"],grad
        return z.dic["value"],grad
    def __call__(self,x):
        for f in self.fs:
            x = f(x)
        return x


class grad_optimizer():
    def __init__(self,lr,n_iter,F,x,isaad = True,step = None,verbose = False):
        self.F = F
        self.lr = lr
        self.n_iter = n_iter
        self.x = x
        self.isaad = isaad
        self.step = step
        self.history = []
        self.verbose = verbose
    def gradcomp(self):
        if self.isaad:
            z,grad = self.F.call_and_derivate(self.x)
        else:
            z = self.F(self.x)
            grad = np.zeros(len(self.x))
            for i in range(len(self.x)):
                self.x[i] += self.step
                zi = self.F(self.x)
                self.x[i] -= self.step
                grad[i] = (zi-z)/self.step
        return z,grad
    def stepcomp(self):
        ...
    def optimize(self):
        for i in range(self.n_iter):
            self.i = i
            z,grad = self.stepcomp()
            self.x = self.x - grad
            self.history += [z]
            if self.verbose:
                if i%10 == 0:
                    print(i,z)

class gd_opt(grad_optimizer):
    def stepcomp(self):
        z,grad = self.gradcomp()
        res = self.lr*grad
        return z,res


class decay_gd_opt(grad_optimizer):
    def __init__(self,lr,n_iter,F,x,decay,n_decay=50,isaad = True,step = None,verbose = False):
        super(decay_gd_opt,self).__init__(lr,n_iter,F,x,isaad,step,verbose = verbose)
        self.decay = decay
        self.n_decay = n_decay
    def stepcomp(self):
        z,grad = self.gradcomp()
        res = self.lr*grad
        if self.i%self.n_decay == 0:
            self.lr = self.decay*self.lr
        return z,res

class momentum_gd_opt(grad_optimizer):
    def __init__(self,lr,n_iter,F,x,decay,moment,n_decay=50,isaad = True,step = None,verbose = False):
        super(momentum_gd_opt,self).__init__(lr,n_iter,F,x,isaad,step, verbose = verbose)
        self.decay = decay
        self.moment = moment
        self.momentum = np.zeros(len(self.x))
        self.n_decay = n_decay
    def stepcomp(self):
        z,grad = self.gradcomp()
        self.momentum = (1-self.moment)*grad+self.moment*self.momentum
        res = self.lr*self.momentum
        if self.i%self.n_decay == 0:
            self.lr = self.decay*self.lr
        return z,res

# def f1(x):
#     [a,b] = x
#     return([sqrt(a*a+1)+b*b,exp(a-5)])
#
# def f2(x):
#     [a,b] = x
#     return sqrt(a+b)
#
# F = function([f1,f2])
# opt = decay_gd_opt(0.5,1000,F,np.array([1,1]),0.999)
# print(opt.x,F.call_and_derivate(opt.x))
# opt.optimize()
# print(opt.x,F.call_and_derivate(opt.x))
#
# opt = momentum_gd_opt(0.5,1000,F,np.array([1,1]),0.1)
# print(opt.x,F.call_and_derivate(opt.x))
# opt.optimize()
# print(opt.x,F.call_and_derivate(opt.x))

# def f1(x):
#     res1 = x[1]*x[1] + x[0]*x[0]
#     res2 = exp(x[0])
#     for i in range(2,len(x)):
#         res1 = res1 + x[i]*x[i]
#     return [res1,res2]
#
# def f2(x):
#     [a,b] = x
#     return sqrt(a+b)
#
# def f1(x):
#     res = []
#     for i in range(len(x)-1):
#         res += [sigmoid(x[i] + x[i+1])]
#     return res
#
# def f2(x):
#     res = 0
#     for i in range(len(x)):
#         res = res + x[i]
#     return sigmoid(res)
#
# F = function([f1,f2])
# times_aad = []
# times_FDM = []
# times_call = []
# sizes = [2**i for i in range(1,12)]
# for size in sizes:
#     print(size)
#     t = time.time()
#     opt = gd_opt(0.1,100,F,np.ones(size))
#     opt.optimize()
#     dt = time.time()-t
#     times_aad += [dt]
#
#     t = time.time()
#     opt = gd_opt(0.1,100,F,np.ones(size),False,0.001)
#     opt.optimize()
#     dt = time.time()-t
#     times_FDM += [dt]
#
#     t = time.time()
#     ones = np.ones(size)
#     for i in range(100):
#         F(ones)
#     dt = time.time()-t
#     times_call += [dt]
#
# fig,ax = plt.subplots(1,figsize = (8,5))
# ax.plot(sizes,times_aad)
# ax.plot(sizes,np.sqrt(times_FDM))
# ax.set_xlabel("input dimension")
# ax.legend(["required time with AAD","sqrt(required time) with FDM"])
# ax.set_title("computation of 100 gradients")
# plt.show()


class layer():
    def __init__(self,n1,n2,activation = sigmoid):
        self.n1 = n1
        self.n2 = n2
        self.w = None
        self.b = None
        self.activation = activation
    def __call__(self,x):
        res = []
        for j in range(self.n2):
            resj = 0
            for i in range(self.n1):
                resj = resj + (self.w[j][i]*x[i])
            resj = resj + self.b[j]
            resj = self.activation(resj)
            res += [resj]
        if self.n2 > 1:
            return res
        else:
            return res[0]

class NN():
    def __init__(self,layers,loss):
        self.layers = layers
        n_w = 0
        for layer in layers:
            n_w += layer.n1*layer.n2 + layer.n2
        self.n = n_w
        self.X = None
        self.y = None
        self.loss = loss
    def call_on_x(self,x):
        y = x.copy()
        for layer in self.layers:
            y = layer(y)
        return y
    def __call__(self,wb):
        num = 0
        for layer in self.layers:
            w = np.zeros((layer.n2,layer.n1),dtype = object)
            b = np.zeros(layer.n2,dtype = object)
            for j in range(layer.n2):
                for i in range(layer.n1):
                    w[j][i] = wb[num]
                    num += 1
                b[j] = wb[num]
                num += 1
            layer.w = w
            layer.b = b
        y_pred = []
        for x in self.X:
            y_pred += [self.call_on_x(x)]
        e = 0
        for i in range(len(self.y)):
            ee = self.loss(y_pred[i],self.y[i])
            e = ee + e
        return e/len(self.y)
    def call_and_derivate(self,wb):
        global tape
        num = 0
        for layer in self.layers:
            w = np.zeros((layer.n2,layer.n1),dtype = object)
            b = np.zeros(layer.n2,dtype = object)
            for j in range(layer.n2):
                for i in range(layer.n1):
                    w[j,i] = number(wb[num])
                    num += 1
                b[j] = number(wb[num])
                num += 1
            layer.w = w
            layer.b = b
        y_pred = []
        for x in self.X:
            y_pred += [self.call_on_x(x)]
        e = 0
        for i in range(len(self.y)):
            ee = self.loss(y_pred[i],self.y[i])
            e = ee + e
        e = e/len(self.y)
        adj = e.adjoints()
        grad = adj[:self.n]
        tape = []
        return e.dic["value"],np.array(grad)
    def set_weights(self,wb):
        num = 0
        for layer in self.layers:
            w = np.zeros((layer.n2,layer.n1),dtype = object)
            b = np.zeros(layer.n2,dtype = object)
            for j in range(layer.n2):
                for i in range(layer.n1):
                    w[j][i] = wb[num]
                    num += 1
                b[j] = wb[num]
                num += 1
            layer.w = w
            layer.b = b

class sto_gd_opt(grad_optimizer):
    def __init__(self,lr,n_iter,F,x,X,y,batchsize = 32,isaad = True,step = None,verbose = False):
        super(sto_gd_opt,self).__init__(lr,n_iter,F,x,isaad,step, verbose = verbose)
        self.batchsize = batchsize
        self.X = X
        self.y = y
        self.n_batch = len(self.X)//self.batchsize
        if len(self.X)%self.batchsize != 0:
            self.n_batch += 1
    def gradcomp(self):
        if self.isaad:
            z,grad = self.F.call_and_derivate(self.x)
        else:
            z = self.F(self.x)
            grad = np.zeros(len(self.x))
            for i in range(len(self.x)):
                self.x[i] += self.step
                zi = self.F(self.x)
                self.x[i] -= self.step
                grad[i] = (zi-z)/self.step
        return z,grad
    def stepcomp(self):
        z,grad = self.gradcomp()
        res = self.lr*grad
        return z,res
    def optimize(self):
        j = 0
        i = 0
        while i<self.n_iter:
            self.i = i
            self.F.X = self.X[j*self.batchsize:min((j+1)*self.batchsize,len(self.X))]
            self.F.y = self.y[j*self.batchsize:min((j+1)*self.batchsize,len(self.X))]
            z,grad = self.stepcomp()
            self.x = self.x - grad
            self.history += [z]
            j = j+1
            if j == self.n_batch:
                j = 0
                i = i+1
                p = np.random.permutation(len(self.X))
                self.X = self.X[p]
                self.y = self.y[p]
            if self.verbose:
                if i%10 == 0 and j == 0:
                    print(i,z)

def MSE(y_pred,y):
    e = (y_pred-y)
    return e*e
def logloss(y_pred,y):
    return -(y*log(y_pred) + (1-y)*log(1-y_pred))


f1 = layer(2,4,activation = tanh)
f2 = layer(4,2,activation = tanh)
f3 = layer(2,1,activation = sigmoid)
nn = NN([f1,f2,f3],loss = logloss)

X = []
y = []
for i in range(200):
    X += [[2*np.random.random()-1,2*np.random.random()-1]]
    r = X[-1][0]*X[-1][0] + X[-1][1]*X[-1][1]
    if r<0.25:
        X[-1][0] = X[-1][0]*0.7
        X[-1][1] = X[-1][1]*0.7
        y += [1]
    else:
        X[-1][0] = X[-1][0]*1.3
        X[-1][1] = X[-1][1]*1.3
        y += [0]
X = np.array(X)
y = np.array(y)
opt = sto_gd_opt(0.1,500,nn,np.random.normal(size = nn.n),X,y,verbose = True)
opt.optimize()
nn.set_weights(opt.x)

y_pred = []
for x in X:
    y_pred += [nn.call_on_x(x)]

x1 = [x[0] for x in X]
x2 = [x[1] for x in X]
plt.scatter(x1,x2,c = y_pred)
plt.show()
plt.plot(opt.history)
plt.show()

