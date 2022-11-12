# generate noisy data
import matplotlib.pyplot as plt
from sympy import lambdify, sympify
import numpy as np

class TwoDimProblem:
    def __init__(self, value_range: int):
        self.seperator = None
        self.data = None
        self.value_range = value_range 
        self.labels = None
        
    def createData(self, soln_rank: int, noise_frac: float, samples: int) -> np.ndarray:
        '''Create data of length samples that is seperable with a line of given rank with some noise'''
        assert 0 <= noise_frac < 1
        assert soln_rank > 0
        RANGE = self.value_range
        coefficients = np.random.randint(-RANGE, RANGE+1, size=(soln_rank+1))
        expression = '+'.join([f'{c}*x**{i}' for i,c in enumerate(coefficients)])
        self.seperator = lambdify('x', sympify(expression))
        self.data = (np.random.rand(samples,2) - 0.5)*RANGE*2
        self.labels = [1 if (point[1] > self.seperator(point[0])) else 0 for point in self.data]
        
        noise_pts = int(samples * noise_frac)
        self.labels[:noise_pts] = np.mod(np.array(self.labels[:noise_pts]) + 1, 2)
        
        return self.data, self.labels
    
    def plotData(self, show_seperator=False):
        RANGE = self.value_range
        plt.scatter(self.data[:,0], self.data[:,1], c=['b' if l==1 else 'orange' for l in self.labels])
        if show_seperator:
            x = np.linspace(-RANGE, RANGE, 1000)
            x = np.array([i for i in x if -RANGE*1.5 <= self.seperator(i) <= RANGE*1.5])
            y = self.seperator(x)
            plt.plot(x, y, c='black')
        plt.show()
        
    def plotPred(self, y_pred, seperator=None, show_correct=True):
        RANGE = self.value_range
        assert len(y_pred) == len(self.labels)
        print(f"Accuracy = {np.mean(y_pred == self.labels)}")
        scatter_colors = []
        for p, a in zip(y_pred, self.labels):
            if not p == a and a == 0:
                scatter_colors.append('#990000')
            elif  not p == a and a == 1:
                scatter_colors.append('r')
            elif p == a == 1:
                scatter_colors.append('lightgreen')
            else:
                scatter_colors.append('g')
                
        plt.scatter(self.data[:,0], self.data[:,1], c=scatter_colors)
        if seperator is not None:
            x = np.linspace(-RANGE, RANGE, 1000)
            x = np.array([i for i in x if -RANGE*1.5 <= seperator(i) <= RANGE*1.5])
            y = seperator(x)
            plt.plot(x, y, c='black')
        if show_correct:
            x = np.linspace(-RANGE, RANGE, 1000)
            x = np.array([i for i in x if -RANGE*1.5 <= self.seperator(i) <= RANGE*1.5])
            y = self.seperator(x)
            plt.plot(x, y, c='green', linestyle='dashed')
        plt.show()