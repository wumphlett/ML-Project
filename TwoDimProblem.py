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
        
    def createData(self, soln_rank: int, noise_frac: float, samples: int) -> tuple[np.ndarray, np.ndarray]:
        '''Create data of length samples that is seperable with a line of given rank with some noise'''
        assert 0 <= noise_frac < 1
        assert soln_rank > 0
        RANGE = self.value_range
        coefficients = np.random.randint(-RANGE, RANGE+1, size=(soln_rank+1))
        expression = '+'.join([f'{c}*x**{i}' for i,c in enumerate(coefficients)])
        self.seperator = lambdify('x', sympify(expression))
        self.data = (np.random.rand(samples,2) - 0.5)*RANGE*2
        self.labels = np.array([1 if (point[1] > self.seperator(point[0])) else 0 for point in self.data])
        
        noise_pts = int(samples * noise_frac)
        self.labels[:noise_pts] = np.mod(np.array(self.labels[:noise_pts]) + 1, 2)
        
        return (self.data, self.labels)
    
    def plotData(self, show_seperator=False):
        # ensure that data has been created
        assert self.data is not None and self.labels is not None and self.seperator is not None
        RANGE = self.value_range
        plt.scatter(self.data[:,0], self.data[:,1], c=['b' if l==1 else 'orange' for l in self.labels])
        if show_seperator:
            x = np.linspace(-RANGE, RANGE, 1000)
            x = np.array([i for i in x if -RANGE*1.5 <= self.seperator(i) <= RANGE*1.5])
            y = self.seperator(x)
            plt.plot(x, y, c='black')
        plt.show()
        
    def plotPred(self, y_pred, show_correct=True, seperator=None):
        # ensure that data has been created
        assert self.labels is not None and self.data is not None and self.seperator is not None
        assert len(y_pred) == len(self.labels)

        RANGE = self.value_range
        print(f"Accuracy = {np.mean(y_pred == self.labels)}")

        TP = np.logical_and(y_pred == 1, self.labels == 1)
        FP = np.logical_and(y_pred == 1, self.labels == 0)
        TN = np.logical_and(y_pred == 0, self.labels == 0)
        FN = np.logical_and(y_pred == 0, self.labels == 1)
                
        plt.scatter(self.data[TP,0], self.data[TP,1], c='g', label='TP')
        plt.scatter(self.data[FP,0], self.data[FP,1], c='r', label='FP')
        plt.scatter(self.data[TN,0], self.data[TN,1], c='lightgreen', label='TN')
        plt.scatter(self.data[FN,0], self.data[FN,1], c='#990000', label='FN')

        if seperator is not None:
            x = np.linspace(-RANGE, RANGE, 1000)
            x = np.array([i for i in x if -RANGE*1.5 <= seperator(i) <= RANGE*1.5])
            y = seperator(x)
            plt.plot(x, y, c='black')
        if show_correct:
            x = np.array([i for i in np.linspace(-RANGE, RANGE, 1000) if -RANGE*1.5 <= self.seperator(i) <= RANGE*1.5])
            y = self.seperator(x)
            plt.plot(x, y, c='green', linestyle='dashed')
        plt.legend()
        plt.title("Prediction Results for Model")
        plt.show()