import numpy as np
import matplotlib.pyplot as plt
import math

def quad2(x_o=[-9.0,9.0],a=2.0,eta=0.1,threshold=0.01,maxiter=1000,alpha=0,anim = 1):
    it = 0
    x1 = np.linspace(-10,10,21)
    
    x2 = np.linspace(-10,10,21)
    
    [X,Y] = np.meshgrid(x1,x2)
    
    Y = (a*X**2+Y**2)/2
    
    plt.clf()
    plt.contour(Y,10)
    plt.xticks([0,5,10,15,20],[-10, -5, 0, 5, 10])
    plt.yticks([0,5,10,15,20],[-10, -5, 0, 5, 10])
    ax = plt.gca()
    ax.set_aspect('equal','box')
    
    plt.tight_layout()
    
    
    f = (a*x_o[0]**2+x_o[1]**2)/2
    
    varx = np.array([0,0])
    ###Gradient Method####
    while it != maxiter:
        fold = f
        
        grad = np.array([a*x_o[0], x_o[1]])
        
        varx = alpha*varx+(1-alpha)*grad
        x_old = np.asarray(x_o)

        x_o = np.asarray(x_o-eta*varx)
    
        try:
            f = (a*x_o[0]**2+x_o[1]**2)/2
            if (f < threshold or fold < threshold):
                break
            else:
                if anim:
                    plt.plot([x_old[0]+10, x_o[0]+10],[x_old[1]+10,x_o[1]+10],'r.-')
                    plt.pause(0.2)                    
                it += 1
        except:
            print('Diverged!')
            #plt.show()
            break
        
    if it == maxiter:
        return False, it
    else:
        return True, it+1
    

if __name__ == "__main__":
    eta = (0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)
    alpha = (0, 0.5, 0.7, 0.9, 0.95)
    for e in eta:
        for a in alpha:
            conv, iters = quad2(a=20.0,eta=e,anim = 0,alpha=a)
            if conv:
                string = f"{iters} iterations"
            else:
                string = "Diverged!"
            print(f"eta: {e}, alpha: {a} ->" + string)
