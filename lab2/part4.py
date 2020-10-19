import numpy as np
import matplotlib.pyplot as plt
import math

def rosen(x_o=[-1.5,1.0],a=20.0,eta=0.001,threshold=0.001,maxiter=1000,alpha=0.0,anim = 1,up = 1,down = 1,reduce = 1):
    it = 0
    x1 = np.linspace(-2,2,201)
    
    x2 = np.linspace(-1,3,201)
    
    [X,Y] = np.meshgrid(x1,x2)
    
    Y = (1-X)**2 + a*(Y-X**2)**2
    
    v = np.linspace(math.floor(a/80)+3,Y.max(),math.floor(a))

    plt.clf()      
    plt.contour(Y,v)
    plt.xticks([0,50,100,150,200],[-2, -1, 0, 1, 2])
    plt.yticks([0,50,100,150,200],[-1, 0, 1, 2, 3])
    ax = plt.gca()
    ax.set_aspect('equal','box')
    
    plt.tight_layout()
    
    plt.plot(150,100,'b.')
    
    f = (1-x_o[0])**2+a*(x_o[1]-x_o[0]**2)**2
    fold = f
    minf = f

    gradold = np.array([0,0])
    
    eta1 = eta
    eta2 = eta
    
    varx = np.array([0.0,0.0])
    ###Gradient Method####
    while it != maxiter:
    
        grad = np.array([-2.0*(1-x_o[0])-4.0*a*(x_o[1]-x_o[0]**2)*x_o[0], 2.0*a*(x_o[1]-x_o[0]**2)])
        
        x_old = np.asarray(x_o)
 
        if (f>minf and reduce < 1):
            x_o[0] = minx1
            x_o[1] = minx2
            
            grad[0] = mingrad1
            grad[1]= mingrad2
            
            varx = np.array([0.0,0.0])
            
            eta1 = eta1*reduce
            eta2 = eta2*reduce
            
            gradold[0] = 0
            gradold[1] = 0
            
            fold = f
            f = minf
        else:
            minf = f
            
            minx1 = x_o[0]
            minx2 = x_o[1]
            
            mingrad1 = grad[0]
            mingrad2 = grad[1]
            
            if grad[0]*gradold[0] >0:
                eta1 = eta1*up
            else:
                eta1= eta1*down
            
            if grad[1]*gradold[1] >0:
                eta2 = eta2*up
            else:
                eta2 = eta2*down
                
            varx[0] = alpha*varx[0]-(1-alpha)*grad[0]
            varx[1] = alpha*varx[1]-(1-alpha)*grad[1]
            
            x_o[0] = x_o[0] + eta1*varx[0]
            x_o[1] = x_o[1] + eta2*varx[1]
            
            gradold = grad
            fold = f
 
        try:
            f = (x_o[0]-1)**2 + a*(x_o[1]-x_o[0]**2)**2
            if (f < threshold or fold < threshold):
                break
            else:
                if anim:
                    plt.plot([50*x_old[0]+100, 50*x_o[0]+100],[50*x_old[1]+50,50*x_o[1]+50],'r.-')
                    plt.xticks([0,50,100,150,200],[-2, -1, 0, 1, 2])
                    plt.yticks([0,50,100,150,200],[-1, 0, 1, 2, 3])
                    ax = plt.gca()
                    ax.set_aspect('equal','box')
                    plt.tight_layout()
                    plt.pause(0.1)
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
    eta = (0.08, 0.09, 0.1, 0.11, 0.12)
    alpha = (0.95,)
    for e in eta:
        for a in alpha:
            conv, iters = rosen(x_o=[-1.5,1.0],a=20.0,eta=e,threshold=.001,maxiter=1000,alpha=a,anim = 0,up = 1,down = 1,reduce = 1)
            if conv:
                string = f"{iters} iterations"
            else:
                string = "Diverged!"
            print(f"eta: {e}, alpha: {a} ->" + string)

    eta = (0.001, 0.01, 0.1, 1, 10)
    alpha = (0, 0.5, 0.7, 0.9, 0.95, 0.99)
    for e in eta:
        for a in alpha:
            conv, iters = rosen(x_o=[-1.5,1.0],a=20.0,eta=e,threshold=.001,maxiter=1000,alpha=a,anim = 0,up = 1.1,down = 0.9,reduce = 0.5)
            if conv:
                string = f"{iters} iterations"
            else:
                string = "Diverged!"
            print(f"eta: {e}, alpha: {a} ->" + string)
