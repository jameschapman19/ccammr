from ccammr import CCAMMR, CCASample
import numpy as np

#example comparing CCAMMR and CCASample for finding effects in
#subsets of data with different distributions based on their learnt
#sample weights
def main():
    #generate data
    n = 200
    p = 5
    q = 5
    noise=1e-10
    z= np.ones((n//2, 1))
    wx= np.random.normal(size=(p, 1))
    wy= np.random.normal(size=(q, 1))
    X=z@wx.T + np.random.normal(0,noise,size=(n//2, p))
    Y=z@wy.T + np.random.normal(0,noise,size=(n//2, q))

    #standardise
    X=(X-X.mean(axis=0))/X.std(axis=0)
    Y=(Y-Y.mean(axis=0))/Y.std(axis=0)
    X_noise=np.random.normal(size=(n//2, p))
    Y_noise=np.random.normal(size=(n//2, q))
    #standardise
    X_noise=(X_noise-X_noise.mean(axis=0))/X_noise.std(axis=0)
    Y_noise=(Y_noise-Y_noise.mean(axis=0))/Y_noise.std(axis=0)
    #combine X and Y with noise
    X=np.concatenate((X, X_noise))
    Y=np.concatenate((Y, Y_noise))

    #fit models
    #This is the Maximum Margin CCA Model where the sample weights are whether the dot product of the transformed X and the original Y
    #is greater than 1
    mmr = CCAMMR(C=1000000)
    mmr.fit([X, Y])
    #This is a normal CCA model where the sample weights are whether the product of the latent variables is greater than 0
    cca = CCASample()
    cca.fit([X, Y])

    #print sample weights. The first half of the sample weights (those with the signal) should be true and the second half false (those with the noise)
    print(mmr.sample_weights((X,Y)))
    print(cca.sample_weights((X,Y)))

if __name__ == '__main__':
    main()