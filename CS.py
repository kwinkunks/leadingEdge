from matplotlib import pyplot as plt
from numpy import real, imag, zeros, arange, pi, sin, linspace
import numpy.fft
from numpy.random import randn, random_sample

import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


class demo():

    def sig(self, t):

        
        return 10.0*sin(2*pi*self.f1*t + pi*self.phi1) + \
          20.0*sin(2*pi*self.f2*t + pi*self.phi2)

    
    def test_signal(self,f1, f2, phi1, phi2, duration):

        self.f1 = f1
        self.f2 = f2
        self.phi1 = phi1
        self.phi2 = phi2
        self.duration = duration

        self.dt = .001
        
        t = arange(0, duration, self.dt)
        xt = self.sig(t)
        return [xt,t]

    def fft(self, xt, t):

        dt = t[1]-t[0]

        Xf = numpy.fft.fft(xt) / xt.size
        f = numpy.fft.fftfreq(xt.size, self.dt)
    
        return [Xf, f]
    
    def fourier_plot(self,f,X, xlim=[-50,50], stem=True):
    
        # find non-zeros
        valid = abs(X) > .1

        if stem:
            plt.stem(f[valid], abs(X[valid]), linefmt='b-',
                    markerfmt='go', basefmt='b-')
            plt.stem(f[valid], imag(X[valid]), linefmt='b-',
                    markerfmt='ro',basefmt='b-')
            plt.legend(['real', 'imag'])
        else:
            plt.plot(f, abs(X))
            #plt.plot(f, imag(X), 'r-')
        
        plt.xlabel('frequency [Hz]')
        plt.ylabel('amplitude')
        plt.xlim(xlim)

  

    
    def ts_plot(self, t, xt):
        
        plt.plot(t, xt)
        plt.xlabel("time [s]"); plt.ylabel("amplitude")


    def uniform_sample(self, n_samples):

        t = linspace(0, self.duration, n_samples)

        xn = zeros(10*n_samples)
        xn[::10] = self.sig(t)
        
        Xf,f = self.fft(xn,t)
        
        plt.subplot(211)
        plt.stem(t,xn)

        plt.subplot(212)
        self.fourier_plot(f, Xf)
        
        

        
    
    def aliased(self):


        ny = (1/self.dt)/2
        fs = ny/8.0

        t = arange(0, self.duration, self.dt)
        sig1 = sin(2*pi*10*t + self.phi1)
        sig2 = sin(2*pi*(10 + fs) + self.phi1)

        plt.plot(t, sig1)
        plt.plot(t, sig2, 'g')

        samples = arange(0, self.duration, 1/fs)

        plot.plot(samples, sig1, 'rs')

    def randomized(self):

        ny = (1/self.dt)/2
        fs = ny/8.0

        t = arange(0, self.duration, self.dt)
        sig1 = sin(2*pi*10*t + self.phi1)
        sig2 = sin(2*pi*(10 + fs)*t + self.phi1)

        plt.plot(t, sig1)
        plt.plot(t, sig2, 'g')

        samples = random_sample(self.duration*fs)*duration

        plt.plot(samples, sin(2*pi*10*samples + self.phi1), 'rs')
        plt.plot(samples, sin(2*pi*(10 + fs)*samples + self.phi1))

    def random_sample(self, n_samples):
        pass
    
    

    
    



    
    

    
    
