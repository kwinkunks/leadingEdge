from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy import real, imag, zeros, arange, pi, sin, linspace, ceil,\
     argsort, eye, abs
import numpy.fft
from numpy.random import randn, choice, random_sample

import numpy as np
from matplotlib import gridspec






    
class demo():

    def __init__(self):

        self.f1 = 10. #[Hz]
        self.f2 = 40. #[Hz]
        self.phi1 = pi/5.
        self.phi2 = pi/9.
        self.duration = 2.0#[s]

        self.ny_dt = 1.0/(2*(self.f2+1))

        self.dt = .001 # high sample rate, pretend to be continuous

    def sig(self, t):

        
        return 50.0*sin(2*pi*self.f1*t + pi*self.phi1) + \
          100.0*sin(2*pi*self.f2*t + pi*self.phi2)

    
    def test_signal(self):
        
        t = arange(0, self.duration, self.dt)
        xt = self.sig(t)
        
        return [xt,t]

    def fft(self, xt, t):

        dt = t[1]-t[0]

        # pad up so dt is
        Xf = numpy.fft.fft(xt) / xt.size
        f = numpy.fft.fftfreq(xt.size, self.dt)
    
        return [Xf, f]
    
    def fourier_plot(self,f,X, xlim=[-50,50], stem=True):

        # find non-zeros
        valid = abs(X) > 1

        if stem:
            
            plt.stem(f[valid], abs(X[valid]),
                     linefmt='b-',
                    markerfmt='go', basefmt='k-')
            
            plt.plot(xlim, [0,0])
            
        else:

            index = argsort(f)
            plt.plot(f[index],abs(X[index]))
            #plt.plot(f, imag(X), 'r-')
        
        plt.xlabel('frequency [Hz]')
        plt.ylabel('|A|')
        plt.xlim(xlim)
        #plt.ylim(0,75)
  

    
    def ts_plot(self, t, xt):
        
        plt.plot(t, xt)
        plt.xlabel("time [s]"); plt.ylabel("amplitude")
        plt.xlim([0,.5])
        plt.ylim([-300,300])


    def uniform_sample(self, downsample_factor):

        ds_factor = downsample_factor
        
        t = arange(0,self.duration,self.dt)
        xt = self.sig(t)

        ny_t = arange(0,self.duration,self.ny_dt)
        x_nyq = self.sig(ny_t)

        xn = zeros(x_nyq.size)
        xn[::ds_factor] = self.sig(ny_t[::ds_factor])


        Xfn = (numpy.fft.fft(xn))/xn.size
        fn = numpy.fft.fftfreq(xn.size, self.ny_dt)

        # zero pad for recon
        reconF = zeros(xt.size, dtype=np.complex64)
        reconF[:Xfn.size/2] = Xfn[:Xfn.size/2]
        reconF[-Xfn.size/2:] = Xfn[Xfn.size/2:]
        
        recon = real(numpy.fft.ifft(reconF))

        gs = gridspec.GridSpec(2, 1)

        ax1 = plt.subplot(gs[0,0])
  
        ax1.plot(t, xt, 'g', linewidth=.65, label='True')
        ax1.plot(t, recon*xt.size,'r', linewidth=.65,
                 label='Reconstructed')

        ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax1.stem(ny_t[::ds_factor],x_nyq[::ds_factor],
                 basefmt='b-')

       
        ax1.set_xlim([0,.5])
        ax1.set_ylim([-300,300])
              
        ax2 =  plt.subplot(gs[1,0])
        self.fourier_plot(fn,Xfn, stem=False)

        

        plt.tight_layout()
        
    def aliased(self):

        duration = .1
        ny = (2000.)/2
        t = arange(0, duration, 1/(2*ny))
        sig1 = sin(2*pi*10*t)
        sig2 = sin(2*pi*(10 + ny/8)*t)


        gs = gridspec.GridSpec(3,1)
        ax1 = plt.subplot(gs[0,0])
        
        ax1.plot(t, sig1);plt.xlim(0,.1)
        ax1.plot(t, sig2, 'g')
        samples = arange(0, duration, 1/(ny/8))
        ax1.plot(samples,sin(2*pi*10*samples), 'rs')
        ax1.set_xticks(samples, minor=True)
        ax1.xaxis.grid(which='minor')

        ax2 = plt.subplot(gs[1,0])

        ax2.stem(samples, sin(2*pi*10*samples),
                 linefmt='b-', markerfmt='bo', basefmt='k-')
        ax2.set_xlim(0,.1)
        ax2.plot([0,.1], [0,0], 'k-')

        ax3 = plt.subplot(gs[2,0])

        ax3.stem(samples, sin(2*pi*(10 + ny/8)*samples),
                 linefmt='g-', markerfmt='go', basefmt='k-')
        ax3.set_xlim(0,.1)
        plt.tight_layout()

        
    def randomized(self):


        duration = 0.1
        
        ny = (2000.)/2
        t = arange(0, duration, 1/(2*ny))
        sig1 = sin(2*pi*10*t)
        sig2 = sin(2*pi*(10 + ny/8)*t)
        plt.plot(t, sig1);plt.xlim(0,.1)
        plt.plot(t, sig2, 'g')
        
        samples = random_sample(duration*(ny/8))*duration


        gs = gridspec.GridSpec(3,1)
        ax1 = plt.subplot(gs[0,0])
        
        ax1.plot(t, sig1);plt.xlim(0,.1)
        ax1.plot(t, sig2, 'g')

        
        ax1.plot(samples,sin(2*pi*10*samples), 'bs')
        ax1.plot(samples,sin(2*pi*(10 + ny/8)*samples), 'gs')
        ax1.set_xticks(samples, minor=True)
        ax1.xaxis.grid(which='minor')

        ax2 = plt.subplot(gs[1,0])

        ax2.stem(samples, sin(2*pi*10*samples),
                 linefmt='b-', markerfmt='bo', basefmt='k-')
        ax2.set_xlim(0,.1)
        ax2.plot([0,.1], [0,0], 'k-')
        ax3 = plt.subplot(gs[2,0])
        ax3.stem(samples, sin(2*pi*(10 + ny/8)*samples),
                 linefmt='g-', markerfmt='go', basefmt='k-')
        ax3.set_xlim(0,.1)
        ax3.plot([0,.1], [0,0], 'k-')
        plt.tight_layout()

    def random_sample(self, ds_factor=15, threshold=0,
                      ncoeffs='all'):

        
        t = arange(0,self.duration,self.dt)
        xt = self.sig(t)
        Xf,f = self.fft(xt,t)

        samples = choice(range(xt.size), int(xt.size/ds_factor),
                         replace=False)
        
        xn = zeros(xt.size)
        xn[samples] = self.sig(t[samples])


        Xfn = numpy.fft.fft(xn)*ds_factor
        fn = numpy.fft.fftfreq(xn.size, self.dt)


        Xfn[abs(fn) > 50] = 0.0
        Xfn[abs(Xfn) < threshold] = 0.0


        if ncoeffs != 'all':

            coeffs = argsort(abs(Xfn))[::-1][:ncoeffs]
            values = Xfn[coeffs]
            Xfn *= 0
            Xfn[coeffs] = values
            
        recon = real(numpy.fft.ifft(Xfn))

        gs = gridspec.GridSpec(2, 1)

        ax1 = plt.subplot(gs[0,0])
        ax1.stem(t[samples], xn[samples])
        ax1.plot(t, xt, 'g', linewidth=.65)
        ax1.plot(t, recon[:recon.size],'r', linewidth=.65)
        ax1.set_xlim([0,.5])

        ax2 =  plt.subplot(gs[1,0])
        self.fourier_plot(fn,Xfn, stem=False)

        plt.tight_layout()
        

        
    

        
    
    

    
    



    
    

    
    
