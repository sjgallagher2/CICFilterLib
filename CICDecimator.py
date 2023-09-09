# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:31:19 2023

@author: Sam.Gallagher

Implement a basic CIC decimator with a given number of stages
"""

from collections import deque  # Double-ended queue (FIFO/shift register)
import numpy as np
from numpy import sin,cos,exp,sqrt,log10,pi,arange
from numpy.random import random_sample
import scipy.signal as signal
from scipy.fft import fft, fftfreq, fftshift

import matplotlib.pyplot as plt
import matplotlib as mpl

from fxpmath import Fxp as fi 


# Consistent stem plots
def stem(t, y, color='k', newfig=False):
    if newfig:
        plt.figure()
    mrkline,_,_ = plt.stem(t,y,color,basefmt=color)
    plt.xlim(np.min(t),np.max(t))
    
    mrkline.set_markerfacecolor('none')

def plot_fft(data,Fs=1,two_sided=True):
    # Window and run FFT
    fftsize = len(data)
    data_windowed = data*np.hanning(fftsize)
    data_fft = fft(data_windowed) * 2/fftsize  # x2 multiplier for Hann window
    # Convert from volts to dB
    data_fft_dB = 20*log10(np.abs(data_fft))
    # Make frequency bins for x axis
    fbins = fftfreq(fftsize, 1/Fs)
    data_fft_dB = fftshift(data_fft_dB)
    fbins = fftshift(fbins)

    fig,(ax1) = plt.subplots(figsize=(10,6))
    ax1.plot(fbins,data_fft_dB,'k')
    ax1.grid(True)
    if two_sided:
        ax1.set_xlim(np.min(fbins),np.max(fbins))
    else:
        ax1.set_xlim(0,np.max(fbins))

    fig.tight_layout()


class CICBlockIntegrator:
    def __init__(self,wordlength=None,fractionlength=None):
        self.y = 0
        self.yprev = 0
        self.prec=''
        # Handle fixed-point initialization
        if wordlength is not None:
            if fractionlength is not None:
                self.prec = f'fxp-s{wordlength}/{fractionlength}'
                self.y = fi(0, dtype=self.prec, overflow='wrap')
            else:
                raise ValueError("Trying to instantiate word length without specifying fraction length")
        else:
            if fractionlength is not None:
                raise ValueError("Trying to instantiate fraction length without specifying word length")

    def update(self,x):
        if self.prec != '':
            x = fi(x,dtype=self.prec)
            self.yprev = fi(self.y,dtype=self.prec,overflow='wrap')
            self.y = fi(x+self.yprev,dtype=self.prec,overflow='wrap')
        else:
            self.yprev = self.y
            self.y = x + self.yprev


class CICBlockComb:
    def __init__(self,M,wordlength=None,fractionlength=None):
        self.xprev = deque(np.zeros(M), maxlen=M)  # Shift register of length M
        
        self.y = 0 
        self.prec=''
        # Handle fixed-point initialization
        if wordlength is not None:
            if fractionlength is not None:
                self.prec = f'fxp-s{wordlength}/{fractionlength}'
                self.y = fi(0, dtype=self.prec, overflow='wrap')
                for i in range(M):
                    self.xprev.append(fi(0,dtype=self.prec,overflow='wrap'))
            else:
                raise ValueError("Trying to instantiate word length without specifying fraction length")
        else:
            if fractionlength is not None:
                raise ValueError("Trying to instantiate fraction length without specifying word length")

    def update(self,x):
        if self.prec != '':
            self.y = fi(x - self.xprev[0],dtype=self.prec, overflow='wrap')  # xprev[0] is oldest element
            self.xprev.append(fi(x,dtype=self.prec, overflow='wrap'))
        else:
            self.y = x - self.xprev[0]  # xprev[0] is oldest element
            self.xprev.append(x)

class CICDecimator:
    def __init__(self, R=2, M=1, N=2, FixedPointDataType='Full precision', SectionWordLengths=[16,16,16,16], 
                 SectionFractionLengths=0, OutputWordLength=32, OutputFractionLength=0, **kwargs):
        """
        A Cascaded Integrator-Comb (CIC) filter, based on the Matlab CICDecimator type.
        To use this object, instantiate it, and then call the object like a function. 
        
        
        
        :param R: Decimation factor, defaults to 2
        :param M: Differential delay, defaults to 1
        :param N: Number of sections, defaults to 2
        :param FixedPointDataType: Type of fixed-point operations to use, one of 
                {'Full precision', 'Minimum section word lengths', 'Specify word lengths',
                 'Specify word and fraction lengths'}. Defaults to 'Full precision'
        :param SectionWordLengths: DESCRIPTION, defaults to [16,16,16,16]
        :param SectionFractionLengths: DESCRIPTION, defaults to 0
        :param OutputWordLength: DESCRIPTION, defaults to 32
        :param OutputFractionLength: DESCRIPTION, defaults to 0
        :param **kwargs: Additional arguments are aliases. They are DecimationFactor (R), 
                        DifferentialDelay (M), and NumSections (N)

        """
        self.DecimationFactor = R   # If not overridden
        self.DifferentialDelay = M  #
        self.NumSections = N        #
        self.FixedPointDataType = FixedPointDataType
        self.SectionWordLengths = SectionWordLengths
        self.SectionFractionLengths = SectionFractionLengths
        self.OutputWordLength = OutputWordLength
        self.OutputFractionLength = OutputFractionLength
        
        # Check for keywords: DecimationFactor, DifferentialDelay, NumSections
        for k,v in kwargs.items():
            if k == 'DecimationFactor':
                self.DecimationFactor = v
            elif k == 'DifferentialDelay':
                self.DifferentialDelay = v
            elif k == 'NumSections':
                self.NumSections = v
        
        # Initialize our integrators and combs
        # Both are used by calling `update(x)` for each new sample and reading `y`
        self.integrators = []
        self.combs = []
        for i in range(self.NumSections):
            # Initialize integrators/combs based on fixed point parameters
            if self.FixedPointDataType == 'Full precision':
                self.integrators.append(CICBlockIntegrator(wordlength=64,fractionlength=32))
                self.combs.append(CICBlockComb(self.DifferentialDelay,wordlength=64,fractionlength=32))
            elif self.FixedPointDataType == 'Minimum section word lengths':
                pass
            elif self.FixedPointDataType == 'Specify word lengths':
                pass
            elif self.FixedPointDataType == 'Specify word and fraction lengths':
                pass
    
    def __call__(self,x: np.ndarray):
        # Go through each step and update the integrators and (if this is the Rth sample) combs
        L = len(x)
        output = []
        for t in range(L):
            self.integrators[0].update(x[t])
            g = self.integrators[0].y
            if self.NumSections > 1:
                for integrator in self.integrators[1:]:
                    integrator.update(g)
                    g = integrator.y 
            # Now `g` is the last output from an integrator block
            # If this is the Rth step, pass through combs
            if t % self.DecimationFactor == 0:
                self.combs[0].update(g)
                y = self.combs[0].y
                if self.NumSections > 1:
                    for comb in self.combs[1:]:
                        comb.update(y)
                        y = comb.y
                output.append(y)
        return output


class CICCompensationDecimator:
    def __init__(self, Decimator : CICDecimator, DecimationFactor, PassbandFrequency, StopbandFrequency, StopbandAttenuation, SampleRate):
        pass

class CICInterpolator:
    def __init__(self, InterpolationFactor, DifferentialDelay, NumSections, FixedPointDataType):
        pass

class CICCompensationInterpolator:
    def __init__(self, Interpolator : CICInterpolator, InterpolationFactor, PassbandFrequency, StopbandFrequency, StopbandAttenuation, SampleRate):
        pass

def CICTransfer(z,R,M,N):
    H = (1-z**(-R*M))/(1-z**(-1))
    H = H**N 
    return H

# %%
if __name__ == '__main__':
    # Filter parameters
    R = 3  # Decimation rate
    M = 2  # Comb delay parameter
    N = 2  # Number of stages
    
    # Input signal
    L = 500
    ns = 0.1  # Noise amplitude
    t = arange(0,L)
    f = 1/100
    x = sin(0.02*t) + 0.5*cos(t)
    #x = np.random.random_sample(len(t))-0.5
    #plot_fft(x,two_sided=False)
    #x = np.zeros_like(t)
    #x[0] = 1
    
    # %%
    # Class version
    decim = CICDecimator(R=R,M=M,N=N)
    #stem(t,x,color='b')
    #stem(t[::R],decim(x),color='r')
    #stem(x,t=t)
    #stem(decim(x),t=t[::R])
    plt.plot(20*log10(np.abs(fftshift(fft(x)))))
    plt.xlim([250,500])
    plt.figure()
    plt.plot(20*log10(np.abs(fftshift(fft(decim(x))))))
    plt.xlim([84,167])
    
    # %% Transfer function
    # w_range = np.linspace(0,1,1000)
    # z = exp(1j*w_range)
    # CIC_response = 20*np.log10(np.abs(CICTransfer(z,R,M,N)))
    # #plt.figure()
    # plt.plot(w_range/2, CIC_response - 56)

            
