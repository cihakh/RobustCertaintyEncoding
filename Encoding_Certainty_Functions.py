#!/usr/bin/env python
# coding: utf-8

# In[1]:


#For Calculations
import math
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
import scipy as scp
from scipy import optimize
from cmath import sqrt
# import random
from numpy.random import default_rng
from scipy.linalg import expm
import scipy.integrate as integrate

#Progress Bars
from tqdm.notebook import tqdm

rng = default_rng()
    
def Heavi(un, theta):
    '''
    Function defining the heaviside step function used in 
    defining the firing thresholds. For one step, the firing is 
    equivalent to a heaviside function.
    Input: Activity profile un and firing threshold
    Output: H(un-theta)
    '''
    x=un-theta
    return np.piecewise(x,[x<0,0<=x],[lambda x: 0,lambda x: 1])

def find_interfaces(x, un, theta):
    '''
    The purpose of this function is to find threshold crossings (or any crossings of a function) on domain x.
    This function is particularly useful for finding stationary solutions beyond the first.
    Inputs: spatial domain x, activity profile un, and firing threshold theta
    Output: array of crossing positions
    '''
    y=un-theta #shifts threshold crossings to zero value
    s = np.abs(np.diff(np.sign(y))).astype(bool) #searches for sign changes where True is returned
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1) #returns approximations of interfaces via linear interpolation

def Tophat(x, low, high):
    '''
    Rectangular function useful in defining cue profiles 
    as well as intervals of integration where it is convenient.
    Input: spatial domain x, lower and upper bounds of nonzero interval
    Output: Top-hat function returning 1 for x in [lower,upper] and 0 otherwise
    '''
    return np.piecewise(x,[x<low,(low<=x)&(x<=high),x>high],[lambda x: 0,lambda x: 1,lambda x:0])

def firing(un,thvec):
    '''
    Function defining the stairstep firing rate. 
    Thetas must be found first using the solution finders above.
    Input: activity profile and the vector of thresholds.
    Output: stairecase "pyramid" defining the increasing activity regions
    '''
    num=len(thvec)
    fire=0*un
    for theta in thvec:
        if max(un)>theta:
            fire+=Heavi(un, theta)
    return fire/num
    
def weight_ker(amplitude, kappa, mode, x):
    '''
    Synaptic weight kernels are finite series approximations and
    Von-Mises-like. That is they are Von-Mises distributions that 
    are rescaled such that the peak value is chosen by the user "amplitude".
    Input: amplitude, synaptic footprint (where larger kappa implies narrower profiles), 
    number of modes for the finite series, and spatial discretization.
    Output: synaptic weight kernel for excitatory or inhibitory population.
    '''
    dum=scp.special.i0(kappa)
    normdum=scp.special.i0(kappa)
    for i in range(1,mode+1):
        dum+=2*scp.special.iv(i,kappa)*np.cos(i*x)
        normdum+=2*scp.special.iv(i,kappa)
    return amplitude*dum/normdum    

def weight_coeff(amplitude, kappa, mode):
    '''
    Coefficients for the synaptic weight kernels utilized in finding stationary solutions
    as well as in amplitude evolution theory equations. 
    Input: amplitude, synaptic footprint (where larger kappa implies narrower profiles), 
    and number of modes for the finite series (should match the weight functions)
    Ouput: vector of the coefficients
    '''
    first=[scp.special.i0(kappa)]
    coeffs=first+[2*scp.special.iv(i,kappa) for i in range(1,mode+1)]
    normalize=sum(coeffs)
    return amplitude*np.array(coeffs)/normalize


def first_stationary_sol(steps, halfwidthvec, w_coeffs):
    '''
    Returns the lowest level stationary solution. Our approach is to feed in possible bump halfwidths
    and then solve for the thresholds.
    Input: number of steps in firing rate, a vector of possible halfwidths, weight coefficients
    Output: Array of possible thresholds yielding a solution.
    '''
    thvec=[w_coeffs[0]*halfw+sum([w_coeffs[i]*np.cos(halfw*i)*np.sin(halfw*i)/i for i in range(1,len(w_coeffs))]) 
           for halfw in halfwidthvec]
    return 2*np.array(thvec)/steps

def amplitude(steps, w_coeff, halfwidths):
    '''
    Returns amplitudes used in choosing thresholds for possible stationary solutions.
    This function is mainly used to return the amplitude for the first stationary solution solve, 
    though it may come in handy if one chooses to solve for stationary solutions another way.
    Input: number of steps in firing rate, weight coefficients, a vector of the corresponding halfwidths
    Output: the amplitude of the bump
    '''
    return (w_coeff[0]*sum([h for h in halfwidths])+
            sum([w_coeff[i]*sum([np.sin(i*h) for h in halfwidths])/i for i in range(1,len(w_coeff))]))*(2/steps)

def stationary_ampl(steps, w, ucoeff, th_known, th_next, Avec,x):
    '''
    Returns the two amplitude stationary solutions (if there are any) for the next
    choice of threshold. This function is largely used to determine the 'next' threshold
    that yields the next state of stable bump solutions. 
    Input: number of steps in firing rate, weight functions, coefficients of normalized activity profile
    (note: the first solution must be obtianed by the above to find these coefficients, after the first
    solution is obtained, subsequent activity profiles have been observed to be of the same shape with nearly
    identical normalized coefficients), a vector of known thresholds, 
    the next threshold to test, a vector of possible amplitudes, the spatial discretization
    Output: narrow and broad solutions if they exist for that next threshold, np.nan's otherwise
    '''
    thvec=np.hstack((th_known,th_next))
    expected_solutions= len(thvec)*2
    def func(amp):
        dum=-steps*amp/len(thvec)+np.trapz(w*firing(amp*sum([ucoeff[i]*np.cos(i*x) for i in range(len(ucoeff))]),thvec),x)
        return dum
    zeros=find_interfaces(Avec, np.array([func(A) for A in Avec]), 0)
    if len(zeros)==expected_solutions:
        narrow=zeros[-2]
        broad=zeros[-1]
    else:
        narrow=np.nan
        broad=np.nan
    return narrow, broad

def stationary_ampl_adhoc(steps, w, ucoeff, th_known, th_next, Avec,x):
    '''
    Returns all amplitude stationary solutions for the set of thresholds. 
    This function is largely used to determine the set of thresholds iteratively based on 
    educated user choices/guesses. It is an alternative and slightly faster way of 
    obtaining a set of valid thresholds for any desired numbers of staircase steps.
    WARNING: this does not check for stability nor does it sort the trivial/narrow/broad solutions apart.
    Input: number of steps in firing rate, weight functions, coefficients of normalized activity profile
    (note: the first solution must be obtianed by the above to find these coefficients, after the first
    solution is obtained, subsequent activity profiles have been observed to be of the same shape with nearly
    identical normalized coefficients), a vector of known thresholds, 
    the next threshold to test, a vector of possible amplitudes, the spatial discretization
    Output: an array of stationary amplitudes (two new should arise for valid th_next's). If Avec contains 0,
    the trivial solution will also be returned before the lowest narrow/broad solutions. 
    If Avec does not include 0, then the first solution will be the lowest level narrow branch followed by the
    broad solution.
    '''
    thvec=np.hstack((th_known,th_next))
    expected_solutions= len(thvec)*2
    def func(amp):
        dum=-steps*amp/len(thvec)+np.trapz(w*firing(amp*sum([ucoeff[i]*np.cos(i*x) for i in range(len(ucoeff))]),thvec),x)
        return dum
    zeros=find_interfaces(Avec, np.array([func(A) for A in Avec]), 0)
    return zeros, np.array([func(A) for A in Avec])

def stability_matrix(state,steps,w_coeff,halfs):
    '''
    Builds the matrix used to check the stability of solutions.
    Input: the amplitude state (1,2,...,or 'steps'), number of steps in firing rate, 
    weight coefficients, a vector of halfwidths (which should all be positive!),
    Output: matrix of coefficients for checking stability.
    '''
    
    u_coeffs=[w_coeff[0]*(2/steps)*sum([h for h in halfs])]
    for i in range(1,len(w_coeff)):
        u_coeffs+=[sum([w_coeff[i]*(2/steps)*sum([np.sin(i*h) for h in halfs])/i])]
    
    u_derivs=[sum([u_coeffs[i]*i*np.sin(i*h) for i in range(1,len(w_coeff))])+10**(-4) for h in halfs]
    
    def w(x):
        return sum([w_coeff[i]*np.cos(i*x) for i in range(0,len(w_coeff))])
    
    def row(x):
        r=[]
        for i in range(state):
            r+=[w(x+halfs[i])/u_derivs[i],w(x-halfs[i])/u_derivs[i]]
        return np.array(r)
    
    M=np.array([row(-halfs[0]),row(halfs[0])])
    for i in range(1,len(halfs)):
        M=np.vstack((M,row(-halfs[i]),row(halfs[i])))
    
    return M/steps-np.identity(2*state)

def isStable(sol):
    '''
    Input: array of eigenvalues
    Output: -1 if all eigenvalues are stable and 
    +1 if any eigenvalue in unstable (real part is positive above the tolerance)
    Note: a tolerance is added since we expect to have eigenvalues of zero, numerically this can 
    translate to eigenvalues that are positive but very close to zero (ie 2*10^(-16))
    '''
    yn=-1
    for eig in sol:
        if np.real(eig)>10**(-6): #the tolerance added to not flag zero eigenvalues as unstable
            yn=1
    return yn 

def stability(state, steps, w_coeff, halfs):
    '''
    Checks stability of solutions.
    Input: the amplitude state (1,2,...,steps), number of steps in firing rate, 
    weight coefficients, a vector of halfwidths,
    Output: -1 for stable solutions, +1 for unstable solutions, np.nan for no/invalid solutions
    '''
    if np.any(np.isnan(halfs)):
        return np.empty(state)
    else:
        M=stability_matrix(state,steps,w_coeff,halfs)
        if np.any(np.isnan(halfs)) or np.any(np.isinf(halfs)):
            return np.nan
        else:
            eigvals,eigvec=la.eig(M)
            stable=isStable(eigvals)
            return stable

def circular_conv(signal, ker):
    '''
    signal: real 1D array
    ker: real 1D array
    signal and ker must have same shape
    '''
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))


def simulation(x,weight,I0,thu,
                e=0.001, 
                tcs=0, tce=10,
                Tend=100, disable_bars=False, **params):
    '''
    Runs a single simulation of the system
    Returns profiles at each time step
    '''
    
    dx=(x[2]-x[1])
    dt=0.025
    nt=int(Tend/dt)
    eps=e
    kernel=np.roll(weight,-int((len(x))/2))
    
    cuvec=np.zeros((len(x), nt))
    U=np.zeros(len(x))
    for i in tqdm(range(nt), disable=disable_bars):
        fireU=firing(U,thu)
        if (i<= tce/dt-1) and (i>=tcs/dt):
            U+=(dt*(-U+dx*circular_conv(fireU+I0,kernel)))
        else:
            U+=(dt*(-U+dx*circular_conv(fireU,kernel)))
        cuvec[:, i]=np.real(U)
    return cuvec

def coeff_ansatz_theory(x, w_coeff, I0, thu,
                e=0.001, 
                tcs=0, tce=10,
                Tend=100, **params):
    '''
    Runs a single simulation of the coefficient ansatz system. Note that the weight coefficients are utilized here and not the weight kernel. This is for the deterministic system (no noise) centered at 0.
    Returns a vector of the amplitudes and an array of the coefficients
    at each time step
    '''
    dt=0.025
    nt=int(Tend/dt)
    dx=x[2]-x[1]
    
    modes=len(w_coeff)
    U_coeff=np.zeros(modes)
    cos_array=np.array([np.cos(i*x) for i in range(modes)])
   
    Amp_vec=np.zeros(nt)
    for i in tqdm(range(nt)):
        dume=np.sum((U_coeff*cos_array.T).T, axis=0)
        fire=firing(dume,thu)

        if (i<= tce/dt-1) and (i>=tcs/dt):
            dum=(cos_array*(fire+I0))
            integral=dx*np.sum(dum[:,~np.all(dum== 0, axis=0)],axis=1)
            U_coeff+=dt*(w_coeff*integral-U_coeff)
        else:
            dum=(cos_array*(fire))
            integral=dx*np.sum(dum[:,~np.all(dum == 0, axis=0)],axis=1)
            U_coeff+=dt*(w_coeff*integral-U_coeff)
        Amp_vec[i]=np.sum(U_coeff)
    return Amp_vec, U_coeff

def ampl_ansatz_norm(x, weight, I0, thu, utilde,
                e=0.001, 
                tcs=0, tce=10,
                Tend=100, **params):
    '''
    Runs a single simulation of the amplitude ansatz system. Note the additional ucoeff input needed along with the weight kernel itself.
    Returns a vector of the amplitudes at each time step
    '''
    dt=0.025
    nt=int(Tend/dt)
    dx=x[2]-x[1]
    
    Amp_vec=np.zeros(nt)
    amp=0
    kernel=np.roll(weight,-int((len(x))/2))
    u_shape=np.copy(utilde)
    u_norm=np.trapz(utilde**2,x)
    Factor=dx*circular_conv(np.copy(u_shape),kernel)/u_norm
    Input=np.trapz(weight*I0,x)
    for i in tqdm(range(nt)):
        fire=firing(amp*u_shape,thu)

        if (i<= tce/dt-1) and (i>=tcs/dt):
            if amp < thu[1]:
                dum=(Factor*(fire))
                amp+=dt*(-amp+Input+dx*np.sum(dum[dum != 0]))
            else:
                dum=(Factor*(fire+I0))
                amp+=dt*(-amp+dx*np.sum(dum[dum != 0]))
        else:
            dum=(Factor*(fire))
            amp+=dt*(-amp+dx*np.sum(dum[dum != 0]))
        Amp_vec[i]=amp
    return Amp_vec

def simulation_noise(x, weight, I0, thu, noise_filter,
                     e=0.001, 
                     tcs=0, tce=10,
                     Tend=100, disable_bars=False, **params):
    '''
    Runs a single simulation of the system with noise. 
    Returns the activity profiles at each time step.
    '''
    
    dx=(x[2]-x[1])
    dt=0.025
    nt=int(Tend/dt)
    cuvec=np.zeros((len(x), nt))
    eps=e
    
    kernel=np.roll(weight,-int((len(x))/2))
    kernel_n=np.roll(noise_filter,-int((len(x))/2))
    U=np.zeros(len(x))
    for i in tqdm(range(nt),disable=disable_bars):
        W=circular_conv(kernel_n,rng.standard_normal(len(x)))*dx**0.5
        fireU=firing(U,thu)
        if (i<= tce/dt-1) and (i>=tcs/dt):
            U+=(dt*(-U+dx*circular_conv(fireU+I0,kernel)))
        else:
            U+=(dt*(-U+dx*circular_conv(fireU,kernel))+(np.sqrt(eps*dt)*W))
        cuvec[:, i]=np.real(U)
    return cuvec

def simulation_noise_special(x, weight, utilde, I0, thu, kernel_n, eps,iters,
                     e=0.001, 
                        tcs=0, tce=10, tds=15, tde=25,
                        Tend=100, disable_bars=False, **params):
    '''
    Runs a single simulation of the system
    Returns the six final values of the bump, facilitation, and depression centers of mass 
    '''
    
    dx=(x[2]-x[1])
    dt=0.025
    nt=int(Tend/dt)
    cuvec=np.zeros((len(x), nt))
    
    kernel=np.roll(weight,-int((len(x))/2))
    kernel_noise=np.roll(kernel_n,-int((len(x))/2))
    U=np.zeros(len(x))
    Amp_vec=np.zeros(nt)
    amp=0
    u_shape=np.copy(utilde)
    u_norm=np.trapz(utilde**2,x)
    Factor=dx*circular_conv(np.copy(u_shape),kernel)/u_norm
    Factor_n=dx*circular_conv(np.copy(u_shape),kernel_noise)/u_norm
    Noise=np.zeros((len(x),int(nt)))
    for j in range(int(nt)):
        Noise[:,j]=rng.standard_normal(len(x))
  
    for i in tqdm(range(nt),disable=disable_bars):
        fireU=firing(U,thu)
        fire=firing(amp*u_shape,thu)

        if len(I0[i])>10:
            U+=(dt*(-U+dx*circular_conv(fireU+I0[i],kernel)))
#             dum=(weight*(fire+I0[i]))
            amp=max(U)
            
        else:
            W=circular_conv(kernel_noise,np.copy(Noise[:,i]))*(dx**0.5)
            WA=np.trapz(np.roll(np.copy(u_shape),np.argmax(U)-2048)*np.copy(W),x)/u_norm
            dum=(Factor*(fire))
            amp+=dt*(-amp+dx*np.sum(dum[dum != 0]))+(np.sqrt(eps[i]*dt)*WA)
            U+=(dt*(-U+dx*circular_conv(fireU,kernel))+(np.sqrt(eps[i]*dt)*W))
        cuvec[:, i]=np.real(U)
        Amp_vec[i]=amp
        
    return cuvec, Amp_vec

def simulation_same_noise(x,weight,I0,thu, noise_filter, cue_vec,
                          e=0.001, 
                          tcs=0, tce=10,
                          Tend=100, disable_bars=False, **params):
    '''
    Runs a single simulation of the system
    Returns the six final values of the bump, facilitation, and depression centers of mass 
    '''
    
    dx=(x[2]-x[1])
    dt=0.025
    nt=int((cue_vec[-1]+200)/dt)
    cuvec=np.empty((len(x), nt, len(cue_vec)))
    eps=e
    
    kernel=np.roll(weight,-int((len(x))/2))
    kernel_n=np.roll(noise_filter,-int((len(x))/2))
    U=np.zeros(len(x))
    W=np.zeros((len(x),int((cue_vec[-1]+200)/dt)))
    for j in range(int((cue_vec[-1]+200)/dt)):
        W[:,j]=circular_conv(kernel_n,rng.standard_normal(len(x)))*dx**0.5

    for k, cue_dur in enumerate(cue_vec):
        tce=cue_dur
        Tend=cue_dur+200
        nt=int(Tend/dt)
        U=np.zeros(len(x))
        for i in tqdm(range(nt),disable=disable_bars):
            fireU=firing(U,thu)
            if (i<= tce/dt-1) and (i>=tcs/dt):
                U+=(dt*(-U+dx*circular_conv(fireU+I0,kernel)))
            else:
                U+=(dt*(-U+dx*circular_conv(fireU,kernel))+(np.sqrt(eps*dt)*W[:,i]))
            cuvec[:, i, k]=np.real(U)
    return cuvec

def centers_ampl_noise(x,weight,I0,thu, cue_vec, noise_filter,
                       e=0.001, 
                       tcs=0, tce=10,
                       Tend=100, **params):
    '''
    Runs a single simulation of the system
    Returns the six final values of the bump, facilitation, and depression centers of mass 
    '''
    
    dx=(x[2]-x[1])
    dt=0.025
    
    cuvec=np.zeros(len(cue_vec))
    eps=e
    
    kernel=np.roll(weight,-int((len(x))/2))
    kernel_n=np.roll(noise_filter,-int((len(x))/2))
    W=np.zeros((len(x),int((cue_vec[-1]+200)/dt)))
    for j in range(int((cue_vec[-1]+200)/dt)):
        W[:,j]=circular_conv(kernel_n,rng.standard_normal(len(x)))*dx**0.5

    for k, cue_dur in enumerate(cue_vec):
        tce=cue_dur
        Tend=cue_dur+200
        nt=int(Tend/dt)
        U=np.zeros(len(x))
        for i in range(nt):
            fireU=firing(U,thu)
            if (i<= tce/dt-1) and (i>=tcs/dt):
                U+=(dt*(-U+dx*circular_conv(fireU+I0,kernel)))
            else:
                U+=(dt*(-U+dx*circular_conv(fireU,kernel))+(np.sqrt(eps*dt)*W[:,i]))
        cuvec[k]=x[np.argmax(U)]
    return cuvec

def transition_time_sim(x,weight,U0,low,high,thu,maxend,e,tce,tcs,noise_filter):
    '''
    Runs a single simulation of the system
    Returns the six final values of the bump, facilitation, and depression centers of mass 
    '''
    
    dx=(x[2]-x[1])
    dt=0.025
    nt=int(maxend/dt)
    timey=np.linspace(0,maxend,nt)/100
    eps=e
    kernel=np.roll(weight,-int((len(x))/2))
    kernel_n=np.roll(noise_filter,-int((len(x))/2))
    transitiontime=np.nan
    U=np.copy(U0)
    tol=10**(-4)
    for i in range(nt):
        fireU=firing(U,thu)
        W=circular_conv(kernel_n,rng.standard_normal(len(x)))*dx**0.5
        U+=(dt*(-U+dx*circular_conv(fireU,kernel))+(np.sqrt(eps*dt)*W))
        currentAmpl=np.max(U)
        if (currentAmpl>high+tol) or (currentAmpl<low-tol):
            transitiontime=timey[i]
            break
    return transitiontime

def transitionEstimate_sim(x,weight,U0,low,high,thu,maxend, 
                           number,noise_filter, e=0.001, 
                           tcs=0, tce=10,
                           Tend=100, **params):
    transitions=[]
    n=0
    sims=0
    running=0
    print(e)
    while (n<number) and (sims<100) and (running<25):
        dum=transition_time_sim(x,weight,U0,low,high,thu,maxend,e,tce,tcs,noise_filter)
        if not(np.isnan(dum)):
            transitions+=[dum]
            n+=1
            running=0
        if np.isnan(dum):
            sims+=1
            running+=1
        print(str(n)+','+str(sims),end='\r')
    print('done')
    if len(transitions)!=number:
        transitions=np.nan*np.ones(number)
    return np.array(transitions)

#Transition time theory

def GA_vect(simresult,dx,w,thu):
    length=len(simresult[0,:])
    dum=[sum(firing(simresult[:,i],thu)*w)*dx for i in range(length)]
    return np.array(dum)

def GA_vect2(avec,ucoeff,x,w,thu):
    u_shape=sum([ucoeff[i]*np.cos(i*x) for i in range(len(ucoeff))])
    ga=[np.trapz(w*firing(amp*u_shape,thu),x) for amp in avec]
    return np.array(ga)

def probability_current(Ast, eps, kernel_n, Avec, gvector, low, x):
    vector=np.copy(Avec)
    sigma=eps*(np.trapz(kernel_n**2,x))
    def psi(A):
        integral=(2*np.trapz(-vector[vector<=A]+gvector[vector<=A],vector[vector<=A]))/sigma
        return np.exp(integral)
    dum=[psi(vector[i]) for i in range(len(vector))]
    return np.array(dum)/np.trapz(dum,vector)

def transitiontime_theory(Ast, eps, low, high, th_vec, DA, Avec, GAvec, kernel_n):
    vector=np.copy(Avec)
    gvector=np.copy(GAvec)
    sigma=eps*DA

    def psi(A):
        integral=(-A**2+low**2+2*np.trapz(gvector[vector<=A],vector[vector<=A]))/sigma
        return np.exp(integral)
    
    def int_psi_inverse(a,b):
        vec=np.copy(vector[(vector<=b) & (vector>=a)])
        dum=[1/psi(y) for y in vec]
        return np.trapz(dum,vec)
    
    def nested_integral1(xs,a,b):
        uppervec=np.copy(vector[(vector<=b) & (vector>=xs)])
        dum=[]
        for y in uppervec:
            innervec=np.copy(vector[(vector<=y) & (vector>=a)])
            dum+=[(1/psi(y))*np.trapz([psi(i) for i in innervec],innervec)]
        return np.trapz(dum,uppervec)
    
    def nested_integral2(xs,a,b):
        uppervec=np.copy(vector[(vector<=xs) & (vector>=a)])
        dum=[]
        for y in uppervec:
            innervec=np.copy(vector[(vector<=y) & (vector>=a)])
            dum+=[(1/psi(y))*np.trapz([psi(i) for i in innervec],innervec)]
        return np.trapz(dum,uppervec)
    
    numerator=(int_psi_inverse(low,Ast)*nested_integral1(Ast,low,high) 
               -int_psi_inverse(Ast,high)*nested_integral2(Ast,low,high))*2/sigma
    
    denominator=int_psi_inverse(low,high)*100
    #Returns time in seconds like the simulation results
    return numerator/denominator