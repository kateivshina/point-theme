---
title: Simulating a planetary system 
---

In this project, we write a two-body integrator using fourth-order Runge-Kutta (RK4) integration method. Additionally, we write a python routine to integrate the two-body system using the leapfrog method. We then study HAT-P-21 exoplanetary system, using our two-body integrator and FFT.

## Two-body integrator with RK4 method

First, we visit [this](http://openexoplanetcatalogue.com/planet/HAT-P-21%20b/) website to fetch the planet mass <img src="https://latex.codecogs.com/svg.latex?m_p" title="m_p" />, the orbital eccentricity <img src="https://latex.codecogs.com/svg.latex?e" title="e" />, the semi-major axis 
<img src="https://latex.codecogs.com/svg.latex?a" title="a" />, and the host star's mass <img src="https://latex.codecogs.com/svg.latex?m_s" title="m_s" />.

<img src="https://latex.codecogs.com/svg.latex?\frac{d\overline{v_i}}{dt}&space;=\sum_{i&space;\neq&space;j}^{N}&space;\frac{Gm_j&space;\overline{r_{ij}}}{r_{ij}^3}" title="\frac{d\overline{v_i}}{dt} =\sum_{i \neq j}^{N} \frac{Gm_j \overline{r_{ij}}}{r_{ij}^3}" />

 where <img src="https://latex.codecogs.com/svg.latex?\overline{r_{ij}}&space;=&space;\overline{r_j}-\overline{r_i}" title="\overline{r_{ij}} = \overline{r_j}-\overline{r_i}" /> 

```python 
import numpy as np

# computes gravitational acceleration of mass i due to all masses j != i
def compute_gravity(i,x,y,z):
  ax = 0 ; ay = 0 ; az = 0
  jlist = [k for k in range(nbody) if k != i]
  for j in jlist:                                   # loop over all masses j != i
    xsep = x[j]-x[i]                                # x_{ij} = x_j - x_i
    ysep = y[j]-y[i]                                # y_{ij} = y_j - y_i
    zsep = z[j]-z[i]                                # z_{ij} = z_j - z_i
    grav = m[j]*(xsep**2+ysep**2+zsep**2)**(-1.5)   # gravity on mass i due to mass j
    ax += grav*xsep                                 # x-component of gravity
    ay += grav*ysep                                 # y-component of gravity
    az += grav*zsep                                 # z-component of gravity
  return (ax,ay,az)                                 # return gravitational acceleration

# computes total (kinetic + potential) energy of all masses
def compute_energy(x,y,z,u,v,w):
  ke = 0. ; pe = 0.
  for i in range(nbody):
    ke += 0.5*m[i]*(u[i]**2+v[i]**2+w[i]**2)        # compute kinetic energy of mass i
    jlist = [k for k in range(nbody) if k > i]      # do sum over masses k > i
    for j in jlist:
      xsep = x[j]-x[i]                              # compute potential energy of mass i
      ysep = y[j]-y[i]
      zsep = z[j]-z[i]
      pe  -= m[i]*m[j]*(xsep**2+ysep**2+zsep**2)**(-0.5)
  return (ke+pe)/gconst                             # return total energy (in proper units)

# computes total angular momentum of all masses
def compute_angmom(x,y,z,u,v,w):
  L = 0.
  for i in range(nbody):
    L += m[i]*np.sqrt( (y[i]*w[i]-z[i]*v[i])**2     # compute angular momentum of mass i
                      +(z[i]*u[i]-x[i]*w[i])**2
                      +(x[i]*v[i]-y[i]*u[i])**2)
  return L/gconst                                   # return total ang. mom. (in proper units)

##############################################################################################
##############################################################################################

nbody = 2                # number of bodies
dt = 0.00001              # time step [yr]
tfinal = 2**17*dt        # final time [yr]

# gravitational constant in units where yr = au = Msun = 1
gconst = (2*np.pi)**2

# pre-declare arrays for speed
tsize = int(tfinal/dt)
t = np.zeros(tsize)         ; enrg = np.zeros(tsize)      ; angm = np.zeros(tsize)
x = np.zeros((tsize,nbody)) ; y = np.zeros((tsize,nbody)) ; z = np.zeros((tsize,nbody))
u = np.zeros((tsize,nbody)) ; v = np.zeros((tsize,nbody)) ; w = np.zeros((tsize,nbody))

# two-body planetary system
mp = 4.06 / 1047.592421 # planet mass in terms of solar masses
e = 0.228 # eccentricity
a = 0.0494 # semi-major axis in AU
ms = 0.947 # star mass in terms of solar masses
m = gconst*np.array([ms,mp])
r0 = a*(1-e*e)/(1+e)
v0 = np.sqrt((m[0]+m[1])*(2/r0-1/a))

# initial positions and velocities for each mass
i = 0
x[0,i] =  0.                ; y[0,i] =  0.                ; z[0,i] =  0.
u[0,i] =  0.                ; v[0,i] =  0.                ; w[0,i] =  0.
i += 1
x[0,i] =  r0                ; y[0,i] =  0.                ; z[0,i] =  0.
u[0,i] =  0.                ; v[0,i] =  v0                ; w[0,i] =  0.

# compute initial total energy of system
enrg[0] = compute_energy(x[0,:],y[0,:],z[0,:],u[0,:],v[0,:],w[0,:])

# compute initial angular momentum of system
angm[0] = compute_angmom(x[0,:],y[0,:],z[0,:],u[0,:],v[0,:],w[0,:])

# declare temporary storage arrays for RK integration
k1 = np.zeros((nbody,6)) ; k2 = np.zeros((nbody,6)); k3 = np.zeros((nbody,6)); k4 = np.zeros((nbody,6))
ix = 0 ; iy = 1 ; iz = 2 ; iu = 3 ; iv = 4 ; iw = 5

# begin RK4 integration
for n in range(tsize-1):                         # loop over times
  
  # stage 1
  for i in range(nbody):                         # loop over masses i=0,1,...,(nbody-1)
    # compute gravitational acceleration of mass i due to all masses j != i
    (ax,ay,az) = compute_gravity(i,x[n,:],y[n,:],z[n,:])
    # store k1
    k1[i,ix] = dt*u[n,i]
    k1[i,iy] = dt*v[n,i]
    k1[i,iz] = dt*w[n,i]
    k1[i,iu] = dt*ax
    k1[i,iv] = dt*ay
    k1[i,iw] = dt*az
  
  # stage 2
  for i in range(nbody):                         # loop over masses i=0,1,...,(nbody-1)
    # compute gravitational acceleration of mass i due to all masses j != i
    (ax,ay,az) = compute_gravity(i,x[n,:]+0.5*k1[:,ix],y[n,:]+0.5*k1[:,iy],z[n,:]+0.5*k1[:,iz])
    # store k2
    k2[i,ix] = dt*(u[n,i]+0.5*k1[i,iu])
    k2[i,iy] = dt*(v[n,i]+0.5*k1[i,iv])
    k2[i,iz] = dt*(w[n,i]+0.5*k1[i,iw])
    k2[i,iu] = dt*ax
    k2[i,iv] = dt*ay
    k2[i,iw] = dt*az


  # stage 3
  for i in range(nbody):                         # loop over masses i=0,1,...,(nbody-1)
    # compute gravitational acceleration of mass i due to all masses j != i
    (ax,ay,az) = compute_gravity(i,x[n,:]+0.5*k2[:,ix],y[n,:]+0.5*k2[:,iy],z[n,:]+0.5*k2[:,iz])
    # store k3
    k3[i,ix] = dt*(u[n,i]+0.5*k2[i,iu])
    k3[i,iy] = dt*(v[n,i]+0.5*k2[i,iv])
    k3[i,iz] = dt*(w[n,i]+0.5*k2[i,iw])
    k3[i,iu] = dt*ax
    k3[i,iv] = dt*ay
    k3[i,iw] = dt*az

  # stage 4
  for i in range(nbody):                         # loop over masses i=0,1,...,(nbody-1)
    # compute gravitational acceleration of mass i due to all masses j != i
    (ax,ay,az) = compute_gravity(i,x[n,:]+k3[:,ix],y[n,:]+k3[:,iy],z[n,:]+k3[:,iz])
    # store k4
    k4[i,ix] = dt*(u[n,i]+k3[i,iu])
    k4[i,iy] = dt*(v[n,i]+k3[i,iv])
    k4[i,iz] = dt*(w[n,i]+k3[i,iw])
    k4[i,iu] = dt*ax
    k4[i,iv] = dt*ay
    k4[i,iw] = dt*az

  # update position and velocity of mass i
  for i in range(nbody):
    x[n+1,i] = x[n,i] + 1/6 * (k1[i,ix] + 2*k2[i,ix] + 2*k3[i,ix] + k4[i,ix])
    y[n+1,i] = y[n,i] + 1/6 * (k1[i,iy] + 2*k2[i,iy] + 2*k3[i,iy] + k4[i,iy])
    z[n+1,i] = z[n,i] + 1/6 * (k1[i,iz] + 2*k2[i,iz] + 2*k3[i,iz] + k4[i,iz])
    u[n+1,i] = u[n,i] + 1/6 * (k1[i,iu] + 2*k2[i,iu] + 2*k3[i,iu] + k4[i,iu])
    v[n+1,i] = v[n,i] + 1/6 * (k1[i,iv] + 2*k2[i,iv] + 2*k3[i,iv] + k4[i,iv])
    w[n+1,i] = w[n,i] + 1/6 * (k1[i,iw] + 2*k2[i,iw] + 2*k3[i,iw] + k4[i,iw])

  # increment time
  t[n+1] = t[n] + dt

  # compute total energy of system
  enrg[n+1] = compute_energy(x[n+1,:],y[n+1,:],z[n+1,:],u[n+1,:],v[n+1,:],w[n+1,:])

  # compute angular momentum of system
  angm[n+1] = compute_angmom(x[n+1,:],y[n+1,:],z[n+1,:],u[n+1,:],v[n+1,:],w[n+1,:])

 
``` 
 
 | ![name](/files/simulating-a-planetary-system/rk4.png) | 
 |:--:| 
 | HAT-P-21b |
 
 

## Two-body integrator with leapfrog method



## FFT’ing a planetary system.
