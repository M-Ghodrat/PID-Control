import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu

def initialize(*, r, x0, x0_dot):

    T = 15
    dt = 0.01
    N = int(T/dt)

    c = 1
    m = 1 
    k = 1

    x = np.zeros((2,N))
    y = np.zeros((1,N))
    d = np.zeros((1,N))
    u = np.zeros((1,N))
    r = np.zeros((1,N)) + r
    e = np.zeros((1,N))
    
    x[:,0] = [x0,x0_dot]
    y[0,0] = x[0,0]
    
    return T,dt,N,c,m,k,x,y,d,u,r,e

def run(*, kp, ki, kd, D=0, model='Linear'):
    
    int_e = 0
    
    def disturbance(i):
        return D*np.sin(2*i*dt)
    
    # solve dx/dt = f(x) + U + D
    def Euler(x, u, d, model):
        if model == 'Linear':
            f1 = -c/m*x[0] + x[1]
            f2 = -k/m*x[0]
        else:
            f1 = -c/m*x[0] + x[1]
            f2 = -k/m*x[0]**2*x[1]*0.01
        x_updated = np.array([f1,f2+u+d])*dt + x[:]
        return x_updated, x_updated[0] 
        
    def PID(r, y, e_old, int_e):
        e_now = r - y
        if i > 0:
            derivative = (e_now-e_old)/dt
        else:
            derivative = 0
        u_ = kp * e_now + ki*(e_now*dt+int_e) + kd*derivative
        int_e += e_now*dt
        return e_now, u_, int_e
        
    for i in range(N-1):
        d[0,i] = disturbance(i)
        e[0,i], u[0,i], int_e = PID(r[0,i], y[0,i], e[0,i-1], int_e)
        x[:,i+1], y[0,i+1] = Euler(x[:,i], u[0,i], d[0,i], model)
    d[0,i+1] = disturbance(i+1)
    
    fig = plt.figure(figsize=(15,7))
    plt.plot(np.arange(N)*dt, y[0,:], color='#F1A218', label='System Output')
    plt.plot(np.arange(N)*dt, r[0,:], linestyle='dashed', label='Reference Input')
    plt.legend()
    plt.grid()
    plt.ylim([-5,5])
    st.pyplot(fig)

with st.sidebar:
    selected = option_menu("PID Controller Simulator", ["Home", 'Settings', 'Simulation'], 
        icons=['house-fill', 'sliders', 'bar-chart-fill'], menu_icon="cast", default_index=1)

if selected == 'Home':
    st.header('Home')

if selected == 'Settings':
    st.header('Settings')

if selected == 'Simulation':
    st.header('Simulation')
    
    r = st.slider('Reference Input', -5.0, 5.0, 0.5)
    D = st.slider('Sinusoidal Disturbance Amplitude', 0.0, 5.0, 0.0)
    model = st.radio(
     "Choose a Model",
     ('Linear', 'Nonlinear'))

    x0 = st.slider('x0', 0.0, 5.0, 2.0)
    x0_dot = st.slider('x0_dot', 0.0, 5.0, -3.0)

    kp = st.slider('Kp', 0.0, 50.0, 0.0)
    ki = st.slider('Ki', 0.0, 50.0, 0.0)
    kd = st.slider('Kd', 0.0, 50.0, 0.0)
    

    T,dt,N,c,m,k,x,y,d,u,r,e = initialize(r=r, x0=x0, x0_dot=x0_dot)
    run(kp=kp, ki=ki, kd=kd, D=D, model=model)


with st.sidebar:
    if selected == 'Simulation':
        st.subheader('Disturbance')
        fig = plt.figure(figsize=(7,4))
        plt.plot(np.arange(N)*dt, d[0,:])
        plt.grid()
        plt.ylim([-5,5])
        st.pyplot(fig)