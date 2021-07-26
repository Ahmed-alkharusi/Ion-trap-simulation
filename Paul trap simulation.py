"""
A simulation of the motion of a charged particle in an idealized Paul trap trap.

Ahmed Al-kharusi

Please check the simulation yourself. You may find mistakes!
See the references
"""





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#initial conditios
position_velocity_x = np.array([0, 5])
position_velocity_y = np.array([5, 0])
position_velocity_z = np.array([0, 1])

a_x = 0.15 #a_x =a_y
q_x = 0.45 #q_x =q_y

a_z = -2*a_x
q_z = -2 *q_x

omega = np.pi

# see first reference to understand these
def derivatives_x(position_velocity_x, t):
    return np.array([position_velocity_x[1], -0.25*pow(omega, 2)*(a_x-2*q_x*np.cos(omega*t))*position_velocity_x[0]])

def derivatives_y(position_velocity_y, t):
    return np.array([position_velocity_y[1],-0.25*pow(omega, 2)*(a_x-2*q_x*np.cos(omega*t))*position_velocity_y[0]])

def derivatives_z(position_velocity_z, t):
    return np.array([position_velocity_z[1], -0.25*pow(omega, 2)*(a_z-2*q_z*np.cos(omega*t))*position_velocity_z[0]])



#taken from... (see references)
def rk4(y,dy,t,h):
    k1=dy(y,t)
    k2=dy(y+h/2*k1,t+h/2)
    k3=dy(y+h/2*k2,t+h/2)
    k4=dy(y+h*k3,t+h)
    y=y+h*(k1+2*k2+2*k3+k4)/6
    t=t+h
    return (t,y)


def implement_rk4(position_velocity_x, position_velocity_y, position_velocity_z, t, h, steps_no):
    global time_arr
    time_arr = np.array([t])
    data_x = np.array([position_velocity_x])
    data_y = np.array([position_velocity_y])
    data_z = np.array([position_velocity_z])


    for i in range(steps_no):

        (t, position_velocity_x) = rk4( position_velocity_x, derivatives_x, t,h)
        t -=h
        (t, position_velocity_y) = rk4( position_velocity_y, derivatives_y,  t,h)
        t -=h
        (t, position_velocity_z) = rk4(position_velocity_z, derivatives_z,  t,h)

        #time_arr = np.append(time_arr, t)
        data_x = np.vstack((data_x, position_velocity_x))
        data_y = np.vstack((data_y, position_velocity_y))
        data_z = np.vstack((data_z, position_velocity_z))

        time_arr = np.append(time_arr, t)
        #print (star2_arr_x," ", star1_arr_x)
    [x, vx] = data_x.transpose()
    [y, vy] = data_y.transpose()
    [z, zy] = data_z.transpose()


    return [x ,y, z] # can also return vx, vy, vz for energies and othe info


t = 0 # starting time
h = 1/(100) # step size for the RK4 method
steps_no = 50000 # number of steps of the RK4 method

[x, y, z] = implement_rk4(position_velocity_x, position_velocity_y, position_velocity_z, t, h, steps_no)


save_every_n_frames = 20


(ymin, ymax) = (-7, 7)
(xmin, xmax) = (-7, 7)
(zmin, zmax) = (-7, 7)

"""
for j in range(int(len(x)/save_every_n_frames)-1):
    i = j*save_every_n_frames
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca(projection='3d', autoscale_on=False,xlim=(xmin,xmax), ylim=(ymin,ymax), zlim=(zmin,zmax))

    for t in range(j):
        ax.scatter(x[save_every_n_frames*t], y[save_every_n_frames*t], z[save_every_n_frames*t], s=2, c='black', alpha=0.9)

    ax.scatter(x[i], y[i], z[i], s=200)


    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$y$', fontsize=30)
    ax.set_zlabel('$z$', fontsize=30)
    plt.tight_layout()
    fig.savefig(str(j)+'.png',dpi=150)
    plt.show()
    plt.close()
"""

"""
References:
#The rk4 function is taken from
https://youtu.be/HPreOWKJOiY


"""
