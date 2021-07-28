"""
Demonstration of Earnshaw's theorem

Ahmed Al-kharusi
Please check the simulation yourself. You may find mistakes!
See the references

"""




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#The moving charge
position_velocity_xi = np.array([0, 0])
position_velocity_yi = np.array([-0.01, 0])
position_velocity_zi = np.array([0.01,0])

#The fixed charges
CHARGE1_POSITION = np.array([1, 1, 1])
CHARGE2_POSITION = np.array([1, 1, -1])
CHARGE3_POSITION = np.array([1, -1, 1])
CHARGE4_POSITION = np.array([-1, 1, 1])
CHARGE5_POSITION = np.array([1, -1, -1])
CHARGE6_POSITION = np.array([-1, -1, 1])
CHARGE7_POSITION = np.array([-1, 1, -1])
CHARGE8_POSITION = np.array([-1, -1, -1])

#Coulomb constant arb. units
k = 1
# set all charges =+1 and masses =1



#consider removing place holders for more efficient codes
def acceleration(position_velocity, axis , position_velocity_x, position_velocity_y,position_velocity_z):
    d1 = (position_velocity[0]-CHARGE1_POSITION[axis])
    d2 = (position_velocity[0]-CHARGE2_POSITION[axis])
    d3 = (position_velocity[0]-CHARGE3_POSITION[axis])
    d4 = (position_velocity[0]-CHARGE4_POSITION[axis])
    d5 = (position_velocity[0]-CHARGE5_POSITION[axis])
    d6 = (position_velocity[0]-CHARGE6_POSITION[axis])
    d7 = (position_velocity[0]-CHARGE7_POSITION[axis])
    d8 = (position_velocity[0]-CHARGE8_POSITION[axis])

    r1 = np.linalg.norm(np.array([CHARGE1_POSITION[0]-position_velocity_x[0], CHARGE1_POSITION[1]-position_velocity_y[0], CHARGE1_POSITION[2]-position_velocity_z[0]]))
    r2 = np.linalg.norm(np.array([CHARGE2_POSITION[0]-position_velocity_x[0], CHARGE2_POSITION[1]-position_velocity_y[0], CHARGE2_POSITION[2]-position_velocity_z[0]]))
    r3 = np.linalg.norm(np.array([CHARGE3_POSITION[0]-position_velocity_x[0], CHARGE3_POSITION[1]-position_velocity_y[0], CHARGE3_POSITION[2]-position_velocity_z[0]]))
    r4 = np.linalg.norm(np.array([CHARGE4_POSITION[0]-position_velocity_x[0], CHARGE4_POSITION[1]-position_velocity_y[0], CHARGE4_POSITION[2]-position_velocity_z[0]]))
    r5 = np.linalg.norm(np.array([CHARGE5_POSITION[0]-position_velocity_x[0], CHARGE5_POSITION[1]-position_velocity_y[0], CHARGE5_POSITION[2]-position_velocity_z[0]]))
    r6 = np.linalg.norm(np.array([CHARGE6_POSITION[0]-position_velocity_x[0], CHARGE6_POSITION[1]-position_velocity_y[0], CHARGE6_POSITION[2]-position_velocity_z[0]]))
    r7 = np.linalg.norm(np.array([CHARGE7_POSITION[0]-position_velocity_x[0], CHARGE7_POSITION[1]-position_velocity_y[0], CHARGE7_POSITION[2]-position_velocity_z[0]]))
    r8 = np.linalg.norm(np.array([CHARGE8_POSITION[0]-position_velocity_x[0], CHARGE8_POSITION[1]-position_velocity_y[0], CHARGE8_POSITION[2]-position_velocity_z[0]]))

    return k*(d1/r1**3+d2/r2**3+d3/r3**3+d4/r4**3+d5/r5**3+d6/r6**3+d7/r7**3+d8/r8**3)

def derivatives_x(position_velocity_x, t, x,y,z):
    return np.array([position_velocity_x[1], acceleration(position_velocity_x, 0, x,y,z) ])

def derivatives_y(position_velocity_y, t, x,y,z):
    return np.array([position_velocity_y[1], acceleration(position_velocity_y, 1, x,y,z)])
                                    #place_holder so that it can be used with rk4
def derivatives_z(position_velocity_z, t, x,y,z):
    return np.array([position_velocity_z[1], acceleration(position_velocity_z, 2, x,y,z)])

#taken from... (see references)
def rk4(y,dy,t,h, x,yy,z):
    k1=dy(y,t, x,yy,z)
    k2=dy(y+h/2*k1,t+h/2, x,yy,z)
    k3=dy(y+h/2*k2,t+h/2, x,yy,z)
    k4=dy(y+h*k3,t+h, x,yy,z)
    y=y+h*(k1+2*k2+2*k3+k4)/6
    #t=t+h
    return y



def implement_rk4(position_velocity_x, position_velocity_y, position_velocity_z, t, h, steps_no):
    global time_arr
    time_arr = np.array([t])
    data_x = np.array([position_velocity_x])
    data_y = np.array([position_velocity_y])
    data_z = np.array([position_velocity_z])


    for i in range(steps_no):
        tempx = position_velocity_x
        tempy = position_velocity_y
        

        position_velocity_x = rk4( position_velocity_x, derivatives_x, t, h ,position_velocity_x, position_velocity_y, position_velocity_z)
        position_velocity_y = rk4( position_velocity_y, derivatives_y,  t,h, tempx, position_velocity_y, position_velocity_z)
        position_velocity_z = rk4( position_velocity_z, derivatives_z,  t,h, tempx, tempy, position_velocity_z)
        t +=h

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
h = 1/(30) # step size for the RK4 method
steps_no = 40000 # number of steps of the RK4 method

[x, y, z] = implement_rk4(position_velocity_xi, position_velocity_yi, position_velocity_zi, t, h, steps_no)


save_every_n_frames = 30


(ymin, ymax) = (-1.1, 1.1)
(xmin, xmax) = (-1.1, 1.1)
(zmin, zmax) = (-1.1, 1.1)



for j in range(int(len(x)/save_every_n_frames)-1):
    i = j*save_every_n_frames
    fig = plt.figure(figsize=(10,10))

    ax = fig.gca(projection='3d', autoscale_on=False,xlim=(xmin,xmax), ylim=(ymin,ymax), zlim=(zmin,zmax))


    ax.scatter(x[0:i], y[0:i], z[0:i], s=2, c='black', alpha=0.9)

    ax.scatter(x[i], y[i], z[i], s=200, c='c', label='Free charge')

    ax.set_title("Free charge position:("+str(round(x[i],2))+", "+str(round(y[i],2))+", "+str(round(z[i],2))+")\n Add a small offset"#'Add a small offset\n'
                 , c='blue',fontsize=14,loc='left' )

    ax.scatter(CHARGE1_POSITION[0], CHARGE1_POSITION[1], CHARGE1_POSITION[2] , s=200, c='r', label="Fixed charges\nat cube vertices")
    ax.scatter(CHARGE2_POSITION[0], CHARGE2_POSITION[1], CHARGE2_POSITION[2] , s=200, c='r')
    ax.scatter(CHARGE3_POSITION[0], CHARGE3_POSITION[1], CHARGE3_POSITION[2] , s=200, c='r')
    ax.scatter(CHARGE4_POSITION[0], CHARGE4_POSITION[1], CHARGE4_POSITION[2] , s=200, c='r')
    ax.scatter(CHARGE5_POSITION[0], CHARGE5_POSITION[1], CHARGE5_POSITION[2] , s=200, c='r')
    ax.scatter(CHARGE6_POSITION[0], CHARGE6_POSITION[1], CHARGE6_POSITION[2] , s=200, c='r')
    ax.scatter(CHARGE7_POSITION[0], CHARGE7_POSITION[1], CHARGE7_POSITION[2] , s=200, c='r')
    ax.scatter(CHARGE8_POSITION[0], CHARGE8_POSITION[1], CHARGE8_POSITION[2] , s=200, c='r')

    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$y$', fontsize=30)
    ax.set_zlabel('$z$', fontsize=30)
    ax.legend(fontsize=18)
    plt.tight_layout()
    fig.savefig(str(j)+'.png',dpi=110)
    plt.show()
    plt.close()



"""
References:
#The rk4 function is taken from
https://youtu.be/HPreOWKJOiY
"""
