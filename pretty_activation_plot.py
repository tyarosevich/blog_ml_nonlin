### This code was adapted from the very generously posted code here:
# https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/


import numpy as np
from scipy import integrate
from keras.layers.core import Dense
from keras.models import Sequential
import keras.backend as K
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation
from keras.callbacks import LambdaCallback

# Some figure settings.
sns.set(
        rc={
 'figure.facecolor': 'white',})
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})

# Builds 5 sequential models and saves the activation outputs.
outputs = []


for i in range(3):
    n=100
    model = Sequential()
    model.add(Dense(units=3, activation='relu', input_shape=(3,) ))
    for i in range(n):
        model.add(Dense(units=3, activation='sigmoid'))
    model.add(Dense(units=3, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    input = np.random.rand(1, 3) * 3
    targets = input ** 3 + input ** 4 - input
    model.fit(input, targets, epochs=3)

    current_outs = []
    for layer in model.layers:
        keras_function = K.function([model.input], [layer.output])
        current_outs.append(keras_function([input, 1]))

    xyz=current_outs[:-2]
    xyz= [list(x) for x in xyz]
    xyz = np.concatenate(xyz, axis=1)
    xyz = xyz[0, :, :]
    outputs.append(xyz)

x_t = np.asarray(outputs)
#%%
############# NOTE: the current saved mp4 is a good one to keep, since the training did flip to 1.0 accuracy
############# and had continual decrease in loss, so it represents something like typical 'learning'.
# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')
N_trajectories = 5
# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# prepare the axes limits
ax.set_xlim(0, 0.7)
ax.set_ylim(0, 0.7)
ax.set_zlim(0, 0.7)

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

def animate(i):

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, # init_func=init,
                               frames=100, interval=200, blit=True)


Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
anim.save('pretty_activations.mp4', writer=writer)