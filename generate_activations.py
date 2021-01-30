
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dense
from keras.models import Sequential, load_model
from keras import Input
import keras.backend as K
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import seaborn as sns
#%%
sns.set(font='Franklin Gothic Book',
        rc={
 'axes.axisbelow': False,
 'axes.edgecolor': 'lightgrey',
 'axes.facecolor':'None',
 'axes.grid': False,
 'axes.labelcolor': 'black',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'figure.facecolor': 'grey',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'black',
 'xtick.bottom': False,
 'xtick.color': 'black',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'black',
 'ytick.direction': 'out',
 'ytick.left': False,
 'ytick.right': False})
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})


#%%
outputs = []
for i in range(5):
    n=100
    model = Sequential()
    model.add(Dense(units=3, activation='relu', input_shape=(3,) ))
    for i in range(n):
        model.add(Dense(units=3, activation='sigmoid'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    input=np.random.rand(1,3) * 10
    dummy_train= np.ndarray([1])

    model.fit(input, dummy_train, epochs=1)

    current_outs = []
    for layer in model.layers:
        keras_function = K.function([model.input], [layer.output])
        current_outs.append(keras_function([input, 1]))

    xyz=current_outs[:-2]
    xyz= [list(x) for x in xyz]
    xyz = np.concatenate(xyz, axis=1)
    xyz = xyz[0, :, :]
    outputs.append(xyz)


# %%
mat = np.dstack(tuple(outputs))

#%%

test = data[:,:,1]
#%%
# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
# %%
t = np.arange(0,100)
# df = pd.DataFrame({"time": t ,"x" : mat[:,0, :], "y" : mat[:,1, :], "z" : mat[:,2, :]})

def update_graph(i):
    # data=df[df['time']==num]
    graph.set_data(mat[i, 0, :], mat[i, 1, :])
    graph.set_3d_properties(mat[i, 2,:],'z')
    fig.suptitle('3D Test, time={}'.format(i))
    return graph


fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim(0,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_ylim(0,1)
ax.set_zlim(0,1)
title = fig.suptitle('3D Test')

# data=df[df['time']==0]
graph, = ax.plot(mat[0, 0, :], mat[0, 1, :], mat[0, 2, :], linestyle="", marker="o")

ani = animation.FuncAnimation(fig, update_graph,
                               interval=1000, blit=False)

ani.save('im.mp4', writer=writer)



