import matplotlib.pyplot as plt
import numpy as np
from csv import reader
from sys import argv

x = []
y = []
z = []
number = 20
cmap = plt.get_cmap('gnuplot')

colors = np.random.rand(30,3)

for line in open(argv[1], 'r'):
    lines = [i for i in line.split()]
    x.append(lines[0])
    y.append(int(lines[1]))
    index = int(lines[2])
    z.append(colors[index])

fig = plt.figure()
plt.scatter(x, y, c=z)
plt.xlabel("x-axis")
plt.ylabel("y-axis")               
plt.show()


