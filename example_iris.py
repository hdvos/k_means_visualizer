from numpy.lib.npyio import load
from kmeans_plotter import k_means_gif

from sklearn.datasets import load_iris
data = load_iris()

iris_data = data.data 

k_means_gif(3, iris_data, 'figures_examples/iris_k3.gif')