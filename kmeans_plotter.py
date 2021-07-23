
from bokeh.io import export_png
from bokeh.palettes import viridis
from bokeh.plotting import figure, output_file, show

import imageio

import numpy as np

import os

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from tempfile import TemporaryDirectory

from tqdm import tqdm

def initialize_centroids(k:int, data:np.ndarray, seed:int=42) -> np.ndarray:
    np.random.seed(seed)
    dimensions = data.shape[1]
    
    centroids = np.zeros((k, dimensions))
    
    for i in range(dimensions):
        max_val = data[:,i].max()
        min_val = data[:,i].min()
        val_range = max_val-min_val
#         print(f"min: {min_val} - max: {max_val} - val range {val_range}")
        
        column = min_val + np.random.rand(k)*val_range
        
        centroids[:,i] = column
        
#     print(centroids)
    return centroids

def initialize_centroids_by_sample(k:int, data:np.ndarray, seed:int=42) -> np.ndarray:
    np.random.seed(seed)
    dimensions = data.shape[1]
    
    number_of_rows = data.shape[0]
    random_indices = np.random.choice(number_of_rows, 
                                  size=k, 
                                  replace=False)
    

#     print(rows_id)
    centroids = data[random_indices, :]
        
#     print(centroids)
#     input()
    return centroids


def re_calculate_centroids(data, clusters, k, seed:int=42):
    np.random.seed(seed)
    centroids = np.zeros((k, data.shape[1]))
    for cluster in range(k):
        selection = data[clusters == cluster]
        if selection.shape[0] == 0:
            centroid = data.mean(axis=0) + np.random.normal(0, data.mean(axis=0)/2, data.shape[1])
        else:
            centroid = selection.mean(axis = 0)
            centroids[cluster] = centroid
#     print(centroids)
    return centroids

def add_centroids_trace(p, centroids_trace, unique_labels, colormap):
    tracesx = {label: [] for label in unique_labels}
    tracesy = {label: [] for label in unique_labels}
    
    for centroids_2d in centroids_trace:
#         centroids_2d = pca.transform(centroids)
        
        for i, centroid in enumerate(centroids_2d):
            tracesx[i].append(centroid[0])
            tracesy[i].append(centroid[1])
        
    
    for label in unique_labels:
        p.line(tracesx[label], tracesy[label], color=colormap[label])


        
def make_clusters(data, centroids):
    
    distances = pairwise_distances(data, centroids, metric='euclidean')
#     print(distances)
    clusters = np.argmin(distances, axis=1)
#     print(clusters)
    return clusters


def centroids_changed(centroids_a, centroids_b, allowance):
#     print(centroids_a)
#     print(centroids_b)
    assert centroids_a.shape == centroids_b.shape
    distances = pairwise_distances(centroids_a, centroids_b, metric='euclidean')

    diagonal = np.diagonal(distances)

    if diagonal.sum() > allowance:
        return True
    else:
        return False

def make_first_plot(data_2d, img_dir, img_count):
    p = figure(title = f"Step {img_count:03d}")
    p.xaxis.axis_label = 'PC 1'
    p.yaxis.axis_label = 'PC 2'

    p.circle(data_2d[:,0], data_2d[:,1], color = 'black')
    
    outfilename = os.path.join(img_dir, f"{img_count:03d}.png")
    export_png(p, filename=outfilename)
    
    
def make_other_plots(data_2d, clusters, centroids_trace, img_dir, img_count, add_trace=True):
    centroids_2d = centroids_trace[-1]
#     print(centroids_2d)
    unique_labels = [x for x in range(0,centroids_2d.shape[0] + 1)]
#     print(unique_labels)

    nr_labels = len(unique_labels)
    available_colors = viridis(nr_labels)
    
    colormap = {label:color for label, color in zip(unique_labels, available_colors)}

    colors = [colormap[x] for x in clusters]
    centroids_colors = [colormap[x] for x in range(centroids_2d.shape[0])]
    
    
    p = figure(title = f"Step {img_count:03d}")
    p.xaxis.axis_label = 'PC 1'
    p.yaxis.axis_label = 'PC 2'

    p.circle(data_2d[:,0], data_2d[:,1], color = colors)
    p.circle(centroids_2d[:,0], centroids_2d[:,1], fill_color=centroids_colors, line_color="white", size= 10)
    
    if add_trace:
        add_centroids_trace(p, centroids_trace, unique_labels, colormap)
#     output_file("iris.html", title="iris.py example")

    outfilename = os.path.join(img_dir, f"{img_count:03d}.png")
    export_png(p, filename=outfilename)
    

    
# https://stackoverflow.com/questions/41228209/making-gif-from-images-using-imageio-in-python
def make_gif(gif_name, folder, fps=2):
    png_dir = folder
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_name, images, fps = fps)
    print(f"gif saved at: {gif_name}")

def k_means_gif(k, data, filename, add_trace=True, max_it=100, max_difference = 0.001, seed=42):
    img_count = 1   #TODO: img count as generator?
    with TemporaryDirectory() as tmpdirname:
        pca = PCA(2)
        data_2d = pca.fit_transform(data)
        
        make_first_plot(data_2d, tmpdirname, img_count)
        img_count += 1
        
        print("start k means")
        centroids_trace = []
        centroids = initialize_centroids_by_sample(k, data, seed=seed)
        
        
        for _ in tqdm(range(max_it)):
            clusters = make_clusters(data, centroids)

            centroids_2d = pca.transform(centroids)

            centroids_trace.append(centroids_2d)
            
            make_other_plots(data_2d, clusters, centroids_trace, tmpdirname, img_count, add_trace=add_trace)
            img_count += 1
        
            new_centroids = re_calculate_centroids(data, clusters, k)
            if not centroids_changed(centroids, new_centroids, max_difference):
                print("Exit: convergence")
                break
            else:
                centroids = new_centroids
        
        make_other_plots(data_2d, clusters, centroids_trace, tmpdirname, img_count)
        img_count += 1
        
        print("make gif")
        make_gif(filename, tmpdirname)