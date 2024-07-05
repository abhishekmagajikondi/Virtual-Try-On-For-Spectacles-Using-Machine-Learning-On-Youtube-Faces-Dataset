from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

def add_border_noise(points):
    # Calculate the convex hull to identify border points
      # Add noise to border points and create color array
    
    
    hull = ConvexHull(points)
    border_points = points[hull.vertices]

    # Generate noise (adjust noise_magnitude as needed)
    noise_magnitude = 0.05
    noise = np.random.normal(scale=noise_magnitude, size=border_points.shape)

    # Add noise to border points and create color array
    noisy_border_points = border_points + noise
    color = np.ones_like(points) * 4  # Initialize all points as blue (4 represents blue in Matplotlib)
    color[hull.vertices] = 2  # Set 

    # Replace border points with noisy points and update color
    points[hull.vertices] = noisy_border_points

    return points, color


def read_ply(file_path):
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
    return plydata['vertex']

def visualize_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = points['x']
    y = points['y']
    z = points['z']

    # Plot the point cloud in 3D
    ax.scatter(x, y, z, s=0.3, c='b', marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def main():
    file_path = "chair_1.ply"
    ply_data = read_ply(file_path)
    
    # Convert ply data to numpy array
    points = np.array([(point['x'], point['y'], point['z']) for point in ply_data])

    # Print the coordinate points
    for i, point in enumerate(points):
        print(f"Point {i + 1}: {point}")

    # Visualize point cloud in 3D
    visualize_point_cloud(ply_data)
    
     # Add noise to border points
    points = add_border_noise(points)

    # Visualize the point cloud with noise
    visualize_point_cloud(points)

if __name__ == "__main__":
    main()