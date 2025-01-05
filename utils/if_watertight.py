import os
import re

def parse_obj(filename):
    vertices = []
    faces = []
    edges = {}

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):  # Face
                face = [int(face.split('/')[0]) for face in line.strip().split()[1:]]
                faces.append(face)

                # Update edge counts
                num_vertices = len(face)
                for i in range(num_vertices):
                    v1, v2 = face[i], face[(i + 1) % num_vertices]
                    edge = tuple(sorted([v1, v2]))
                    if edge in edges:
                        edges[edge] += 1
                    else:
                        edges[edge] = 1

    return vertices, faces, edges

def check_watertight(edges):
    non_manifold_edges = [edge for edge, count in edges.items() if count != 2]
    return len(non_manifold_edges) == 0, non_manifold_edges

def process_obj_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.obj'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            vertices, faces, edges = parse_obj(file_path)
            is_watertight, non_manifold_edges = check_watertight(edges)

            if is_watertight:
                print(f"The model {filename} appears to be watertight.\n")
            else:
                print(f"The model {filename} is not watertight. Non-manifold edges found:")
                for edge in non_manifold_edges:
                    print(edge)
                print("\n")

def main():
    folder_path = '/home/wzj/PycharmProjects/sphere_resconstruct/complex_org'  # Replace with your folder path
    process_obj_files(folder_path)

if __name__ == '__main__':
    main()
