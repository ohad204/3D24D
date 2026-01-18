import numpy as np
import yaml
from pytorch3d.io import load_objs_as_meshes, IO

def load_object(image_path):
    # TODO: need to be changed to load object
    # load object
    if image_path.endswith(".obj"):
        mesh = load_objs_as_meshes([image_path])
    if image_path.endswith(".glb"):
        from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        mesh = io.load_mesh(image_path)

    return mesh

def main():
    file_path = '/home/lab/rahamiob/ohadr/projects/object_animation/custom/scripts/objects.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # objects = data['stage2-evolution-unique']
    objects = data['stage2-objaverse-test']

    vertices = []
    for obj in objects:
        mesh = load_object(obj[0])
        vertices.append(mesh.verts_packed().shape[0])

    print(np.array(vertices).mean())

if __name__ == '__main__':
    main()