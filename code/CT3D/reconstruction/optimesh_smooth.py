"""
Generate animation for fixed-point Laplace. Not included as a command-line option as the
lienar solve is much more efficient.
"""
import meshio
import meshplex
import optimesh
import argparse


parser = argparse.ArgumentParser(description = 'CT reconstruction')

parser.add_argument('-in', '--input_dir', default = '', help='Input directory for PNG files')
parser.add_argument('-out', '--output_dir', default = '', help='Output directory for RAW file')

args = parser.parse_args()

mesh = meshio.read(args.input_dir,file_format="stl")
print(mesh.cells)
mesh.write(args.output_dir)

mesh = meshio.read(args.output_dir,file_format="vtk")
# points, cells = optimesh.optimize_points_cells(
#     mesh.points, mesh.cells, "CVT (block-diagonal)", 1.0e-5, 100
# )
mesh = meshplex.MeshTri(mesh.points, mesh.cells)
optimesh.optimize(mesh, "CVT (block-diagonal)", 1.0e-5, 100)
# mesh.save(args.output_dir)
