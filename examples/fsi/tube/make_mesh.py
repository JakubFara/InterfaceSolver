import gmsh
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mesh_level" , "-ml",
    help=(
        "navire slip parameter",
    ),
    default=1,
    type=int,
)

args = vars(parser.parse_args())
mesh_level = args["mesh_level"]

gmsh.initialize()

gmsh.model.add("t19")
model = gmsh.model()

tube_len = 0.022

cylinder1 = gmsh.model.occ.addCylinder(0, 0, -tube_len, 0, 0.0, 2 * tube_len, 0.012)
cylinder2 = gmsh.model.occ.addCylinder(0, 0, -tube_len, 0, 0.0, 2 * tube_len, 0.014)

model_dim_tags = model.occ.cut([(3, cylinder2)], [(
    3, cylinder1)], removeObject=True, removeTool=False)[0][0][1]

model.occ.synchronize()
# resolution = 0.001
# threshold = gmsh.model.mesh.field.add("Threshold")
# gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
# gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20*resolution)
model.addPhysicalGroup(3, [cylinder1], tag=2)
model.addPhysicalGroup(3, [model_dim_tags], tag=1)
#
up_cyl_1, down_cyl_1 = model.getAdjacencies(3, cylinder1)
up_cyl_2, down_cyl_2 = model.getAdjacencies(3, cylinder2)

com_1 = gmsh.model.occ.getCenterOfMass(2, down_cyl_1[0])
com_2 = gmsh.model.occ.getCenterOfMass(2, down_cyl_1[1])
com_3 = gmsh.model.occ.getCenterOfMass(2, down_cyl_1[2])
print(f"com1 = {down_cyl_1}")
print(f"com1 = {down_cyl_2}")

for down in down_cyl_1:
    com = model.occ.getCenterOfMass(2, down)
    if com[2] == -tube_len:
        model.addPhysicalGroup(2, [down], tag=11)
    elif com[2] == tube_len:
        model.addPhysicalGroup(2, [down], tag=12)
    elif abs(com[2] - 0.0) < 1e-10:
        model.addPhysicalGroup(2, [down], tag=13)
        print(f"----------------------")

for down in down_cyl_2:
    com = model.occ.getCenterOfMass(2, down)
    print(com[2])
    if com[2] == -tube_len:
        model.addPhysicalGroup(2, [down], tag=21)
        print(1)
    elif com[2] == tube_len:
        model.addPhysicalGroup(2, [down], tag=22)
        print(2)
    elif abs(com[2] - 0.0) < 1e-10:
        if down not in set(down_cyl_1):
            model.addPhysicalGroup(2, [down], tag=23)
            print(3)

if mesh_level == 1:
    n = 22
elif mesh_level == 2:
    n = 30

dx = 2 * tube_len / n

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", dx)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2)
gmsh.write(f"data/tube3d_lev{mesh_level}.msh")

# Launch the GUI to see the results:
# if '-nopopup' not in sys.argv:
    # gmsh.fltk.run()

gmsh.finalize()
