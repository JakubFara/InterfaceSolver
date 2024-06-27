import gmsh
import sys

gmsh.initialize()

gmsh.model.add("t19")
model = gmsh.model()

cylinder1 = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0.0, 1.0, 0.2)
cylinder2 = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0.0, 1.0, 0.3)

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
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2)
gmsh.write("data/tube3d.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
