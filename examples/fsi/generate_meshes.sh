python3 make_mesh.py
dolfin-convert data/tube3d.msh data/tube3d.xml
python3 xml_to_hdf5.py
python3 make_discontinuous_mesh.py
rm data/*.xml
