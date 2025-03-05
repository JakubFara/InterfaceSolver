ml=1

python3 make_mesh.py -ml $ml
dolfin-convert data/tube3d_lev$ml.msh data/tube3d_lev$ml.xml
python3 xml_to_hdf5.py -ml $ml
python3 make_discontinuous_mesh.py -ml $ml
rm data/*.xml
