from dolfin import compile_cpp_code
import os

path = os.path.dirname(__file__)
 
with open(f'{path}/cpp/FEniCScpp.cpp', "r") as txt_file:
    FEniCScpp = compile_cpp_code(txt_file.read())

with open(f'{path}/cpp/assemble_edge.cpp', "r") as txt_file:
    assembler = compile_cpp_code(txt_file.read())