#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <algorithm>
#include <dolfin/log/log.h>
#include <dolfin/log/Progress.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/AssemblerBase.h>
#include <dolfin/fem/Assembler.h>
#include <dolfin/common/MPI.h>
#include <petscmat.h>

#include <mpi.h>
#include <dolfin/la/GenericMatrix.h>
#include <iostream>
//#include <vector>
using namespace dolfin;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenRowMajorMatrix_t;  
Point middle_point(const Cell& cell, std::size_t facet);
//typedef struct _p_Mat*           Mat;
class SingleAssembler : public Assembler{
public:
  SingleAssembler(const Form& a_);
    
  std::vector<double> assemble_facet(std::size_t local_facet0,std::size_t local_facet1,
				     std::vector<double> coordinate_dofs0,
				     std::vector<double> coordinate_dofs1,
				     int cell0_orientation,
				     int cell1_orientation,
				     std::vector<std::vector<double>> w_);
  
  std::size_t form_rank();
  std::vector<std::vector<double>> w(const Cell& c,
				     const std::size_t local_facet,
				     const std::vector<double>& coordinate_dofs);
private:
  std::size_t form_rank_single;
  const ufc::form& form;
  const Form& a;
  ufc::interior_facet_integral* integral_single;
  std::size_t D;
  const Mesh& mesh_single;
  std::vector<const GenericDofMap*> dofmaps_single;
  int my_mpi_rank;
  UFC ufc_single;
  const std::vector<std::shared_ptr<const dolfin::GenericFunction>> coefficients;
  std::vector<dolfin::FiniteElement> coefficient_elements;
  std::vector<std::vector<double>> _w;
  std::vector<double*> w_pointer;
};


SingleAssembler::SingleAssembler(const Form& a_):
  a(a_),mesh_single(*(a_.mesh())),ufc_single(a_),coefficients(a_.coefficients()),form(*a_.ufc_form())
{
  integral_single = ufc_single.default_interior_facet_integral.get();
  D = mesh_single.topology().dim();
  form_rank_single = ufc_single.form.rank();

  MPI_Comm_rank(mesh_single.mpi_comm(),&my_mpi_rank); //MPI::rank(mesh.mpi_comm());

  for (std::size_t i = 0; i < form_rank_single; ++i)
    dofmaps_single.push_back(a.function_space(i)->dofmap().get());

  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    std::shared_ptr<ufc::finite_element>
      element(form.create_finite_element(form.rank() + i));
    coefficient_elements.push_back(FiniteElement(element));
  }
  
  _w.resize(form.num_coefficients());
  w_pointer.resize(form.num_coefficients());
  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    _w[i].resize(coefficient_elements[i].space_dimension());
    w_pointer[i] = _w[i].data();
  }
}

std::size_t SingleAssembler::form_rank(){
  return form_rank_single;
}
std::vector<std::vector<double>> SingleAssembler::w(const Cell& c,
						    const std::size_t local_facet,
						    const std::vector<double>& coordinate_dofs){
  ufc::cell ufc_cell;
  //if (!ufc_single.form.has_interior_facet_integrals()){
  //  return _w;
  //}
  //auto integral_cell = ufc_single.default_interior_facet_integral.get();

  std::vector<bool> enabled_coefficients = integral_single->enabled_coefficients();
  c.get_cell_data(ufc_cell, local_facet);
  // Restrict coefficients to facet
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!enabled_coefficients[i])
      continue;
    coefficients[i]->restrict(_w[i].data(), coefficient_elements[i], c,
                              coordinate_dofs.data(), ufc_cell);
  }
  return _w;
}

std::vector<double> SingleAssembler::assemble_facet(std::size_t local_facet0,
						    std::size_t local_facet1,
						    const std::vector<double> coordinate_dofs0,
						    const std::vector<double> coordinate_dofs1,
						    int cell0_orientation,
						    int cell1_orientation,
						    std::vector<std::vector<double>> w_){
  
  //if (!ufc_single.form.has_interior_facet_integrals())
  //  return ufc_single.macro_A;
  dolfin_assert(facet->num_entities(D) == 2);
  std::vector<double*>  w_pointer;
  w_pointer.resize(2*form.num_coefficients());
  for (std::size_t i = 0; i < form.num_coefficients(); i++)
  {
    w_pointer[i] = w_[i].data();
  }
  integral_single->tabulate_tensor(ufc_single.macro_A.data(),
				   w_pointer.data(),
				   coordinate_dofs0.data(),
				   coordinate_dofs1.data(),
				   local_facet0,
				   local_facet1,
				   cell0_orientation,
				   cell1_orientation);

  
  return ufc_single.macro_A;
};




//_________________________________________________pybind________________________________________-
namespace py = pybind11;
PYBIND11_MODULE(SIGNATURE, m) {
  py::class_<SingleAssembler, std::shared_ptr<SingleAssembler>, Assembler>(m, "SingleAssembler")
    .def(py::init<const Form& >())
    //.def("interface_edges",&InterfaceAssembler::interface_edges)
    .def("form_rank", &SingleAssembler::form_rank)
    .def("assemble_facet", &SingleAssembler::assemble_facet)
    .def("w", &SingleAssembler::w);
}
