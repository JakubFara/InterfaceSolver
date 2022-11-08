#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/AssemblerBase.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/fem/Form.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/common/types.h>
#include <dolfin/la/IndexMap.h>

#include <iostream>

std::shared_ptr<const dolfin::Mesh> get_mesh(const dolfin::FunctionSpace& V)
{
  return  V.mesh();
}

std::size_t local_facet(dolfin::Facet *f,dolfin::Cell c){
return  c.index(*f);
}
ufc::cell ufc_cell(dolfin::Cell c,std::size_t local_facet){
    ufc::cell uc;
    c.get_cell_data(uc, local_facet);
return uc;
}
int orientation(dolfin::Cell c,std::size_t local_facet){
    ufc::cell uc;
    c.get_cell_data(uc, local_facet);
return uc.orientation;
}
int num_dofs_in_cell(const dolfin::DofMap* dofmap){
  return dofmap->num_element_dofs(2);
}

std::vector<std::vector<int>> dofmap(const dolfin::Form &a,
				     dolfin::Cell c0,dolfin::Cell c1 ){
  const std::size_t form_rank = dolfin::UFC(a).form.rank();
  std::vector<const dolfin::GenericDofMap*> dofmaps;
  
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());
  std::vector<std::vector<dolfin::la_index>> macro_dofs(form_rank);
  for (std::size_t i = 0; i < form_rank; i++){
    // Get dofs for each cell
    auto cell_dofs0 = dofmaps[i]->cell_dofs(c0.index());
    auto cell_dofs1 = dofmaps[i]->cell_dofs(c1.index());
    // Create space in macro dof vector
    
    macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());
    
    // Copy cell dofs into macro dof vector
    std::copy(cell_dofs0.data(), cell_dofs0.data() + cell_dofs0.size(),
	      macro_dofs[i].begin());
    std::copy(cell_dofs1.data(), cell_dofs1.data() + cell_dofs1.size(),
	      macro_dofs[i].begin() + cell_dofs0.size());
  }
  return macro_dofs;
}

std::vector<std::vector<int>> dofmap(const dolfin::Form &a,
				     dolfin::Cell c ){
  const std::size_t form_rank = dolfin::UFC(a).form.rank();
  std::vector<const dolfin::GenericDofMap*> dofmaps;
  
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());
  std::vector<std::vector<dolfin::la_index>> macro_dofs(form_rank);
  for (std::size_t i = 0; i < form_rank; i++){
    // Get dofs for each cell
    auto cell_dofs = dofmaps[i]->cell_dofs(c.index());
    // Create space in macro dof vector
    
    macro_dofs[i].resize(cell_dofs.size());
    
    // Copy cell dofs into macro dof vector
    std::copy(cell_dofs.data(), cell_dofs.data() + cell_dofs.size(),
	      macro_dofs[i].begin());
  }
  return macro_dofs;
}

dolfin::Point middle_point(const dolfin::Facet f){
  // Get global index of vertices on the facet
  const std::size_t v1 = f.entities(0)[0];
  const std::size_t v2 = f.entities(0)[1];

  // Get mesh geometry
  const dolfin::MeshGeometry& geometry = f.mesh().geometry();

  // Get the coordinates of the three vertices
  const dolfin::Point p1 = geometry.point(v1);
  const dolfin::Point p2 = geometry.point(v2);

  // Subtract projection of p2 - p0 onto p2 - p1

  return (p1+p2)/2;
}


PYBIND11_MODULE(SIGNATURE, m){
  m.def("local_facet", &local_facet);
  m.def("dofmap", static_cast<std::vector<std::vector<int>>(*)
	(const dolfin::Form& ,dolfin::Cell ,dolfin::Cell)>(&dofmap));
  m.def("dofmap", static_cast<std::vector<std::vector<int>>(*)
	(const dolfin::Form& ,dolfin::Cell)>(&dofmap));
  m.def("orientation", &orientation);
  m.def("ufc_cell",&ufc_cell);
  m.def("middle_point", &middle_point);
  m.def("num_dofs_in_cell", &num_dofs_in_cell);
  m.def("get_mesh", &get_mesh);
}
