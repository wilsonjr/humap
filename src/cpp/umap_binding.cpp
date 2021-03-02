#include <pybind11/pybind11.h>
#include "hierarchical_umap.h"

namespace py = pybind11;

PYBIND11_MODULE(_hierarchical_umap, m) {

	
	
	py::class_<humap::HierarchicalUMAP>(m, "HUMAP")
		.def(py::init<string, py::array_t<double>, int, double, string, string, bool>())
		.def(py::init<>())
		.def("fit", &humap::HierarchicalUMAP::fit)
		.def("transform", &humap::HierarchicalUMAP::transform)
		.def("get_influence", &humap::HierarchicalUMAP::get_influence)
		.def("get_labels", &humap::HierarchicalUMAP::get_labels)
		.def("get_sigmas", &humap::HierarchicalUMAP::get_sigmas)
		.def("get_indices", &humap::HierarchicalUMAP::get_indices)
		.def("get_data", &humap::HierarchicalUMAP::get_data)
		.def("get_embedding", &humap::HierarchicalUMAP::get_embedding)
		.def("get_original_indices", &humap::HierarchicalUMAP::get_original_indices)
		.def("project", &humap::HierarchicalUMAP::project)
		.def("project_indices", &humap::HierarchicalUMAP::project_indices)
		.def("set_ab_parameters", &humap::HierarchicalUMAP::set_ab_parameters)
		.def("get_labels_selected", &humap::HierarchicalUMAP::get_labels_selected)
		.def("get_indices_selected", &humap::HierarchicalUMAP::get_indices_selected)
		.def("get_influence_selected", &humap::HierarchicalUMAP::get_influence_selected)
		.def("set_landmarks_nwalks", &humap::HierarchicalUMAP::set_landmarks_nwalks)
		.def("set_landmarks_wl", &humap::HierarchicalUMAP::set_landmarks_wl)
		.def("set_influence_nwalks", &humap::HierarchicalUMAP::set_influence_nwalks)
		.def("set_influence_wl", &humap::HierarchicalUMAP::set_influence_wl)
		.def("set_influence_neighborhood", &humap::HierarchicalUMAP::set_influence_neighborhood)
		.def("set_distance_similarity", &humap::HierarchicalUMAP::set_distance_similarity)
		.def("set_path_increment", &humap::HierarchicalUMAP::set_path_increment)
		.def("explain", &humap::HierarchicalUMAP::explain)
		.def("__repr__",
			[](humap::HierarchicalUMAP& a) {
				return "<class.HierarchicalUMAP>";
			});
}