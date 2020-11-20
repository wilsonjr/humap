#include <pybind11/pybind11.h>
#include "hierarchical_umap.h"

namespace py = pybind11;

PYBIND11_MODULE(hierarchical_umap, m) {


	py::class_<umap::UMAP>(m, "UMAP")
		.def(py::init<>())
		.def(py::init<string, string>())
		// .def("fit_hierarchy_sparse", &umap::UMAP::fit_hierarchy_sparse)
		// .def("fit_hierarchy", &umap::UMAP::fit_hierarchy)
		.def("__repr__", 
			[](umap::UMAP& a) {
				return "<class.UMAP named "+a.getName()+">";
			});

	py::class_<humap::HierarchicalUMAP>(m, "HUMAP")
		.def(py::init<string, py::array_t<float>, int, string, bool>())
		.def(py::init<>())
		.def("fit", &humap::HierarchicalUMAP::fit)
		.def("get_labels", &humap::HierarchicalUMAP::get_labels)
		.def("get_sigmas", &humap::HierarchicalUMAP::get_sigmas)
		.def("get_indices", &humap::HierarchicalUMAP::get_indices)
		.def("get_data", &humap::HierarchicalUMAP::get_data)
		.def("get_embedding", &humap::HierarchicalUMAP::get_embedding)
		.def("__repr__",
			[](humap::HierarchicalUMAP& a) {
				return "<class.HierarchicalUMAP>";
			});




	// m.def("add", &umap::add);

}