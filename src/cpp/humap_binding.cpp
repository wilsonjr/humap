// Author: Wilson Estécio Marcílio Júnior <wilson_jr@outlook.com>

/*
 *
 * Copyright (c) 2021, Wilson Estécio Marcílio Júnior (São Paulo State University)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the São Paulo State University.
 * 4. Neither the name of the São Paulo State University nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY WILSON ESTÉCIO MARCÍLIO JÚNIOR ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL WILSON ESTÉCIO MARCÍLIO JÚNIOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#include <pybind11/pybind11.h>
#include "hierarchical_umap.h"

namespace py = pybind11;

PYBIND11_MODULE(_hierarchical_umap, m) {
	
	py::class_<humap::HierarchicalUMAP>(m, "HUMAP")
		.def(py::init<string, py::array_t<float>, int, float, string, string, bool, bool>())
		.def(py::init<>())
		.def("fit", &humap::HierarchicalUMAP::fit)
		.def("transform", &humap::HierarchicalUMAP::transform)
		.def("transform_with_init", &humap::HierarchicalUMAP::transform_with_init)
		.def("get_influence", &humap::HierarchicalUMAP::get_influence)
		.def("get_labels", &humap::HierarchicalUMAP::get_labels)
		.def("get_indices", &humap::HierarchicalUMAP::get_indices)
		.def("get_data", &humap::HierarchicalUMAP::get_data)
		.def("get_embedding", &humap::HierarchicalUMAP::get_embedding)
		.def("get_original_indices", &humap::HierarchicalUMAP::get_original_indices)
		.def("project", &humap::HierarchicalUMAP::project)
		.def("project_indices", &humap::HierarchicalUMAP::project_indices)
		.def("set_ab_parameters", &humap::HierarchicalUMAP::set_ab_parameters)
		.def("get_labels_selected", &humap::HierarchicalUMAP::get_labels_selected)
		.def("get_indices_selected", &humap::HierarchicalUMAP::get_indices_selected)
		.def("get_indices_fixed", &humap::HierarchicalUMAP::get_indices_fixed)
		.def("get_influence_selected", &humap::HierarchicalUMAP::get_influence_selected)
		.def("set_landmarks_nwalks", &humap::HierarchicalUMAP::set_landmarks_nwalks)
		.def("set_landmarks_wl", &humap::HierarchicalUMAP::set_landmarks_wl)
		.def("set_influence_nwalks", &humap::HierarchicalUMAP::set_influence_nwalks)
		.def("set_influence_wl", &humap::HierarchicalUMAP::set_influence_wl)
		.def("set_influence_neighborhood", &humap::HierarchicalUMAP::set_influence_neighborhood)
		.def("set_distance_similarity", &humap::HierarchicalUMAP::set_distance_similarity)
		.def("set_focus_context", &humap::HierarchicalUMAP::set_focus_context)
		.def("set_fixed_datapoints", &humap::HierarchicalUMAP::set_fixed_datapoints)
		.def("set_fixing_term", &humap::HierarchicalUMAP::set_fixing_term)
		.def("set_info_file", &humap::HierarchicalUMAP::set_info_file)
		.def("set_n_epochs", &humap::HierarchicalUMAP::set_n_epochs)
		.def("get_knn", &humap::HierarchicalUMAP::get_knn)
		.def("get_knn_dists", &humap::HierarchicalUMAP::get_knn_dists)
		.def("__repr__",
			[](humap::HierarchicalUMAP& a) {
				return "<class.HierarchicalUMAP>";
			});
}