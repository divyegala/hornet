/**
 * @author Divye Gala                                                  <br>
 *         Georgia Institute Of Technlogy, Dept. of Computer Science                   <br>
 *         divye.gala@gatech.edu
 * @date May, 2019
 * @version v1
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "Static/BreadthFirstSearch/BottomUp.cuh"
#include "Auxilary/DuplicateRemoving.cuh"
#include <Graph/GraphStd.hpp>
#include <Graph/BFS.hpp>

namespace hornets_nest {

const dist_t INF = std::numeric_limits<dist_t>::max();

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct BFSBUOperator {
    dist_t* d_distances;
    dist_t current_level;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex) {
    vid_t vertex_id = vertex.id();
    eoff_t start = 0, end = vertex.degree();
        for(eoff_t i = start; i < end; i++) {
            vid_t possible_parent = vertex.edge(i).dst_id();
	//printf("In for loop, child id: %d, parent id: %d, i: %d\n", vertex_id, possible_parent, i);
            if(d_distances[possible_parent] == current_level - 1) {
                d_distances[vertex_id] = current_level;
                queue.insert(vertex_id);
                break;
            }
        }
    }
};
//------------------------------------------------------------------------------
/////////////////
// BfsBottomUp //
/////////////////
BfsBottomUp::BfsBottomUp(HornetGraph& hornet) :
                                 StaticAlgorithm(hornet),
                                 queue(hornet, 5),
                                 load_balancing(hornet) {
    //std::cout<<"HI"<<std::endl;
    gpu::allocate(d_distances, hornet.nV());
    reset();
}

//BfsBottomUp::BfsBottomUp(HornetGraph& hornet, HornetGraph& hornet_inverse) :
 //                                StaticAlgorithm(hornet),
   //                              hornet_inverse(hornet_inverse),
     //                            queue(hornet, 5),
       //                          load_balancing(hornet) {
    //std::cout<<"HI"<<std::endl;
//    gpu::allocate(d_distances, hornet.nV());
//    gpu::allocate(p_parent, hornet.nV());
//   gpu::allocate(f_frontier, hornet.nV());
//    gpu::allocate(n_next, hornet.nV());
//    std::cout<<"Constructor Finished"<<std::endl;
//    reset();
//}

BfsBottomUp::~BfsBottomUp() {
    gpu::free(d_distances);
}

void BfsBottomUp::reset() {
    std::cout<<"Reset Started"<<std::endl;
    current_level = 1;
    queue.clear();

    auto distances = d_distances;
    forAllnumV(hornet, [=] __device__ (int i){ distances[i] = INF; } );
}

void BfsBottomUp::set_parameters(vid_t source) {
    // std::cout << "Set Params started";
       bfs_source = source;
       //auto parent = p_parent;
       queue.insert(bfs_source);               // insert bfs source in the frontier
       gpu::memsetZero(d_distances + bfs_source);  //reset source distance
}
/*
void BfsBottomUp::run() {
    while (queue.size() > 0) {
        forAllEdges(hornet, queue, BFSOperator1 { d_distances, queue },
                    load_balancing);
        queue.swap();
        forAll(queue, BFSOperator2 { d_distances, current_level });
        current_level++;
    }
}*/

void BfsBottomUp::run() {
    // while (queue.size() > 0) {
    //     forAllEdges(hornet, queue,
    //                 BFSOperatorAtomic { current_level, d_distances, queue },
    //                 load_balancing);
    //     queue.swap();
    //     current_level++;
    // }
//gpu::memsetZero(d_distances + bfs_source);

    while(queue.size() > 0) {
            // auto start = std::chrono::high_resolution_clock::now();
        auto distances = d_distances;
        forAllnumV(hornet, [=] __device__ (int i) {
            if(distances[i] == INF) {
                queue.insert(i);
            }
        });
        queue.swap();
        forAllVertices(hornet, queue, BFSBUOperator{d_distances, current_level, queue });
        current_level++;
        queue.swap();
        std::cout<<"BU, Frontier Size: "<<queue.size()<<std::endl;
        // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout<<"Time: "<<duration.count()<<std::endl;
    }
}

void BfsBottomUp::release() {
    gpu::free(d_distances);
    d_distances = nullptr;
}

bool BfsBottomUp::validate() {
    std::cout << "\nTotal enqueue vertices: "
              << xlib::format(queue.enqueue_items())
              << std::endl;

    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(hornet.csr_offsets(), hornet.nV(),
                                  hornet.csr_edges(), hornet.nE());
    BFS<vid_t, eoff_t> bfs(graph);
    bfs.run(bfs_source);
//std::cout<<"GPU Answer: "<<std::endl;
//gpu::printArray(p_parent, hornet.nV());
    auto h_distances = bfs.result();
//std::cout<<"CPU Answer: "<<std::endl;
//for(int i = 0; i < graph.nV(); i++) {
//	std::cout<<" "<<h_parent[i];
//}
    return gpu::equal(h_distances, h_distances + graph.nV(), d_distances);
}

} // namespace hornets_nest
