/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
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

struct BFSOperator {
    dist_t* p_parent;
    dist_t* f_frontier;
    dist_t* n_next;
    dist_t* next_size;
    const eoff_t* in_offsets;
    const vid_t* in_edges;

    OPERATOR(Vertex& vertex) {
	//std::cout<<"In Operator, vertex id: "<<vertex.id()<<std::endl;
        //printf("In Operator, vertex id: %d", vertex.id());
	if (p_parent[vertex.id()] == -1) {
            // for (auto eit = vertex.begin(); eit != vertex.end(); eit++) {
            //     if (f_frontier[*eit.dst_id()] == 1) {
            //         n_next[vertex.id()] = 1;
            //         next_size++;
            //         p_parent[vertex.id()] = *eit.dst_id();
            //         break;
            //     }
            // }
            //printf("In Operator with no parent, vertex id: %d\n", vertex.id());
	    eoff_t start = in_offsets[vertex.id()], end = in_offsets[vertex.id()] + vertex.degree();
            for(eoff_t i = start; i < end; i++) {
                vid_t possible_parent = in_edges[i];
		//printf("In for loop, child id: %d, parent id: %d, i: %d\n", vertex.id(), possible_parent, i);
                if(f_frontier[possible_parent] == 1) {
                    n_next[vertex.id()] = 1;
                    //++*next_size;
		    int discard = atomicAdd(next_size, 1);
                    //printf("Next_Size: %d", *next_size);
		    p_parent[vertex.id()] = possible_parent;
                    break;
                }
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
    gpu::allocate(p_parent, hornet.nV());
    gpu::allocate(f_frontier, hornet.nV());
    gpu::allocate(n_next, hornet.nV());
    std::cout<<"Constructor Finished"<<std::endl;
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
    gpu::free(p_parent);
    gpu::free(f_frontier);
    gpu::free(n_next);
    gpu::free(next_size);
}

void BfsBottomUp::reset() {
    std::cout<<"Reset Started"<<std::endl;
    current_level = 1;
    queue.clear();

    auto distances = d_distances;
    forAllnumV(hornet, [=] __device__ (int i){ distances[i] = INF; } );

    auto next = n_next;
    forAllnumV(hornet, [=] __device__ (int i){ next[i] = 0; } );
    //*next_size = 0;
    cudaMallocManaged((void**)&next_size, sizeof(dist_t));
    *next_size = 0;
    std::cout<<"Reset Finished"<<std::endl;
}

void BfsBottomUp::set_parameters(vid_t source) {
std::cout << "Set Params started";
   bfs_source = source;
   auto parent = p_parent;
   forAllnumV(hornet, [=] __device__ (int i){
       if(i == bfs_source) {
           parent[i] = bfs_source;
       }
       else {
           parent[i] = -1;
       }
   } );

   auto frontier = f_frontier;
   forAllnumV(hornet, [=] __device__ (int i){
       if(i == bfs_source) {
           frontier[i] = 1;
       }
       else {
           frontier[i] = 0;
       }
   } );
   frontier_size = 1;
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

    while (frontier_size > 0) {
        //std::cout<<"In while loop: "<<std::endl;
    	forAllVertices(hornet, BFSOperator { p_parent, f_frontier, n_next, next_size, hornet.csr_offsets(), hornet.csr_edges() });
    	//gpu::printArray(n_next, hornet.nV());
    	auto frontier = f_frontier;
            auto next = n_next;
    	//dist_t tmp = 0;
    	forAllnumV(hornet, [=] __device__ (int i){ frontier[i] = next[i]; } );
    	//std::swap(frontier, next);
    	forAllnumV(hornet, [=] __device__ (int i) { next[i] = 0; } );
            //forAllnumV(hornet, [=] __device__ (int i){ frontier[i] = next[i]; } );
            frontier_size = *next_size;
    	std::cout<<"Frontier Size: "<<*next_size<<std::endl;
    	//gpu::printArray(n_next, hornet.nV());
            //gpu::printArray(p_parent, hornet.nV());
            //forAllnumV(hornet, [=] __device__ (int i){ next[i] = 0; } );
            *next_size = 0;
    }
}

void BfsBottomUp::release() {
    gpu::free(d_distances);
    gpu::free(p_parent);
    gpu::free(f_frontier);
    gpu::free(n_next);
    d_distances = nullptr;
    p_parent = nullptr;
    f_frontier = nullptr;
    n_next = nullptr;
    gpu::free(next_size);
    next_size = nullptr;
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
    auto h_parent = bfs.result_parents();
//std::cout<<"CPU Answer: "<<std::endl;
//for(int i = 0; i < graph.nV(); i++) {
//	std::cout<<" "<<h_parent[i];
//}
    return gpu::equal(h_parent, h_parent + graph.nV(), p_parent);
}

} // namespace hornets_nest
