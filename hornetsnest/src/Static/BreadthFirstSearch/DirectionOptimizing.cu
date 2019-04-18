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
#include "Static/BreadthFirstSearch/DirectionOptimizing.cuh"
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
    dist_t* p_parent;
    dist_t* f_frontier;
    dist_t* n_next;
    dist_t* next_size;
    const eoff_t* in_offsets;
    const vid_t* in_edges;
    dist_t* d_distances;
    dist_t current_level;

    OPERATOR(Vertex& vertex) {
	if (p_parent[vertex.id()] == -1) {
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
                    d_distances[vertex.id()] = current_level;
                    break;
                }
            }
        }
    }
};

struct BFSTDOperator {
    TwoLevelQueue<vid_t> queue;
    dist_t* p_parent;
    dist_t* scout_count;
    const degree_t* device_degrees;
    dist_t* d_distances;
    dist_t current_level;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        if(p_parent[dst] == -1) {
            p_parent[dst] = vertex.id();
            d_distances[dst] = current_level;
            queue.insert(dst);
            int discard = atomicAdd(scout_count, device_degrees[dst]);
        }
    }
};

struct QueueToBitmap {
    TwoLevelQueue<vid_t> queue;
    dist_t* f_frontier;
    OPERATOR(Vertex& vertex) {
        f_frontier[vertex.id()] = 1;
    }
};

struct BitmapToQueue {
    dist_t* f_frontier;
    TwoLevelQueue<vid_t> queue;
    OPERATOR(int i) {
        if(f_frontier[i] == 1) {
            queue.insert(i);
        }
    }
};
//------------------------------------------------------------------------------
/////////////////
// BfsDirOpt //
/////////////////
BfsDirOpt::BfsDirOpt(HornetGraph& hornet) :
                                 StaticAlgorithm(hornet),
                                 hornet_inverse(hornet),
                                 queue(hornet, 5),
                                 load_balancing(hornet),
                                 is_directed(false) {
    //std::cout<<"HI"<<std::endl;
    gpu::allocate(d_distances, hornet.nV());
    gpu::allocate(p_parent, hornet.nV());
    gpu::allocate(f_frontier, hornet.nV());
    gpu::allocate(n_next, hornet.nV());
    std::cout<<"Constructor Finished"<<std::endl;
    reset();
}

BfsDirOpt::BfsDirOpt(HornetGraph& hornet, HornetGraph& hornet_inverse) :
                                StaticAlgorithm(hornet),
                                hornet_inverse(hornet_inverse),
                                queue(hornet, 5),
                                load_balancing(hornet),
                                is_directed(true) {
    // std::cout<<"HI"<<std::endl;
   gpu::allocate(d_distances, hornet.nV());
   gpu::allocate(p_parent, hornet.nV());
  gpu::allocate(f_frontier, hornet.nV());
   gpu::allocate(n_next, hornet.nV());
   std::cout<<"Constructor Finished"<<std::endl;
   reset();
}

BfsDirOpt::~BfsDirOpt() {
    gpu::free(d_distances);
    gpu::free(p_parent);
    gpu::free(f_frontier);
    gpu::free(n_next);
    gpu::free(next_size);
}

void BfsDirOpt::reset() {
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
    cudaMallocManaged((void**)&scout_count, sizeof(dist_t));
    std::cout<<"Reset Finished"<<std::endl;
}

void BfsDirOpt::set_parameters(vid_t source) {
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
   *scout_count = hornet.max_degree();
   //edges_to_check = is_directed ? hornet.nE() : 2 * hornet.nE();
   edges_to_check = hornet.nE();
}

void BfsDirOpt::run() {
    // while (queue.size() > 0) {
    //     forAllEdges(hornet, queue,
    //                 BFSOperatorAtomic { current_level, d_distances, queue },
    //                 load_balancing);
    //     queue.swap();
    //     current_level++;
    // }
//gpu::memsetZero(d_distances + bfs_source);
    dist_t alpha = 15, beta = 18;
    while (queue.size() > 0) {
        if (*scout_count > edges_to_check / alpha) {
            dist_t old_next_size;
            forAllVertices(hornet, queue, QueueToBitmap { queue, f_frontier});
            *next_size = queue.size();
            queue.swap();
            do {
                old_next_size = *next_size;
                std::cout<<"BU, Frontier Size: "<<*next_size<<std::endl;
                *next_size = 0;
                forAllVertices(hornet, BFSBUOperator { p_parent, f_frontier, n_next,
                                next_size, hornet.csr_offsets(), hornet.csr_edges(),
                                d_distances, current_level });
                current_level++;
                //gpu::printArray(n_next, hornet.nV());
                auto frontier = f_frontier;
                auto next = n_next;
                //dist_t tmp = 0;
                forAllnumV(hornet, [=] __device__ (int i){ frontier[i] = next[i]; } );
                //std::swap(frontier, next);
                forAllnumV(hornet, [=] __device__ (int i) { next[i] = 0; } );
                    //forAllnumV(hornet, [=] __device__ (int i){ frontier[i] = next[i]; } );
                frontier_size = *next_size;
                //std::cout<<"Old next Size: "<<old_next_size<<std::endl;
                //std::cout<<"N vertices / beta: "<<hornet.nV() / beta<<std::endl;
                //gpu::printArray(n_next, hornet.nV());
                    //gpu::printArray(p_parent, hornet.nV());
                    //forAllnumV(hornet, [=] __device__ (int i){ next[i] = 0; } );
                // *next_size = 0;
            }
            while ((frontier_size >= old_next_size) || (frontier_size > hornet.nV() / beta));
            //std::cout<<"Before BitmapToQueue, Frontier Size: "<<frontier_size<<", Queue Size: "<<queue.size()<<std::endl;
	    forAllnumV(hornet, BitmapToQueue { f_frontier, queue } );
//std::cout<<"After BitmapToQueue, Frontier Size: "<<frontier_size<<", Queue Size: "<<queue.size()<<std::endl;
            queue.swap();
//std::cout<<"After Swap, Frontier Size: "<<frontier_size<<", Queue Size: "<<queue.size()<<std::endl;
*scout_count = 1;
        }
        else{
        //std::cout << queue.size() << std::endl;
        //for all edges in "queue" applies the operator "BFSOperator" by using
        //the load balancing algorithm instantiated in "load_balancing"
            edges_to_check -= *scout_count;
                        std::cout<<"TD, Queue Size: "<<queue.size()<<std::endl;
            *scout_count = 0;
            forAllEdges(hornet, queue,
                        BFSTDOperator { queue, p_parent, scout_count, hornet.device_degrees(), d_distances, current_level },
                        load_balancing);
            //todo: gpu::forAllEdges
            //std::cout<<"TD, Queue Size: "<<queue.size()<<std::endl;
            current_level++;
            queue.swap();
	 }
    }
}

void BfsDirOpt::release() {
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

bool BfsDirOpt::validate() {
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
