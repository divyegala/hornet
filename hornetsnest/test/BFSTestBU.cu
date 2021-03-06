/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BreadthFirstSearch/BottomUp.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv,false);
    //graph.read(argv[1], SORT|PRINT_INFO);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);

    // HornetInit hornet_init_inverse(graph.nV(), graph.nE(), graph.csr_in_offsets(),
    //                        graph.csr_in_edges());
    //
    // HornetGraph hornet_graph_inverse(hornet_init_inverse);
    BfsBottomUp bfs_bottom_up(hornet_graph);
    // BfsBottomUp bfs_bottom_up(hornet_graph, hornet_graph_inverse);

	vid_t root = graph.max_out_degree_id();

	// if (argc==3)
	//   root = atoi(argv[2]);

    std::cout << "My root is " << root << std::endl;

	std::cout<<"Set params started in test"<<std::endl;
    bfs_bottom_up.set_parameters(root);
    std::cout << "Set Parameter in test" << std::endl;

    Timer<DEVICE> TM;
    cudaProfilerStart();
    TM.start();

    bfs_bottom_up.run();

    TM.stop();
    cudaProfilerStop();
    TM.print("BottomUp");

    // std::cout << "Number of levels is : " << bfs_bottom_up.getLevels() << std::endl;

    auto is_correct = bfs_bottom_up.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return !is_correct;
}

int main(int argc, char* argv[]) {
    int ret = 0;
#if defined(RMM_WRAPPER)
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
#endif

    ret = exec(argc, argv);

#if defined(RMM_WRAPPER)
    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();
#endif

    return ret;
}

