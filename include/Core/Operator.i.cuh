
namespace custinger_alg {
namespace detail {

template<void (*Operator)(int, void*)>
__global__ void forAllKernel(int num_items, void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_items; i += stride)
        Operator(i, optional_data);
}

//------------------------------------------------------------------------------
template<void (*Operator)(custinger::vid_t, void*)>
__global__ void forAllnumVKernel(custinger::vid_t d_nV, void* optional_data) {
    using custinger::vid_t;
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (vid_t i = id; i < d_nV; i += stride)
        Operator(i, optional_data);
}

//------------------------------------------------------------------------------
template<void (*Operator)(custinger::eoff_t, void*)>
__global__ void forAllnumEKernel(custinger::eoff_t d_nE, void* optional_data) {
    using custinger::eoff_t;
    int      id = blockIdx.x * blockDim.x + threadIdx.x;
    int  stride = gridDim.x * blockDim.x;

    for (eoff_t i = id; i < d_nE; i += stride)
        Operator(i, optional_data);
}

//------------------------------------------------------------------------------
template<void (*Operator)(const custinger::Vertex&, void*)>
__global__ void forAllVerticesKernel(custinger::cuStingerDevData data,
                                     void* optional_data) {
    using custinger::vid_t;
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (vid_t i = id; i < data.nV; i += stride)
        Operator(custinger::Vertex(data, i), optional_data);
}

//------------------------------------------------------------------------------

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK,
      void (*Operator)(const custinger::Vertex&, const custinger::Edge&, void*)>
__global__
void forAllEdgesKernel(const custinger::eoff_t* __restrict__ csr_offsets,
                       custinger::cuStingerDevData           data,
                       void*                    __restrict__ optional_data) {

    __shared__ custinger::degree_t smem[ITEMS_PER_BLOCK];
    const auto lambda = [&](int pos, custinger::degree_t offset) {
                        custinger::Vertex vertex(data, pos);
                        Operator(vertex, vertex.edge(offset), optional_data);
                    };
    xlib::binarySearchLB<BLOCK_SIZE>(csr_offsets, data.nV + 1, smem, lambda);
}

//------------------------------------------------------------------------------

/*
template<void (*Operator)(Vertex, void*)>
__global__ void forAllBatchEdgesKernel(EdgeBatch edge_batch,
                                       void* optional_data) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int   size = edge_batch.size;

    for (int i = id; i < size; i += stride)
        Operator(Vertex(i), Edge(i), optional_data);
}*/

} // namespace detail

//==============================================================================
//==============================================================================

template<void (*Operator)(int, void*)>
void forAll(int num_items, void* optional_data) noexcept {
    detail::forAllKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(num_items), BLOCK_SIZE_OP1 >>>
        (num_items, optional_data);
}

//------------------------------------------------------------------------------

template<void (*Operator)(custinger::vid_t, void*)>
void forAllnumV(const custinger::cuStinger& custinger, void* optional_data)
                noexcept {

    detail::forAllnumVKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(custinger.nV()), BLOCK_SIZE_OP1 >>>
        (custinger.nV(), optional_data);
}

//------------------------------------------------------------------------------

template<void (*Operator)(custinger::eoff_t, void*)>
void forAllnumE(const custinger::cuStinger& custinger, void* optional_data)
                noexcept {

    detail::forAllnumEKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(custinger.nE()), BLOCK_SIZE_OP1 >>>
        (custinger.nE(), optional_data);
}

//==============================================================================

template<void (*Operator)(const custinger::Vertex&, void*)>
void forAllVertices(const custinger::cuStinger& custinger, void* optional_data)
                    noexcept {

    detail::forAllVerticesKernel<Operator>
        <<< xlib::ceil_div<BLOCK_SIZE_OP1>(custinger.nV()), BLOCK_SIZE_OP1 >>>
        (custinger.nV(), optional_data);
}

//------------------------------------------------------------------------------

template<void (*Operator)(const custinger::Vertex&, const custinger::Edge&,
                          void*)>
void forAllEdges(custinger::cuStinger& custinger, void* optional_data)
                 noexcept {

    using custinger::vid_t;
    const unsigned BLOCK_SIZE = 256;
    const int  PARTITION_SIZE = xlib::SMemPerBlock<BLOCK_SIZE, vid_t>::value;
    int num_partitions = xlib::ceil_div<PARTITION_SIZE>(custinger.nE());

    detail::forAllEdgesKernel<BLOCK_SIZE_OP1, PARTITION_SIZE, Operator>
       <<< num_partitions, BLOCK_SIZE_OP1 >>>
       (custinger.device_csr_offsets(), custinger.device_data(), optional_data);
}

//==============================================================================


/*template<void (*Operator)(custinger::Vertex, custinger::Edge, void*)>
void forAllBatchEdges(const EdgeBatch& edge_batch, void* optional_data) {

}*/

} // namespace custinger_alg
