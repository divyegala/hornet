/**
 * @brief DirectionOptimizing implementation of Breadth-first Search by using C++11-Style
 *        APIs
 * @author Divye Gala                                                  <br>
 *         Georgia Institute Of Technlogy, Dept. of Computer Science                   <br>
 *         divye.gala@gatech.edu
 * @date April, 2019
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
 *
 * @file
 */
#pragma once

#include "HornetAlg.hpp"

namespace hornets_nest {

using HornetGraph = gpu::Csr<EMPTY, EMPTY>;
//using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

using dist_t = int;

class BfsDirOpt : public StaticAlgorithm<HornetGraph> {
public:
    BfsDirOpt(HornetGraph& hornet);
    BfsDirOpt(HornetGraph& hornet, HornetGraph& hornet_inverse);
    ~BfsDirOpt();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    void set_parameters(vid_t source);

    dist_t getLevels(){return current_level;}

private:
    TwoLevelQueue<vid_t>        queue;
    load_balancing::BinarySearch load_balancing;
    //load_balancing::VertexBased1 load_balancing;
    HornetGraph& hornet_inverse;
    dist_t* d_distances   { nullptr };
    vid_t   bfs_source    { 0 };
    dist_t  current_level { 0 };
    bool is_directed {false};
};

} // namespace hornets_nest