/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
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
#pragma once

#include "Support/Device/SafeCudaAPI.cuh"
#include "Support/Host/Numeric.hpp"

namespace custinger {

inline BitTree::BitTree(int block_items, int blockarray_items) noexcept :
        _block_items(block_items),
        _blockarray_items(blockarray_items),
        _log_block_items(xlib::log2(block_items)),
        _log_blockarray_items(xlib::log2(blockarray_items)),
        _num_blocks(_blockarray_items / _block_items),
        _num_levels(xlib::max(xlib::ceil_log<WORD_SIZE>(_num_blocks), 1)),
        _internal_bits(xlib::geometric_serie<WORD_SIZE>(_num_levels - 1) - 1),
        _internal_words(xlib::ceil_div<WORD_SIZE>(_internal_bits)),
        _external_words(xlib::ceil_div<WORD_SIZE>(_num_blocks)),
        _num_words(_internal_words + _external_words),
        _total_bits(_num_words * WORD_SIZE) {

    assert(xlib::is_power2(block_items));
    assert(xlib::is_power2(blockarray_items));

    _h_ptr = new byte_t[sizeof(edge_t) << _log_blockarray_items];
    cuMalloc(_d_ptr, sizeof(edge_t) << _log_blockarray_items);

    const word_t EMPTY = static_cast<word_t>(-1);
    std::fill(_array, _array + _num_words, EMPTY);
    _last_level = _array + _internal_words;
}

inline BitTree::BitTree(BitTree&& obj) noexcept :
                    _block_items(obj._block_items),
                    _blockarray_items(obj._blockarray_items),
                    _log_block_items(obj._log_block_items),
                    _log_blockarray_items(obj._log_blockarray_items),
                    _num_blocks(obj._num_blocks),
                    _num_levels(obj._num_levels),
                    _internal_bits(obj._internal_bits),
                    _internal_words(obj._internal_words),
                    _external_words(obj._external_words),
                    _num_words(obj._num_words),
                    _total_bits(obj._total_bits),
                    _last_level(_array + _internal_words),
                    _h_ptr(obj._h_ptr),
                    _d_ptr(obj._d_ptr),
                    _size(obj._size) {
    assert(_block_items == obj._block_items);
    assert(_blockarray_items == obj._blockarray_items);

    std::copy(obj._array, obj._array + _num_words, _array);
    obj._last_level = nullptr;
    obj._h_ptr      = nullptr;
    obj._d_ptr      = nullptr;
    obj._size       = 0;
}

inline BitTree& BitTree::operator=(BitTree&& obj) noexcept {
    assert(_block_items == obj._block_items || _block_items == 0 ||
           obj._block_items == 0);
    assert(_blockarray_items == obj._blockarray_items ||
           _blockarray_items == 0 || obj._blockarray_items == 0);
    if (_block_items == 0) {
        const_cast<int&>(_block_items)          = obj._block_items;
        const_cast<int&>(_blockarray_items)     = obj._blockarray_items;
        const_cast<int&>(_log_block_items)      = obj._log_block_items;
        const_cast<int&>(_log_blockarray_items) = obj._log_blockarray_items;
        const_cast<int&>(_num_blocks)           = obj._num_blocks;
        const_cast<int&>(_num_levels)           = obj._num_levels;
        const_cast<int&>(_internal_bits)        = obj._internal_bits;
        const_cast<int&>(_internal_words)       = obj._internal_words;
        const_cast<int&>(_external_words)       = obj._external_words;
        const_cast<int&>(_num_words)            = obj._num_words;
        const_cast<int&>(_total_bits)           = obj._total_bits;
    }
    std::copy(obj._array, obj._array + _num_words, _array);
    _last_level = _array + _internal_words;
    _d_ptr      = obj._d_ptr;
    _h_ptr      = obj._h_ptr;
    _size       = obj._size;

    obj._last_level = nullptr;
    obj._h_ptr      = nullptr;
    obj._d_ptr      = nullptr;
    obj._size       = 0;
    return *this;
}

#if defined(B_PLUS_TREE)

inline BitTree::BitTree() :
                _block_items(0),
                _blockarray_items(0),
                _log_block_items(0),
                _log_blockarray_items(0),
                _num_blocks(0),
                _num_levels(0),
                _internal_bits(0),
                _internal_words(0),
                _external_words(0),
                _num_words(0),
                _total_bits(0) {}

inline BitTree::BitTree(const BitTree& obj) noexcept :
                    _block_items(obj._block_items),
                    _blockarray_items(obj._blockarray_items),
                    _log_block_items(obj._log_block_items),
                    _log_blockarray_items(obj._log_blockarray_items),
                    _num_blocks(obj._num_blocks),
                    _num_levels(obj._num_levels),
                    _internal_bits(obj._internal_bits),
                    _internal_words(obj._internal_words),
                    _external_words(obj._external_words),
                    _num_words(obj._num_words),
                    _total_bits(obj._total_bits),
                    _last_level(_array + _internal_words),
                    _h_ptr(obj._h_ptr),
                    _d_ptr(obj._d_ptr),
                    _size(obj._size) {
    assert(_block_items == obj._block_items);
    assert(_blockarray_items == obj._blockarray_items);

    std::copy(obj._array, obj._array + _num_words, _array);
    _last_level = _array + _internal_words;
    _d_ptr      = obj._d_ptr;
    _h_ptr      = obj._h_ptr;
    _size       = obj._size;

    const_cast<BitTree&>(obj)._last_level = nullptr;
    const_cast<BitTree&>(obj)._h_ptr      = nullptr;
    const_cast<BitTree&>(obj)._d_ptr      = nullptr;
    const_cast<BitTree&>(obj)._size       = 0;
}

#endif

inline BitTree::~BitTree() noexcept {
    cuFree(_d_ptr);
    delete[] _h_ptr;
}

inline void BitTree::free_host_ptr() noexcept {
    delete[] _h_ptr;
    _h_ptr = nullptr;
}

//------------------------------------------------------------------------------

inline std::pair<byte_t*, byte_t*> BitTree::insert() noexcept {
    assert(_size < _num_blocks && "tree is full");
    _size++;
    //find the first empty location
    int index = 0;
    for (int i = 0; i < _num_levels - 1; i++) {
        assert(index < _total_bits && _array[index / WORD_SIZE] != 0);
        int pos = __builtin_ctz(_array[index / WORD_SIZE]);
        index   = (index + pos + 1) * WORD_SIZE;
    }
    assert(index < _total_bits && _array[index / WORD_SIZE] != 0);
    index += __builtin_ctz(_array[index / WORD_SIZE]);
    assert(index < _total_bits);

    xlib::delete_bit(_array, index);
    if (_array[index / WORD_SIZE] == 0) {
        const auto& lambda = [&](int index) {
                                          xlib::delete_bit(_array, index);
                                          return _array[index / WORD_SIZE] != 0;
                                        };
        parent_traverse(index, lambda);
    }
    int block_index = index - _internal_bits;
    assert(block_index >= 0 && block_index < _blockarray_items);

    auto offset = (block_index * sizeof(vid_t)) << _log_block_items;
    return std::pair<byte_t*, byte_t*>(_h_ptr + offset, _d_ptr + offset);
}

//------------------------------------------------------------------------------

inline void BitTree::remove(void* device_ptr) noexcept {
    assert(_size != 0 && "tree is empty");
    _size--;
    int p_index = remove_aux(device_ptr) + _internal_bits;

    parent_traverse(p_index, [&](int index) {
                                bool ret = _array[index / WORD_SIZE] != 0;
                                xlib::write_bit(_array, index);
                                return ret;
                            });
}

//------------------------------------------------------------------------------

inline int BitTree::remove_aux(void* device_ptr) noexcept {
    unsigned diff = std::distance(reinterpret_cast<edge_t*>(_d_ptr),
                                  static_cast<edge_t*>(device_ptr));
    int     index = diff >> _log_block_items;   // diff / block_items
    assert(index < _num_words);
    assert(xlib::read_bit(_last_level, index) == 0 && "not found");

    xlib::write_bit(_last_level, index);
    return index;
}

template<typename Lambda>
inline void BitTree::parent_traverse(int index, const Lambda& lambda) noexcept {
    index /= WORD_SIZE;
    while (index != 0) {
        index--;
        if (lambda(index))
            return;
        index /= WORD_SIZE;
    }
}

inline int BitTree::size() const noexcept {
    return _size;
}

inline bool BitTree::full() const noexcept {
    return _size == _num_blocks;
}

inline std::pair<byte_t*, byte_t*>
BitTree::base_address() const noexcept {
    return std::pair<byte_t*, byte_t*>(_h_ptr, _d_ptr);
}

inline bool BitTree::belong_to(void* to_check) const noexcept {
    return to_check >= _d_ptr &&
           to_check < _d_ptr + (sizeof(edge_t) << _log_blockarray_items);
}

inline void BitTree::print() const noexcept {
    const int ROW_SIZE = 64;
    int          count = WORD_SIZE;
    auto       tmp_ptr = _array;
    std::cout << "BitTree:\n";

    for (int i = 0; i < _num_levels - 1; i++) {
        std::cout << "\nlevel " << i << " :\n";
        assert(count < ROW_SIZE || count % ROW_SIZE == 0);

        int size = std::min(count, ROW_SIZE);
        for (int j = 0; j < count; j += ROW_SIZE) {
            xlib::printBits(tmp_ptr, size);
            tmp_ptr += size / WORD_SIZE;
            if (tmp_ptr >= _array + _num_words)
                break;
        }
        count *= WORD_SIZE;
    }
    std::cout << "\nlevel " << _num_levels - 1 << " :\n";
    xlib::printBits(tmp_ptr, _external_words * WORD_SIZE);
    std::cout << std::endl;
}

inline void BitTree::statistics() const noexcept {
    std::cout << "\nBitTree Statistics:\n"
              << "\n     BLOCK_ITEMS: " << _block_items
              << "\nBLOCKARRAY_ITEMS: " << _blockarray_items
              << "\n      NUM_BLOCKS: " << _num_blocks
              << "\n       sizeof(T): " << sizeof(edge_t)
              << "\n   BLOCK_SIZE(b): " << _block_items * sizeof(edge_t) << "\n"
              << "\n      NUM_LEVELS: " << _num_levels
              << "\n       WORD_SIZE: " << WORD_SIZE
              << "\n   INTERNAL_BITS: " << _internal_bits
              << "\n   EXTERNAL_BITS: " << _num_blocks
              << "\n      TOTAL_BITS: " << _total_bits
              << "\n  INTERNAL_WORDS: " << _internal_words
              << "\n EXTERNAL_WORLDS: " << _external_words
              << "\n       NUM_WORDS: " << _num_words << "\n\n";
}

} // namespace custinger
