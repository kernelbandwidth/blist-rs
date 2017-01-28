# tlist-rs
A List-type datastructure that can be used in place of a linked list or doubly linked list.
The TList is backed by a Left-Leaning Red-Black tree (http://www.cs.princeton.edu/~rs/talks/LLRB/LLRB.pdf) with asympototic O(log N) insert, delete, and indexing operations.
Implementation uses entirely safe code with a Vec-based Arena-style internal allocation, so should have similar cache-behavior as a Vec.

Named TList because it is a Tree-based List.
It's not supposed to be clever.

No dependencies except the rand crate, which is only used to generate randomized data for testing.

