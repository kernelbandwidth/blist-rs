# tlist-rs

## What is it?
A List-type datastructure that can be used in place of a linked list or doubly linked list.
The TList is backed by an order-statistic Red-Black tree (CLRS ch. 13, 14.1) with asympototic O(log N) insert, delete, and indexing operations.
Implementation uses entirely safe code with a Vec-based Arena-style internal allocation, so should have similar cache-behavior as a Vec.
The TList provides both an Iterator and IntoIterator interface, but IterMut has been replaced with a more restricted closure-based traverse() function to allow systematic mutation without using unsafe code.

Named TList because it is a Tree-based List, and TbList would sound like a disease.
It's not supposed to be clever.

## What is it good for?
Generally speaking, the TList is a replacement for those rare cases where a Linked List might be appropriate.
Insertion into a Linked List is O(1) assuming you already have a reference to the insertion location, but search on a Linked List (whether by index or value) is O(N).
In particular, this means that random insertion or deletion is O(N).
So, Linked Lists are more effective than Arrays/Vectors when you're doing a lot of insertions *and* you have a way of finding the insertion location quickly, typically by holding a reference to the insert location.
Theoretically a Linked List and a Vector have the same performance when used as a Stack (pushing/popping at the end), but in practice cache-locality makes Vectors a better choice.
Vectors, of course, have O(N) insert time for random insertions due to copying/moving data.

In summary, Vectors are bad at insertion (and deletion), and Linked Lists are only conditionally good at insertion, while both provide O(N) iteration over the whole collection.
The TList in this crate provides a compromise focused on efficient random indexing, insert, and delete operations, which TList implements in O(log N) time with O(N) space overhead relative to a Vector.
This makes a TList worse than a Vector for indexing, but better than a Linked List, and is better than either a Vector or a Linked List for *random* insert and delete operations.
Additionally, the TList internally is just a pair of Vectors, so it should have better cache behavior than a Linked List.

### Okay, but I'm still confused about what data structure I want
I'll assume your data is ordered in some fashion.
If not, you want a probably want a hashmap or a set.

If your data is ordered by insertion order (e.g. the first thing in should be index 0), then you want a Vector, Linked List, or the TList here.
If you have a key that can be ordered, you're probably looking for a BTreeMap (or set).

So then we've got Vector vs Linked List vs TList.
Okay, there's also Deques, but you probably know if you need one.

Are you only pushing/popping to the end?
Use a Vector.

Do you need to read at random points in the data, but are only modifying from the end?
Use a Vector.

Do you need to bulk insert at one location?
Use a pair of Vectors as a Gap Buffer (for maximum efficiency), or use a TList.

Do you need to do a lot of insertions or deletions at random index locations?
Use a TList.

As a concrete example, I wrote TList in the first place because I wanted to manipulate a document as an array of Strings indexed by line number.
I have a list of Strings (the document) and a list of (line number, edit) pairs as a diff, so the TList was one reasonable solution to the edit problem.

### None of the examples show when to use a Linked List
Right. Don't do that.

### Upshot
If the above doesn't clarify whether you want to use TList or not, you probably want a Vector. 

## Dependencies
None for downstream users; the rand crate is used to generate randomized data for testing.

