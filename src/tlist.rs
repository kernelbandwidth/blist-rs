// tlist.rs
//
// (c) 2017 James Crooks
//
// An indexable list structure based on a Red-Black tree that uses
// sub-tree counts as a 'key' to achieve O(lg N) insertion,
// deletion, and access. Originally designed for rapid
// merge operations of a list with a sequence of indexed deltas.
//
// The key difference in the TList from a standard Red-Black
// tree, and in particular from using a RBT<usize, T>, e.g.
// using a usize 'index' as a key, is that the index value is implicit
// in TList rather than explicit like a normal key in a BST. If the index
// were used as an explicit key, then all nodes to the right
// of the insertion location would need their key adjusted upward
// by one, essentially reducing the RBT<usize, T> to an array
// with O(lg N) search, and the same big-O behavior as an array
// for other operations. This is strictly worse than an array
// or vector.
//
// Expressing the index implicity allows TList to act like an Array or Vector
// with O(lg N) random insert and delete at the cost of O(lg N)
// random-access instead of O(1). This is more efficient
// in situations where reads are primarily via iteration
// over the whole collection, or ranges, but mutation
// is primarily through insert/delete, e.g. delta-merge
// operations.
// 
// Implemented internally using a Vec and indexes
// in an Arena-like fashion.

use std::mem;

const INITIAL_SIZE: usize = 256; // Initial number of nodes allocated by default
const DEFAULT_STACK_DEPTH: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
struct NodeLoc {
     color: Color,
     parent: Option<usize>,
     left_edge: usize,
     right_edge: usize,
     node_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Color {
    Red,
    Black,
}

impl Color {
    fn flip(&self) -> Color {
        match *self {
            Color::Black => Color::Red,
            Color::Red => Color::Black,
        }
    }
}

#[derive(Debug, Clone)]
struct Node<T: Sized> {
    data: T,
    color: Color,
    parent: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    size: usize,
}

impl<T> Node<T> where T: Sized {
    #[inline]
    fn new_leaf(elem: T, color: Color) -> Node<T> {
        Node {
            data: elem,
            color: color,
            parent: None,
            left: None,
            right: None,
            size: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Dir {
    Left,
    Right
}

#[derive(Debug, Clone)]
pub struct TList<T: Sized> {
    node_list: Vec<Option<Node<T>>>,
    free_list: Vec<usize>,
    root_idx: usize,
}

impl<T> TList<T> where T: Sized {
    // Public API

    pub fn new() -> TList<T> {
        Self::with_capacity(INITIAL_SIZE)
    }

    pub fn with_capacity(capacity: usize) -> TList<T> {
        let mut node_list = Vec::with_capacity(capacity);
        let mut free_list = Vec::with_capacity(capacity);
        for i in 0..capacity {
            node_list.push(None);
            free_list.push(i);
        }

        TList {
            node_list: node_list,
            free_list: free_list,
            root_idx: 0,
        }
    }

    pub fn len(&self) -> usize {
        // Return the number of filled nodes
        self.node_list.len() - self.free_list.len()
    }

    pub fn capacity(&self) -> usize {
        // Capacity is the number of free slots
        self.free_list.len()
    }

    // from_data() builds a new TList from a vector
    // of data elements, such that the inorder traversal
    // of the TList maintains the same ordering as
    // the original vector.
    pub fn from_data<U>(data: &[U]) -> TList<U> where U: Sized + Clone {
        // We allocate the node_list with capacity for data.len() items,
        // and empty the free_list since we will return a tree with
        // capacity exactly the same as the size of data, and all slots will
        // be filled.

        let mut index_tree = TList::with_capacity(data.len());
        index_tree.free_list.clear();

        // degenerate case of no data
        if 0 == data.len() {
            return index_tree;
        }
        
        // otherwise we choose the central data value as our root node value
        let root_data_val_loc = data.len() >> 1;

        // allocate a stack for storing next node locations to insert data into
        // and initialize with the data to insert the root node.
        let mut build_stack = Vec::<NodeLoc>::with_capacity(DEFAULT_STACK_DEPTH);
        build_stack.push(
            NodeLoc {
                color: Color::Black,
                parent: None,
                left_edge: 0,
                right_edge: data.len(),
                node_idx: root_data_val_loc,
            });
        index_tree.root_idx = root_data_val_loc;

        while let Some(node_loc) = build_stack.pop() {
            let mut node = Node::new_leaf(data[node_loc.node_idx].clone(), node_loc.color);

            if let Some((loc, count)) = Self::prepare_left_child(&mut build_stack, &node_loc) {
                node.left = Some(loc);
                node.size += count;
            }

            if let Some((loc, count)) = Self::prepare_right_child(&mut build_stack, &node_loc) {
                node.right = Some(loc);
                node.size += count;
            }
            node.parent = node_loc.parent;

            index_tree.node_list[node_loc.node_idx] = Some(node);
        }

        index_tree
    }
    
    pub fn insert(&mut self, elem: T, index: usize) {

        // Insert element into the node_list and get it's index
        let insert_idx = self.add_leaf(elem);
        
        // Get the node currently at the target index
        // panics if the index is not available
        // copied from search() but with mutability to update
        // sizes
        
        let target_idx = {
            if index > self.len() {
                return;
            }

            let mut rank_idx = index;
            let mut search_idx = self.root_idx;

            loop {
                self.node_list[search_idx]
                    .as_mut()
                    .map(|n| n.size += 1);

                let rank = self.get_child_size(search_idx, Dir::Left);
            
                if rank == rank_idx {
                    break;
                }

                search_idx = if rank_idx < rank {
                    match self.get_child_idx(search_idx, Dir::Left) {
                        Some(idx) => idx,
                        None => {
                            if cfg!(test) {
                                panic!("No left child!")
                            }
                            return;
                        },
                    }
                } else {
                    match self.get_child_idx(search_idx, Dir::Right) {
                        Some(idx) => {
                            rank_idx -= rank + 1;
                            idx
                        }
                        None => {
                            if cfg!(test) {
                                panic!("No right child from {} getting index {}!", search_idx, index)
                            }
                            return;
                        },
                    }
                };
            }
            search_idx
        };
        
        // We insert the new node to the left of the target node
        // If the left child is None, we insert there and proceed to
        // the fix-up stage.
        // If the left child is not None, then we go down the left
        // link and take every right link until we encouter an empty
        // right child and insert the leaf there.

        match self.get_child_idx(target_idx, Dir::Left) {
            Some(l_idx) => {
                let mut prev_idx = l_idx;
                let mut right_idx = self.get_child_idx(l_idx, Dir::Right);
                while let Some(idx) = right_idx {
                    prev_idx = idx;
                    right_idx = self.get_child_idx(idx, Dir::Right);
                    self.node_list[prev_idx]
                        .as_mut()
                        .map(|n| n.size += 1);
                }
                // At this point, right_idx is None, so prev_idx is a node with a free right child
                self.node_list[prev_idx]
                    .as_mut()
                    .unwrap()
                    .right = Some(insert_idx);
            },
            None => {
                self.node_list[target_idx]
                    .as_mut()
                    .unwrap()
                    .left = Some(insert_idx);
            }
        }

        self.insert_fix_up(insert_idx);
    }

    pub fn push(&mut self, elem: T) {
        let loc = self.len();
        self.insert(elem, loc);
    }

    pub fn insert_or_push(&mut self, elem: T, index: usize) {
        if index >= self.len() {
            self.push(elem);
        } else {
            self.insert(elem, index);
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        let target_idx = match self.search(index) {
            Some(idx) => idx,
            None => return None,
        };

        let node = match self.node_list[target_idx] {
            Some(ref node) => node,
            None => return None,
        };

        Some(&node.data)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let target_idx = match self.search(index) {
            Some(idx) => idx,
            None => return None,
        };

        let node = match self.node_list[target_idx] {
            Some(ref mut node) => node,
            None => return None
        };

        Some(&mut node.data)
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }

        unimplemented!()
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len() == 0 {
            None
        } else {
            let loc = self.len() - 1;
            self.remove(loc)
        }
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            tree: self,
            loc: 0
        }
    }

    pub fn traverse<F>(&mut self, f: &F) where F: Fn(&T) -> &T {
        let len = self.len();

        for i in 0..len {
            let mut old_entry = self.get(i);
            let mut new_entry = old_entry.map(f);
            mem::swap(&mut old_entry, &mut new_entry);
        }
    }

    pub fn into_iter(self) -> IntoIter<T> {
        let iter = IntoIter {
            tree: self,
            traversal_list: Vec::new(),
        };
        unimplemented!()
    }

    // Private auxillary functions for implementing Red-Black semantics

    #[inline]
    fn get_child_idx(&self, index: usize, dir: Dir) -> Option<usize> {
        match self.node_list[index] {
            Some(ref node) => match dir {
                Dir::Left => node.left,
                Dir::Right => node.right,
            },
            None => None,
        }
    }

    #[inline]
    fn get_grandparent_idx(&self, index: usize) -> Option<usize> {
        self.node_list[index].as_ref()
            .and_then(|n| n.parent)
            .and_then(|pidx| self.node_list[pidx].as_ref())
            .and_then(|n| n.parent)
    }

    #[inline]
    fn get_parent_idx(&self, index: usize) -> Option<usize> {
        self.node_list[index]
            .as_ref()
            .and_then(|n| n.parent)
    }

    #[inline]
    fn get_child_size(&self, index: usize, dir: Dir) -> usize {
        let child_idx = self.get_child_idx(index, dir);
        
        match child_idx {
            Some(c_idx) => {
                match self.node_list[c_idx] {
                    Some(ref node) => node.size,
                    None => {
                        if cfg!(test) {
                            panic!()
                        }
                        return 0;
                    },
                }
            },
            None => 0
        }
    }

    #[inline]
    fn insert_fix_up(&mut self, index: usize) {
        // Repairs invariants damaged during an insertion event
        // This should only be called after inserting a node, called
        // on the index of the newly inserted node
        //
        // Following CLRS, the current node is z, with index z_idx,
        // and is always Red by design. Since we inserted a Red node,
        // this is true at the start, and each time we move the z_idx ptr,
        // we move to the index of another red node (or we return).

        let mut z_idx = index;
        loop {
            let z_p = match self.get_parent_idx(z_idx) {
                Some(idx) => idx,
                None => break, // no parent => we're at the root
            };

            let red_parent = match self.node_list[z_p] {
                Some(ref node) => node.color == Color::Red,
                None => {
                    if cfg!(test) {
                        panic!()
                    }   
                    break;
                },
            };

            if !red_parent {
                break;
            }
        
            // Root is always black by construction, so a grandparent of z exists
            let gp_idx = self.get_grandparent_idx(z_idx).unwrap();

            // Get the uncle of z, y as an index
            let y_idx = {
                let gp_node = self.node_list[gp_idx]
                    .as_ref()
                    .unwrap();
                if gp_node.left == Some(z_p) {
                    gp_node.right
                } else {
                    gp_node.left
                }
            };

            // None is "black", as per CLRS: nil nodes are always black.
            let y_color = y_idx
                .and_then(|idx| self.node_list[idx].as_ref())
                .map(|n| n.color);

            if Some(Color::Red) == y_color {
                self.color_flip(gp_idx);
                z_idx = gp_idx;
            } else {
                let z_p_dir = {
                    if Some(z_p) == self.node_list[gp_idx].as_ref().and_then(|n| n.left) {
                        Dir::Left
                    } else {
                        Dir::Right
                    }
                };

                let z_dir = {
                    if Some(z_idx) == self.node_list[z_p].as_ref().and_then(|n| n.left) {
                        Dir::Left
                    } else {
                        Dir::Right
                    }
                };

                if z_dir != z_p_dir {
                    // CLRS case 2 fall through to case 3 
                    self.left_rotate(z_p);
                    z_idx = z_p;
                } 
                // CLRS only case 3
                self.get_parent_idx(z_idx)
                    .and_then(|idx| self.node_list[idx].as_mut())
                    .map(|n| n.color = Color::Black);

                let n_gp = match self.get_grandparent_idx(z_idx) {
                    Some(idx) => idx,
                    None => {
                        if cfg!(test) {
                            panic!()
                        }
                        break;
                    }
                };
                self.node_list[n_gp].as_mut()
                    .map(|n| n.color = Color::Red);

                self.right_rotate(gp_idx);
            }
        }

        self.node_list[self.root_idx].as_mut().unwrap().color = Color::Black;
            
    }

    #[inline]
    fn left_rotate(&mut self, h_idx: usize) -> usize {
        // Performs a left tree rotation of the node at h_idx, and returns the new index
        // of the entry node to the sub-tree

        // Fetch the current parent node and pull it out as an owned object
        // in the current scope. Replace it with a None so that the underlying
        // vector doesn't reshuffle.
        let h_node_opt = mem::replace(&mut self.node_list[h_idx], None);

        // left_rotate should not be called in a situation where either the h node
        // or right child don't exist, but left_rotate can't guarentee that directly,
        // so we go through the option matching, and replace the h_node and return if
        // something is missing. In test, we panic since this should violate the invariants.
        let (mut h_node, x_node_opt, x_idx) = match h_node_opt {
            Some(h_node) => {
                match h_node.right {
                    Some(x_idx) => {
                        let x_node = mem::replace(&mut self.node_list[x_idx], None);
                        (h_node, x_node, x_idx)
                    },
                    None => {
                        if cfg!(test) {
                            panic!();
                        }
                        mem::swap(&mut self.node_list[h_idx], &mut Some(h_node));
                        return h_idx;
                    },
                }
            },
            None => {
                // If h_node is None, then we don't need to replace it, since we put
                // a None in it's place above.
                // Also, this really shouldn't happen.
                if cfg!(test) {
                    panic!();
                }
                return h_idx;
            },
        };

        let mut x_node = match x_node_opt {
            Some(x_node) => x_node,
            None => {
                if cfg!(test) {
                    panic!();
                }
                mem::replace(&mut self.node_list[h_idx], Some(h_node));
                return h_idx;
            }
        };

        h_node.right = x_node.left;
        x_node.left = Some(h_idx);
        x_node.color = h_node.color;
        h_node.color = Color::Red;

        // Swap parent back-links
        mem::swap(&mut x_node.parent, &mut h_node.parent);

        
        // Re-insert the nodes into their positions in the node list;
        mem::replace(&mut self.node_list[h_idx], Some(h_node));
        mem::replace(&mut self.node_list[x_idx], Some(x_node));

        // reset size calculations
        // this has to be done after re-inserting to ensure child links work properly
        {
            let h_size = self.get_child_size(h_idx, Dir::Left) + self.get_child_size(h_idx, Dir::Right) + 1;
            self.node_list[h_idx]
                .as_mut()
                .unwrap()
                .size = h_size;
        }

        {
            let x_size = self.get_child_size(x_idx, Dir::Left) + self.get_child_size(x_idx, Dir::Right) + 1;
            self.node_list[x_idx]
                .as_mut()
                .unwrap()
                .size = x_size;
        }

        if let Some(xp_idx) = self.get_parent_idx(x_idx) {
            self.node_list[xp_idx]
                .as_mut()
                .map(|n| {
                    if n.left == Some(h_idx) {
                        n.left = Some(x_idx);
                    } else {
                        n.right = Some(x_idx);
                    }
                });
        }

        // if we did the rotation on the root node, we switch the root 
        if self.root_idx == h_idx {
            self.root_idx = x_idx;
        }

        x_idx
    }

    #[inline]
    fn right_rotate(&mut self, h_idx: usize) -> usize {
        // follows the same logic as left_rotate, properly mirror reversed
        let h_node_opt = mem::replace(&mut self.node_list[h_idx], None);

        let (mut h_node, x_node_opt, x_idx) = match h_node_opt {
            Some(h_node) => {
                match h_node.left {
                    Some(x_idx) => {
                        let x_node = mem::replace(&mut self.node_list[x_idx], None);
                        (h_node, x_node, x_idx)
                    },
                    None => {
                        if cfg!(test) {
                            panic!();
                        }
                        mem::swap(&mut self.node_list[h_idx], &mut Some(h_node));
                        return h_idx;
                    },
                }
            },
            None => {
                if cfg!(test) {
                    panic!();
                }
                return h_idx;
            },
        };

        let mut x_node = match x_node_opt {
            Some(x_node) => x_node,
            None => {
                if cfg!(test) {
                    panic!();
                }
                mem::replace(&mut self.node_list[h_idx], Some(h_node));
                return h_idx;
            }
        };

        h_node.left = x_node.right;
        x_node.right = Some(h_idx);
        x_node.color = h_node.color;
        h_node.color = Color::Red;

        mem::swap(&mut h_node.parent, &mut x_node.parent);

        mem::replace(&mut self.node_list[h_idx], Some(h_node));
        mem::replace(&mut self.node_list[x_idx], Some(x_node));

        {
            let h_size = self.get_child_size(h_idx, Dir::Left) + self.get_child_size(h_idx, Dir::Right) + 1;
            self.node_list[h_idx]
                .as_mut()
                .unwrap()
                .size = h_size;
        }

        {
            let x_size = self.get_child_size(x_idx, Dir::Left) + self.get_child_size(x_idx, Dir::Right) + 1;
            self.node_list[x_idx]
                .as_mut()
                .unwrap()
                .size = x_size;
        }

        if let Some(xp_idx) = self.get_parent_idx(x_idx) {
            self.node_list[xp_idx]
                .as_mut()
                .map(|n| {
                    if n.left == Some(h_idx) {
                        n.left = Some(x_idx);
                    } else {
                        n.right = Some(x_idx);
                    }
                });
        }

        if self.root_idx == h_idx {
            self.root_idx = x_idx;
        }

        x_idx
    }

    #[inline]
    fn color_flip(&mut self, elem_index: usize) {
        if elem_index > self.len() {
            return;
        }

        let (left, right) = {
            match self.node_list[elem_index] {
                Some(ref mut node) => {
                    node.color = node.color.flip();
                    (node.left, node.right)
                },
                None => return,
            }
        };

        if let Some(left_idx) = left {
            if let Some(ref mut lnode) = self.node_list[left_idx] {
                lnode.color = lnode.color.flip();
            }
        }

        if let Some(right_idx) = right {
            if let Some(ref mut rnode) = self.node_list[right_idx] {
                rnode.color = rnode.color.flip();
            }
        }
    }

    #[inline]
    fn search(&self, mut index: usize) -> Option<usize> {
        if index > self.len() {
            return None;
        }

        if index == self.root_idx {
            return Some(self.root_idx);
        }

        let mut search_idx = self.root_idx;

        loop {
            let rank = self.get_child_size(search_idx, Dir::Left);
            
            if rank == index {
                return Some(search_idx);
            }

            search_idx = if index < rank {
                match self.get_child_idx(search_idx, Dir::Left) {
                    Some(idx) => idx,
                    None => {
                        if cfg!(test) {
                            panic!("No left child!")
                        }
                        return None;
                    },
                }
            } else {
                match self.get_child_idx(search_idx, Dir::Right) {
                    Some(idx) => {
                        index -= rank + 1;
                        idx
                    }
                    None => {
                        if cfg!(test) {
                            panic!("No right child from {} getting index {}!", search_idx, index)
                        }
                        return None;
                    },
                }
            };
        }
    }

    #[inline]
    fn prepare_left_child(build_stack: &mut Vec<NodeLoc>, loc_node: &NodeLoc) -> Option<(usize, usize)> {

        if loc_node.left_edge >= loc_node.node_idx {
            return None;
        }

        let new_left_loc = loc_node.left_edge + ((loc_node.node_idx - loc_node.left_edge) >> 1);
        let new_loc_node = NodeLoc {
            color: match loc_node.color {
                Color::Black => Color::Red,
                Color::Red => Color::Black,
            },
            parent: Some(loc_node.node_idx),
            left_edge: loc_node.left_edge,
            right_edge: loc_node.node_idx,
            node_idx: new_left_loc,
        };

        build_stack.push(new_loc_node);
        Some((new_left_loc, loc_node.node_idx - loc_node.left_edge))
    }

    #[inline]
    fn prepare_right_child(build_stack: &mut Vec<NodeLoc>, loc_node: &NodeLoc) -> Option<(usize, usize)> {
        
            if loc_node.right_edge <= loc_node.node_idx + 1 {
                return None;
            }

            let new_left_edge = loc_node.node_idx + 1;
            let new_right_loc = loc_node.node_idx + ((loc_node.right_edge - loc_node.node_idx) >> 1);
            let new_loc = NodeLoc {
                color: match loc_node.color {
                    Color::Black => Color::Red,
                    Color::Red => Color::Black,
                },
                parent: Some(loc_node.node_idx),
                left_edge: new_left_edge,
                right_edge: loc_node.right_edge,
                node_idx: new_right_loc,
            };

            build_stack.push(new_loc);
            Some((new_right_loc, loc_node.right_edge - loc_node.node_idx - 1))
        }

    #[inline]
    fn realloc(&mut self) -> usize {
        let current_len = self.len();
        let additional = 1 + (current_len >> 1);
        self.node_list.reserve_exact(additional);
        for i in 0..additional {
            self.node_list.push(None);
            self.free_list.push(i + current_len);
        }
        self.free_list.pop().expect("Memory allocation failure.")
    }

    #[inline]
    fn add_leaf(&mut self, elem: T) -> usize {
        let leaf = Node::new_leaf(elem, Color::Red);
        let insert_idx = match self.free_list.pop() {
            Some(insert_idx) => insert_idx,
            None => self.realloc(),
        };
        mem::replace(&mut self.node_list[insert_idx], Some(leaf));
        insert_idx
    }
}

pub struct Iter<'a, T: 'a> {
    tree: &'a TList<T>,
    loc: usize,
}

impl<'a, T> Iterator for Iter<'a, T> where T: 'a {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // naive iterator takes O(N log N) to complete
        // to speed up testing of the core implementation
        // a more efficient implementation would be to
        // pre-compute the list of node indexes to 
        // traverse in order or use the parent pointers.
        if self.loc >= self.tree.len() {
            return None;
        }

        let nxt = self.tree.get(self.loc);
        self.loc += 1;
        nxt
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.tree.len() - self.loc;
        (exact, Some(exact))
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> where T: 'a {}

// The IntoIter will destroy the LLRB invariants as it consumes the
// tree. It pre-calculates the order of node traversal on construction
// and produces values by memory swapping 'None' values into the node_list
// slots similar to a deletion process, but without fixing up the tree
// since the affine type-system ensures the tree is inacessible to outside
// code and is dropped when the IntoIter goes out of scope.
pub struct IntoIter<T> {
    tree: TList<T>,
    traversal_list: Vec<usize>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.tree.len() - self.traversal_list.len();
        (exact, Some(exact))
    }
}

#[cfg(test)]
mod tests {
    extern crate rand;

    use super::{TList, Node, NodeLoc, Color, DEFAULT_STACK_DEPTH};

    use self::rand::Rng;

    #[test]
    fn build_from_data_test() {
        // Test that we don't crash...
        let test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let test_tree = TList::<usize>::from_data(&test_data);
        let mut idx = 0usize;
        for i in test_tree.node_list.iter() {
            let node = i.clone();
            assert_eq!(Some(idx), node.map(|n| n.data));
            idx += 1;
        }
    }

    #[test]
    fn test_prepare_left_child() {
        let test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut build_stack = Vec::<NodeLoc>::with_capacity(DEFAULT_STACK_DEPTH);
        let root_node_loc = NodeLoc {
            color: Color::Black,
            parent: None,
            left_edge: 0,
            right_edge: test_data.len(),
            node_idx: test_data.len() >> 1,
        };

        let left_of_root = TList::<usize>::prepare_left_child(&mut build_stack, &root_node_loc);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((2, 5)), left_of_root);

        let next_left_loc = build_stack.pop().unwrap();
        assert_eq!(2, next_left_loc.node_idx);
        assert_eq!(Some(5), next_left_loc.parent);

        let next_left = TList::<usize>::prepare_left_child(&mut build_stack, &next_left_loc);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((1, 2)), next_left);

        let left_2_loc = build_stack.pop().unwrap();
        assert_eq!(1, left_2_loc.node_idx);
        let left_2 = TList::<usize>::prepare_left_child(&mut build_stack, &left_2_loc);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((0, 1)), left_2);

        let left_final_loc = build_stack.pop().unwrap();
        assert_eq!(0, left_final_loc.node_idx);
        let left_final = TList::<usize>::prepare_left_child(&mut build_stack, &left_final_loc);
        assert_eq!(0, build_stack.len());
        assert_eq!(None, left_final);
    }

    #[test]
    fn test_prepare_right_child() {
        let test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut build_stack = Vec::<NodeLoc>::with_capacity(DEFAULT_STACK_DEPTH);
        let root_node_loc = NodeLoc {
            color: Color::Black,
            parent: Some(4),
            left_edge: 0,
            right_edge: test_data.len(),
            node_idx: test_data.len() >> 1,
        };

        let right_of_root = TList::<usize>::prepare_right_child(&mut build_stack, &root_node_loc);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((7, 4)), right_of_root);

        let next_right_loc = build_stack.pop().unwrap();
        assert_eq!(7, next_right_loc.node_idx);
        let right_2 = TList::<usize>::prepare_right_child(&mut build_stack, &next_right_loc);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((8, 2)), right_2);

        let right_2_loc = build_stack.pop().unwrap();
        assert_eq!(8, right_2_loc.node_idx);
        let right_2 = TList::<usize>::prepare_right_child(&mut build_stack, &right_2_loc);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((9, 1)), right_2);

        let right_final_loc = build_stack.pop().unwrap();
        assert_eq!(9, right_final_loc.node_idx);
        let right_final = TList::<usize>::prepare_right_child(&mut build_stack, &right_final_loc);
        assert_eq!(0, build_stack.len());
        assert_eq!(None, right_final);
    }

    #[test]
    fn test_prepare_child_mixed() {
        let test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut build_stack = Vec::<NodeLoc>::with_capacity(DEFAULT_STACK_DEPTH);
        let root_node_loc = NodeLoc {
            color: Color::Black,
            parent: None,
            left_edge: 0,
            right_edge: test_data.len(),
            node_idx: test_data.len() >> 1,
        };

        TList::<usize>::prepare_left_child(&mut build_stack, &root_node_loc);
        let idx_2_loc = build_stack.pop().unwrap();
        let idx_3 = TList::<usize>::prepare_right_child(&mut build_stack, &idx_2_loc);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((3, 2)), idx_3);

        let idx_4_loc = build_stack.pop().unwrap();
        let idx_4 = TList::<usize>::prepare_right_child(&mut build_stack, &idx_4_loc);
        let idx_3_left = TList::<usize>::prepare_left_child(&mut build_stack, &idx_4_loc);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((4, 1)), idx_4);
        assert_eq!(None, idx_3_left);
    }

    #[test]
    fn test_prepare_child_root() {
        let test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut build_stack = Vec::<NodeLoc>::with_capacity(DEFAULT_STACK_DEPTH);
        let root_node_loc = NodeLoc {
            color: Color::Black,
            parent: None,
            left_edge: 0,
            right_edge: test_data.len(),
            node_idx: test_data.len() >> 1,
        };

        TList::<usize>::prepare_left_child(&mut build_stack, &root_node_loc);
        TList::<usize>::prepare_right_child(&mut build_stack, &root_node_loc);
        assert_eq!(2, build_stack.len());
        assert_eq!(Some(NodeLoc {
            color: Color::Red,
            parent: Some(5),
            left_edge: 6,
            right_edge: 10,
            node_idx: 7
        }), build_stack.pop());
        assert_eq!(Some(NodeLoc {
            color: Color::Red,
            parent: Some(5),
            left_edge: 0,
            right_edge: 5,
            node_idx: 2
        }), build_stack.pop());

    }

    #[test]
    fn test_node_layout() {
        let test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let test_tree = TList::<usize>::from_data(&test_data);
        assert_eq!(test_tree.root_idx, 5);

        let mut expected_nodes = Vec::<Node<usize>>::with_capacity(10);

        // Node value 0, has no children
        expected_nodes.push(Node {
            data: 0usize,
            color: Color::Red,
            parent: Some(1),
            left: None,
            right: None,
            size: 1,
        });

        // Node value 1, has left child 0
        expected_nodes.push(Node {
            data: 1usize,
            color: Color::Black,
            parent: Some(2),
            left: Some(0),
            right: None,
            size: 2,
        });

        // Node value 2, has left 1, right 3
        expected_nodes.push(Node {
            data: 2usize,
            color: Color::Red,
            parent: Some(5),
            left: Some(1),
            right: Some(3),
            size: 5,
        });

        // Node value 3, has right 4
        expected_nodes.push(Node {
            data:3usize,
            color: Color::Black,
            parent: Some(2),
            left: None,
            right: Some(4),
            size: 2,
        });

        // Node value 4, has no children
        expected_nodes.push(Node {
            data: 4usize,
            color: Color::Red,
            parent: Some(3),
            left: None,
            right: None,
            size: 1,
        });

        // Node value 5, root node, left 2 right 7
        expected_nodes.push(Node {
            data: 5usize,
            color: Color::Black,
            parent: None,
            left: Some(2),
            right: Some(7),
            size: 10,
        });

        // Node value 6, no children
        expected_nodes.push(Node {
            data: 6usize,
            color: Color::Black,
            parent: Some(7),
            left: None,
            right: None,
            size: 1,
        });

        // Node value 7, left 6 right 8
        expected_nodes.push(Node {
            data: 7usize,
            color: Color::Red,
            parent: Some(5),
            left: Some(6),
            right: Some(8),
            size: 4,
        });

        // Node value 8, right 9
        expected_nodes.push(Node {
            data: 8usize,
            color: Color::Black,
            parent: Some(7),
            left: None,
            right: Some(9),
            size: 2,
        });

        // Node value 9, no children
        expected_nodes.push(Node {
            data: 9usize,
            color: Color::Red,
            parent: Some(8),
            left: None,
            right: None,
            size: 1,
        });

        expected_nodes.iter().zip(test_tree.node_list.iter()).map(|(expect, r)| {
            let r_test = r.clone().unwrap();
            assert_eq!(expect.data, r_test.data);
            assert_eq!(expect.color, r_test.color);
            assert_eq!(expect.parent, r_test.parent);
            assert_eq!(expect.left, r_test.left);
            assert_eq!(expect.right, r_test.right);
            assert_eq!(expect.size, r_test.size);
        }).collect::<Vec<()>>();
    }

    #[test]
    fn test_get() {
        let test_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let test_tree = TList::<usize>::from_data(&test_data);

        assert_eq!(test_tree.root_idx, 5);
        assert_eq!(test_tree.len(), 10);
        assert_eq!(test_tree.node_list.len(), 10);

        assert_eq!(None, test_tree.get(11));
        assert_eq!(Some(&5), test_tree.get(5));

        for i in 0..10 {
            assert_eq!(Some(&i), test_tree.get(i));
        }
    }

    #[test]
    fn test_get_randomized() {
        let mut test_data: Vec<i32> = Vec::with_capacity(10000);
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            test_data.push(rng.gen::<i32>());
        }

        let test_tree = TList::<i32>::from_data(&test_data);

        for i in 0..10000 {
            assert_eq!(Some(&test_data[i]), test_tree.get(i));
        }
    }

    #[test]
    fn test_get_mut_randomized() {
        let mut test_data: Vec<i32> = Vec::with_capacity(10000);
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            test_data.push(rng.gen::<i32>());
        }

        let mut test_tree = TList::<i32>::from_data(&test_data);

        for i in 0..10000 {
            assert_eq!(Some(&mut test_data[i]), test_tree.get_mut(i));
        }
    }

    #[test]
    fn test_iter_rand() {
        let mut test_data: Vec<i32> = Vec::with_capacity(10000);
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            test_data.push(rng.gen::<i32>());
        }

        let test_list = TList::<i32>::from_data(&test_data);
        let test_iter = test_list.iter();
        test_iter
            .enumerate()
            .map(|(idx, data_val)| assert_eq!(&test_data[idx], data_val))
            .collect::<Vec<()>>();
    }

    #[test]
    fn test_rotations() {
        let mut test_node_list = Vec::<Option<Node<usize>>>::with_capacity(5);
        test_node_list.push(Some(Node {
            data: 0,
            color: Color::Red,
            parent: Some(1),
            left: None,
            right: None,
            size: 1,
        }));
        test_node_list.push(Some(Node {
            data: 1,
            color: Color::Black,
            parent: None,
            left: Some(0),
            right: Some(3),
            size: 5,
        }));
        test_node_list.push(Some(Node {
            data: 2,
            color: Color::Black,
            parent: Some(1),
            left: None,
            right: None,
            size: 1,
        }));
        test_node_list.push(Some(Node {
            data: 3,
            color: Color::Red,
            parent: Some(1),
            left: Some(2),
            right: Some(4),
            size: 3,
        }));
        test_node_list.push(Some(Node {
            data: 4,
            color: Color::Black,
            parent: Some(3),
            left: None,
            right: None,
            size: 1,
        }));

        let free_list = Vec::new();
        let mut test_tree = TList::<usize> {
            node_list: test_node_list,
            free_list: free_list,
            root_idx: 1,
        };

        assert_eq!(3, test_tree.left_rotate(1));
        assert_eq!(3, test_tree.root_idx);
        {
            let nref = test_tree.node_list[test_tree.root_idx].clone();
            assert_eq!(None, nref.unwrap().parent);
        }

        test_tree
            .iter()
            .enumerate()
            .map(|(expected, got)| assert_eq!(expected, *got))
            .collect::<Vec<()>>();

        assert_eq!(1, test_tree.right_rotate(3));
        {
            let nref = test_tree.node_list[test_tree.root_idx].clone();
            assert_eq!(None, nref.unwrap().parent);
        }

        test_tree
            .iter()
            .enumerate()
            .map(|(expected, got)| assert_eq!(expected, *got))
            .collect::<Vec<()>>();
    }

    #[test]
    fn test_color_flip() {
        let test_data = vec![0i32, 1i32, 2i32];
        let mut test_list = TList::<i32>::from_data(&test_data); 
        test_list.color_flip(1);
        match test_list.node_list[0] {
            Some(ref node) => assert_eq!(Color::Black, node.color),
            None => panic!(),
        };

        match test_list.node_list[1] {
            Some(ref node) => assert_eq!(Color::Red, node.color),
            None => panic!(),
        };

        match test_list.node_list[2] {
            Some(ref node) => assert_eq!(Color::Black, node.color),
            None => panic!(),
        };
    }

    #[test]
    fn test_insert() {
        let test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 6, 7, 8, 9];
        let mut test_tree = TList::<usize>::from_data(&test_data);
        test_tree.insert(5, 5);
        test_tree
            .iter()
            .enumerate()
            .map(|(expect, got)| assert_eq!(expect, *got))
            .collect::<Vec<()>>();
    }
}
