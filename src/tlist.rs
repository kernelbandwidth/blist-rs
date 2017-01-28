// tlist.rs
//
// (c) 2017 James Crooks
//
// An indexable list structure based on Sedgewick's
// Left-Leaning Red-Black tree and order statistics. Uses
// sub-tree counts as a 'key' to achieve O(lg N) insertion,
// deletion, and access. Originally designed for rapid
// merge operations of a list with a sequence of indexed deltas.
//
// The LLRB paper can be found at:
// http://www.cs.princeton.edu/~rs/talks/LLRB/LLRB.pdf
//
//
// The key difference in the TList from a standard LLRB
// tree, and in particular from using a LLRB<usize, T>, e.g.
// using a usize 'index' as a key, is that the index value is implicit
// in TList rather than explicit like a normal key in a BST. If the index
// were used as an explicit key, then all nodes to the right
// of the insertion location would need their key adjusted upward
// by one, essentially reducing the LLRB<usize, T> to an array
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
     left_edge: usize,
     right_edge: usize,
     node_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Color {
    Red,
    Black,
}

#[derive(Debug, Clone)]
struct MapNode {
    loc: usize,
    left: Option<usize>,
    right: Option<usize>,
    left_count: usize,
    right_count: usize,
}

#[derive(Debug, Clone)]
struct Node<T: Sized> {
    data: T,
    color: Color,
    left: Option<usize>,
    right: Option<usize>,
    left_count: usize,
    right_count: usize,
}

impl<T> Node<T> where T: Sized {
    #[inline]
    fn new_leaf(elem: T, color: Color) -> Node<T> {
        Node {
            data: elem,
            color: color,
            left: None,
            right: None,
            left_count: 0,
            right_count: 0,
        }
    }

    fn get_map(&self, loc: usize) -> MapNode {
        MapNode {
            loc: loc,
            left: self.left,
            right: self.right,
            left_count: self.left_count,
            right_count: self.right_count,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TList<T: Sized> {
    node_list: Vec<Option<Node<T>>>,
    free_list: Vec<usize>,
    root_idx: usize,
    zipper: Vec<usize>, 
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
            zipper: Vec::with_capacity(capacity >> 1),
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
                left_edge: 0,
                right_edge: data.len(),
                node_idx: root_data_val_loc,
            });
        index_tree.root_idx = root_data_val_loc;

        while let Some(node_loc) = build_stack.pop() {
            let mut node = Node::new_leaf(data[node_loc.node_idx].clone(), node_loc.color);

            if let Some((loc, count)) = Self::prepare_left_child(&mut build_stack, &node_loc, data) {
                node.left = Some(loc);
                node.left_count = count;
            }

            if let Some((loc, count)) = Self::prepare_right_child(&mut build_stack, &node_loc, data) {
                node.right = Some(loc);
                node.right_count = count;
            }

            index_tree.node_list[node_loc.node_idx] = Some(node);
        }

        index_tree
    }
    
    pub fn insert(&mut self, elem: T, index: usize) {
        let node_slot = match self.free_list.pop() {
            Some(idx) => idx,
            None => {
                self.node_list.push(None); // allocate a new slot
                self.len() - 1
            }
        };

        let mut node = Node::new_leaf(elem, Color::Black);

        unimplemented!()
    }

    pub fn push(&mut self, elem: T) {
        let loc = self.len();
        self.insert(elem, loc);
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }

        let mut current_node = match self.node_list[self.root_idx] {
            Some(ref node) => node,
            None => return None,
        };

        let mut loc = current_node.left_count;
        
        while index != loc {
            if loc < index { // go right
                let right_idx = match current_node.right {
                    Some(idx) => idx,
                    None => return None,
                };

                current_node = match self.node_list[right_idx] {
                    Some(ref node) => node,
                    None => return None,
                };

                loc += current_node.left_count + 1;
            } else { // go left
                let left_idx = match current_node.left {
                    Some(idx) => idx,
                    None => return None,
                };

                current_node = match self.node_list[left_idx] {
                    Some(ref node) => node,
                    None => return None,
                };

                loc -= current_node.right_count + 1;
            }
        }

        Some(&current_node.data)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index > self.len() {
            return None;
        }

        let mut map_node = {
            let root_node = match self.node_list[self.root_idx] {
                Some(ref node) => node,
                None => return None,
            };

            root_node.get_map(self.root_idx)
        };

        let mut loc = map_node.left_count;

        while index != loc {
            if loc < index { //go right
                let right_idx = match map_node.right {
                    Some(idx) => idx,
                    None => return None,
                };

                map_node = match self.node_list[right_idx] {
                    Some(ref node) => node.get_map(right_idx),
                    None => return None,
                };

                loc += map_node.left_count + 1;

            } else { // go left
                let left_idx = match map_node.left {
                    Some(idx) => idx,
                    None => return None,
                };

                map_node = match self.node_list[left_idx] {
                    Some(ref node) => node.get_map(left_idx),
                    None => return None,
                };

                loc -= map_node.right_count + 1;
            }
        }

        let node = match self.node_list[map_node.loc] {
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
        let zipper_cap = self.len() >> 1;
        Iter {
            tree: self,
            zipper: Vec::with_capacity(zipper_cap),
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
        IntoIter {
            tree: self,
            loc: 0,
        }
    }

    // Private auxillary functions for implementing LLRB semantics
    fn left_rotate(&mut self, elem_index: usize) {

    }

    fn right_rotate(&mut self, elem_index: usize) {

    }

    #[inline]
    fn prepare_left_child<U>(build_stack: &mut Vec<NodeLoc>, loc_node: &NodeLoc, data: &[U]) 
        -> Option<(usize, usize)> where U: Sized + Clone {

        if loc_node.left_edge >= loc_node.node_idx {
            return None;
        }

        let new_left_loc = loc_node.left_edge + ((loc_node.node_idx - loc_node.left_edge) >> 1);
        let new_loc_node = NodeLoc {
            color: match loc_node.color {
                Color::Black => Color::Red,
                Color::Red => Color::Black,
            },
            left_edge: loc_node.left_edge,
            right_edge: loc_node.node_idx,
            node_idx: new_left_loc,
        };

        build_stack.push(new_loc_node);
        Some((new_left_loc, loc_node.node_idx - loc_node.left_edge))
    }

    #[inline]
    fn prepare_right_child<U>(build_stack: &mut Vec<NodeLoc>, loc_node: &NodeLoc, data: &[U])
        -> Option<(usize, usize)> where U: Sized + Clone {
        
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
                left_edge: new_left_edge,
                right_edge: loc_node.right_edge,
                node_idx: new_right_loc,
            };

            build_stack.push(new_loc);
            Some((new_right_loc, loc_node.right_edge - loc_node.node_idx - 1))
        }
}

pub struct Iter<'a, T: 'a> {
    tree: &'a TList<T>,
    zipper: Vec<usize>,
    loc: usize,
}

impl<'a, T> Iterator for Iter<'a, T> where T: 'a {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // naive iterator takes O(N log N) to complete
        // a more efficient implementation would be to
        // pre-compute the list of node indexes to 
        // traverse in order.
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

pub struct IntoIter<T> {
    tree: TList<T>,
    loc: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
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
            left_edge: 0,
            right_edge: test_data.len(),
            node_idx: test_data.len() >> 1,
        };

        let left_of_root = TList::<usize>::prepare_left_child(&mut build_stack, &root_node_loc, &test_data);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((2, 5)), left_of_root);

        let next_left_loc = build_stack.pop().unwrap();
        assert_eq!(2, next_left_loc.node_idx);

        let next_left = TList::<usize>::prepare_left_child(&mut build_stack, &next_left_loc, &test_data);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((1, 2)), next_left);

        let left_2_loc = build_stack.pop().unwrap();
        assert_eq!(1, left_2_loc.node_idx);
        let left_2 = TList::<usize>::prepare_left_child(&mut build_stack, &left_2_loc, &test_data);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((0, 1)), left_2);

        let left_final_loc = build_stack.pop().unwrap();
        assert_eq!(0, left_final_loc.node_idx);
        let left_final = TList::<usize>::prepare_left_child(&mut build_stack, &left_final_loc, &test_data);
        assert_eq!(0, build_stack.len());
        assert_eq!(None, left_final);
    }

    #[test]
    fn test_prepare_right_child() {
        let test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut build_stack = Vec::<NodeLoc>::with_capacity(DEFAULT_STACK_DEPTH);
        let root_node_loc = NodeLoc {
            color: Color::Black,
            left_edge: 0,
            right_edge: test_data.len(),
            node_idx: test_data.len() >> 1,
        };

        let right_of_root = TList::<usize>::prepare_right_child(&mut build_stack, &root_node_loc, &test_data);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((7, 4)), right_of_root);

        let next_right_loc = build_stack.pop().unwrap();
        assert_eq!(7, next_right_loc.node_idx);
        let right_2 = TList::<usize>::prepare_right_child(&mut build_stack, &next_right_loc, &test_data);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((8, 2)), right_2);

        let right_2_loc = build_stack.pop().unwrap();
        assert_eq!(8, right_2_loc.node_idx);
        let right_2 = TList::<usize>::prepare_right_child(&mut build_stack, &right_2_loc, &test_data);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((9, 1)), right_2);

        let right_final_loc = build_stack.pop().unwrap();
        assert_eq!(9, right_final_loc.node_idx);
        let right_final = TList::<usize>::prepare_right_child(&mut build_stack, &right_final_loc, &test_data);
        assert_eq!(0, build_stack.len());
        assert_eq!(None, right_final);
    }

    #[test]
    fn test_prepare_child_mixed() {
        let test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut build_stack = Vec::<NodeLoc>::with_capacity(DEFAULT_STACK_DEPTH);
        let root_node_loc = NodeLoc {
            color: Color::Black,
                left_edge: 0,
                right_edge: test_data.len(),
                node_idx: test_data.len() >> 1,
        };

        TList::<usize>::prepare_left_child(&mut build_stack, &root_node_loc, &test_data);
        let idx_2_loc = build_stack.pop().unwrap();
        let idx_3 = TList::<usize>::prepare_right_child(&mut build_stack, &idx_2_loc, &test_data);
        assert_eq!(1, build_stack.len());
        assert_eq!(Some((3, 2)), idx_3);

        let idx_4_loc = build_stack.pop().unwrap();
        let idx_4 = TList::<usize>::prepare_right_child(&mut build_stack, &idx_4_loc, &test_data);
        let idx_3_left = TList::<usize>::prepare_left_child(&mut build_stack, &idx_4_loc, &test_data);
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
                left_edge: 0,
                right_edge: test_data.len(),
                node_idx: test_data.len() >> 1,
        };

        TList::<usize>::prepare_left_child(&mut build_stack, &root_node_loc, &test_data);
        TList::<usize>::prepare_right_child(&mut build_stack, &root_node_loc, &test_data);
        assert_eq!(2, build_stack.len());
        assert_eq!(Some(NodeLoc {
            color: Color::Red,
            left_edge: 6,
            right_edge: 10,
            node_idx: 7
        }), build_stack.pop());
        assert_eq!(Some(NodeLoc {
            color: Color::Red,
            left_edge: 0,
            right_edge: 5,
            node_idx: 2
        }), build_stack.pop());

    }

    #[test]
    fn test_node_layout() {
        let mut test_data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let test_tree = TList::<usize>::from_data(&test_data);
        assert_eq!(test_tree.root_idx, 5);

        let mut expected_nodes = Vec::<Node<usize>>::with_capacity(10);

        // Node value 0, has no children
        expected_nodes.push(Node {
            data: 0usize,
            color: Color::Red,
            left: None,
            right: None,
            left_count: 0,
            right_count: 0,
        });

        // Node value 1, has left child 0
        expected_nodes.push(Node {
            data: 1usize,
            color: Color::Black,
            left: Some(0),
            right: None,
            left_count: 1,
            right_count: 0,
        });

        // Node value 2, has left 1, right 3
        expected_nodes.push(Node {
            data: 2usize,
            color: Color::Red,
            left: Some(1),
            right: Some(3),
            left_count: 2,
            right_count: 2,
        });

        // Node value 3, has right 4
        expected_nodes.push(Node {
            data:3usize,
            color: Color::Red,
            left: None,
            right: Some(4),
            left_count: 0,
            right_count: 1,
        });

        // Node value 4, has no children
        expected_nodes.push(Node {
            data: 4usize,
            color: Color::Black,
            left: None,
            right: None,
            left_count: 0,
            right_count: 0,
        });

        // Node value 5, root node, left 2 right 7
        expected_nodes.push(Node {
            data: 5usize,
            color: Color::Black,
            left: Some(2),
            right: Some(7),
            left_count: 5,
            right_count: 4,
        });

        // Node value 6, no children
        expected_nodes.push(Node {
            data: 6usize,
            color: Color::Black,
            left: None,
            right: None,
            left_count: 0,
            right_count: 0,
        });

        // Node value 7, left 6 right 8
        expected_nodes.push(Node {
            data: 7usize,
            color: Color::Red,
            left: Some(6),
            right: Some(8),
            left_count: 1,
            right_count: 2,
        });

        // Node value 8, right 9
        expected_nodes.push(Node {
            data: 8usize,
            color: Color::Black,
            left: None,
            right: Some(9),
            left_count: 0,
            right_count: 1,
        });

        // Node value 9, no children
        expected_nodes.push(Node {
            data: 9usize,
            color: Color::Red,
            left: None,
            right: None,
            left_count: 0,
            right_count: 0,
        });

        expected_nodes.iter().zip(test_tree.node_list.iter()).map(|(expect, r)| {
            let r_test = r.clone().unwrap();
            assert_eq!(expect.data, r_test.data);
            //assert_eq!(expect.color, r_test.color);
            assert_eq!(expect.left, r_test.left);
            assert_eq!(expect.right, r_test.right);
            assert_eq!(expect.left_count, r_test.left_count);
            assert_eq!(expect.right_count, r_test.right_count);
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

}
