use crate::io::Vertex;
use crate::space_filling::*;
use glam::*;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct OctreePos {
    pub coords: IVec3,
    pub depth: u32,
}

impl OctreePos {
    pub const fn zero(depth: u32) -> Self {
        Self {
            coords: IVec3::ZERO,
            depth,
        }
    }

    pub const fn simplify(&self, max_depth: u32) -> Self {
        let cords = IVec3::new(
            self.coords.x & !((1 << (max_depth - self.depth)) - 1),
            self.coords.y & !((1 << (max_depth - self.depth)) - 1),
            self.coords.z & !((1 << (max_depth - self.depth)) - 1),
        );

        Self {
            coords: cords,
            depth: self.depth,
        }
    }

    pub const fn is_simple(&self, max_depth: u32) -> bool {
        const fn vaildate(dim: i32, depth: u32, max_depth: u32) -> bool {
            dim.trailing_zeros() >= (max_depth - depth)
        }

        let a = vaildate(self.coords.x, self.depth, max_depth);
        let b = vaildate(self.coords.y, self.depth, max_depth);
        let c = vaildate(self.coords.z, self.depth, max_depth);

        a && b && c
    }
}

#[derive(Debug, Clone)]
pub struct Octree {
    pub data: Vec<u32>,
    pub depth: u32,
}

pub const fn get_octree_idx(cords: IVec3, depth: u32) -> i32 {
    let x = (cords.x >> depth) & 1;
    let y = (cords.y >> depth) & 1;
    let z = (cords.z >> depth) & 1;

    x | (y << 1) | (z << 2)
}

impl Octree {
    pub const fn get_oct_inverted(&self, cords: IVec3, i: u32) -> i32 {
        let depth = self.depth - i;
        get_octree_idx(cords, depth)
    }

    pub fn store(&mut self, position: IVec3, val: image::Rgba<u8>) {
        let node = OctreePos {
            coords: position,
            depth: self.depth,
        };

        // bc floating point error some erroneous voxels
        if node.coords.min_element() < 1
            || node.coords.max_element() >= ((1 << (self.depth + 1)) - 1)
        {
            return;
        }

        self.insert(&node, val);
    }

    pub fn fill_space(&self, max_size: u32) -> Vec<Vertex> {
        let mut empty_tree = Octree::new(self.depth);
        let mut current = HashSet::new();
        let mut next = HashSet::new();

        let start = IVec3::ZERO;
        let depth = self.insert_max_start(&mut empty_tree, start);
        let start = OctreePos {
            coords: start,
            depth,
        };
        current.insert(start);

        'outer: loop {
            for cord in &current {
                for i in 0..6 {
                    let adjcent = self.min_adjcent_depth(&mut empty_tree, &mut next, cord, i);
                    if adjcent.is_none() {
                        continue;
                    }

                    let adjcent = adjcent.unwrap();
                    let mut thing = FillSpaceData {
                        empty_tree: &mut empty_tree,
                        next: &mut next,
                    };

                    self.recursive_collect(&adjcent, &mut thing);
                }
            }

            core::mem::swap(&mut current, &mut next);
            next.clear();
            if current.len() == 0 {
                break 'outer;
            }
        }

        let nodes = Self::empty_to_mesh(self, &empty_tree);

        let triangles = nodes
            .iter()
            .map(|(node, color)| {
                let color = color.0;
                let triangles = node.to_vertices(self.depth as u8);

                let mapping = |x: IVec3| {
                    let position = (x + IVec3::NEG_ONE).as_vec3() / max_size as f32;
                    let position = position.mul_add(Vec3::splat(2.0), Vec3::NEG_ONE);
                    Vertex { position, color }
                };

                let a = <[_; 3]>::try_from(&triangles[0..3]).unwrap().map(mapping);
                let b = <[_; 3]>::try_from(&triangles[0..3]).unwrap().map(mapping);

                [a, b]
            })
            .collect::<Vec<_>>();

        let triangles = bytemuck::cast_vec::<_, Vertex>(triangles);

        triangles
    }

    fn insert_max_start(&self, empty_tree: &mut Self, start: IVec3) -> u32 {
        let mut empty_pointer: u32 = 0;
        let mut filled_pointer: u32 = 0;

        for d in 0..=self.depth {
            let filled_header = self.data[filled_pointer as usize];
            let empty_header = &mut empty_tree.data[empty_pointer as usize];
            let oct = self.get_oct_inverted(start, d) as u32;

            //if octree_header::get_final(filled_header, oct as u32){panic!();}

            if !octree_header::get_exists(filled_header, oct as u32) {
                octree_header::set_final(empty_header, oct as u32);
                octree_header::set_exists(empty_header, oct as u32);

                return d;
            }

            if !octree_header::get_exists(*empty_header, oct as u32) {
                octree_header::set_exists(empty_header, oct as u32);

                let next = empty_tree.create_empty_oct(d);
                empty_tree.data[(empty_pointer + 1 + oct) as usize] = next as u32;
            }

            filled_pointer = self.data[(filled_pointer + 1 + oct) as usize];
            empty_pointer = empty_tree.data[(empty_pointer + 1 + oct) as usize];
        }

        panic!();
    }

    fn min_adjcent_depth(
        &self,
        empty: &mut Self,
        next: &mut CoordMap,
        cord: &OctreePos,
        side: u8,
    ) -> Option<FilledIterStruct> {
        let max_size = 1 << (self.depth + 1);
        let min_octant_size = 1 << (self.depth - cord.depth);

        let mut base = cord.coords;
        let dim = side % 3;
        base[dim as usize] += if side < 3 { min_octant_size } else { -1 };

        if (base[dim as usize] >= max_size) || (base[dim as usize] < 0) {
            return None;
        }

        let adjcent = base;

        let mut empty_offset: u32 = 0;
        let mut filled_offset: u32 = 0;

        for d in 0..(cord.depth + 1) {
            let adjacent_oct = self.get_oct_inverted(adjcent, d);

            let empty_header = empty.data[empty_offset as usize];
            let filled_header = self.data[filled_offset as usize];

            if octree_header::get_final(filled_header | empty_header, adjacent_oct as u32) {
                return None;
            }

            if !octree_header::get_exists(filled_header, adjacent_oct as u32) {
                let cord = OctreePos {
                    coords: base,
                    depth: d,
                };
                next.insert(cord);

                octree_header::set_exists(
                    &mut empty.data[empty_offset as usize],
                    adjacent_oct as u32,
                );
                octree_header::set_final(
                    &mut empty.data[empty_offset as usize],
                    adjacent_oct as u32,
                );

                return None;
            }

            if !octree_header::get_exists(empty_header, adjacent_oct as u32) {
                let next = empty.create_empty_oct(d);
                octree_header::set_exists(
                    &mut empty.data[empty_offset as usize],
                    adjacent_oct as u32,
                );
                empty.data[(empty_offset + 1 + adjacent_oct as u32) as usize] = next as u32;
            }

            empty_offset = empty.data[(empty_offset + 1 + adjacent_oct as u32) as usize];
            filled_offset = self.data[(filled_offset + 1 + adjacent_oct as u32) as usize];
        }

        let base = OctreePos {
            coords: adjcent,
            depth: (cord.depth + 1),
        };
        let new_cord = FilledIterStruct {
            cords: base,
            empty_offset,
            filled_offset,
            side,
        };

        return Some(new_cord);
    }

    fn recursive_collect(&self, adjcent: &FilledIterStruct, info: &mut FillSpaceData) {
        let empty_header = info.empty_tree.data[adjcent.empty_offset as usize];
        let filled_header = self.data[adjcent.filled_offset as usize];

        for oct in ALL_OCTREE_SIDES[adjcent.side as usize] {
            let oct = oct as u32;
            if octree_header::get_final(filled_header, oct) {
                continue;
            }
            if octree_header::get_final(empty_header, oct) {
                continue;
            }

            let pos = bit_toggle(adjcent.cords.coords, self.depth - adjcent.cords.depth, oct);
            let octant = OctreePos {
                coords: pos,
                depth: adjcent.cords.depth,
            };

            if !octree_header::get_exists(filled_header, oct) {
                octree_header::set_exists(
                    &mut info.empty_tree.data[adjcent.empty_offset as usize],
                    oct,
                );
                octree_header::set_final(
                    &mut info.empty_tree.data[adjcent.empty_offset as usize],
                    oct,
                );

                let out = octant.simplify(self.depth);
                info.next.insert(out);
                continue;
            }

            if !octree_header::get_exists(empty_header, oct) {
                octree_header::set_exists(
                    &mut info.empty_tree.data[adjcent.empty_offset as usize],
                    oct,
                );
                let next = info.empty_tree.create_empty_oct(adjcent.cords.depth);
                info.empty_tree.data[(adjcent.empty_offset + 1 + oct) as usize] = next as u32;
            }

            let filled_offset = self.data[(adjcent.filled_offset + 1 + oct) as usize];
            let empty_offset = info.empty_tree.data[(adjcent.empty_offset + 1 + oct) as usize];

            let next_octant = OctreePos {
                coords: pos,
                depth: adjcent.cords.depth + 1,
            };
            let new_adjcent = FilledIterStruct {
                cords: next_octant,
                filled_offset,
                empty_offset,
                side: adjcent.side,
            };
            self.recursive_collect(&new_adjcent, info);
        }
    }

    fn empty_to_mesh(filled: &Self, empty: &Self) -> Vec<(MeshNode, image::Rgba<u8>)> {
        let mut mesh = Vec::new();

        let nodes = filled.collect_nodes();
        let max_size = 1 << (filled.depth + 1);

        for (cord, value) in &nodes {
            let color = octree_header::to_color(*value);

            for i in 0..6 {
                let mut adjcent = cord.coords;
                let dim = (i / 2) as usize;
                let positive = (i % 2) == 0;

                adjcent[dim] += if positive { 1 } else { -1 };
                if adjcent[dim] >= max_size || adjcent[dim] < 0 {
                    continue;
                }
                let cords = adjcent;
                let node = OctreePos {
                    coords: cords,
                    depth: filled.depth,
                };

                if empty.contains_point(&node) {
                    let mesh_node = MeshNode {
                        cords: cord.coords,
                        dim: dim as u8,
                        positive,
                        depth: filled.depth as u8,
                    };
                    mesh.push((mesh_node, color));
                }
            }
        }

        mesh
    }

    fn create_new_empty_oct(&mut self) -> usize {
        let old_len = self.data.len();
        let mut header = 0;
        octree_header::set_header_tag(&mut header);
        self.data.push(header);

        old_len
    }

    fn create_empty_oct(&mut self, depth: u32) -> usize {
        if self.depth == depth {
            self.create_new_empty_oct()
        } else {
            self.create_new_oct(0)
        }
    }
}

pub mod octree_header {
    pub const EXISTS_OFFSET: u32 = 0;
    pub const FINAL_OFFSET: u32 = 8;
    pub const EMPTY_OFFSET: u32 = 16;
    pub const TAG_OFFSET: u32 = 24;

    pub const COLOR_TAG: u8 = 118;
    pub const HEADER_TAG: u8 = 68;

    pub const fn from_color(color: image::Rgba<u8>) -> u32 {
        let [r, g, b, a] = color.0;
        u32::from_le_bytes([r, g, b, a])
    }

    pub const fn to_color(offset: u32) -> image::Rgba<u8> {
        let [r, g, b, a] = offset.to_le_bytes();
        image::Rgba([r, g, b, a])
    }

    pub const fn get_empty(header: u32, idx: u32) -> bool {
        ((header >> (EMPTY_OFFSET + idx)) & 1) != 0
    }

    pub fn set_empty(header: &mut u32, idx: u32) {
        *header |= 1 << (EMPTY_OFFSET + idx);
    }

    pub fn set_header_tag(header: &mut u32) {
        *header |= (HEADER_TAG as u32) << TAG_OFFSET;
    }

    pub const fn is_header(header: u32) -> bool {
        (header >> TAG_OFFSET) == HEADER_TAG as u32
    }

    pub const fn get_exists(header: u32, idx: u32) -> bool {
        ((header >> (idx + EXISTS_OFFSET)) & 1) != 0
    }

    pub fn set_exists(header: &mut u32, idx: u32) {
        *header |= 1 << (idx + EXISTS_OFFSET);
    }

    pub const fn get_final(header: u32, idx: u32) -> bool {
        ((header >> (idx + FINAL_OFFSET)) & 1) != 0
    }

    pub fn set_final(header: &mut u32, idx: u32) {
        *header |= 1 << (idx + FINAL_OFFSET)
    }
}

#[derive(Debug, Clone)]
pub struct IterStruct {
    pub offset: u32,
    pub cords: OctreePos,
}

const fn gen_oct_perumations() -> [IVec3; 8] {
    let mut cube: [IVec3; 8] = [IVec3::new(0, 0, 0); 8];
    let mut counter: i32 = 0;

    while counter < 8 {
        cube[counter as usize] = IVec3::new(counter & 1, (counter >> 1) & 1, (counter >> 2) & 1);

        counter += 1;
    }

    cube
}

const fn gen_octree_sides() -> [[u8; 4]; 6] {
    const fn generate_mask(dim: u8, positive: bool) -> [u8; 4] {
        let mut counter = 0;
        let mut output_counter = 0;
        let mut output = [0_u8; 4];

        loop {
            if counter == 8 {
                break;
            }
            if ((counter >> dim) & 1) == (positive as u8) {
                output[output_counter] = counter;

                output_counter += 1;
            }

            counter += 1;
        }

        output
    }

    [
        generate_mask(0, false),
        generate_mask(1, false),
        generate_mask(2, false),
        generate_mask(0, true),
        generate_mask(1, true),
        generate_mask(2, true),
    ]
}

pub const ALL_OCTREE_SIDES: [[u8; 4]; 6] = gen_octree_sides();
pub const OCT_PERMS: [IVec3; 8] = gen_oct_perumations();

impl Octree {
    pub fn new(depth: u32) -> Self {
        let mut output = Self {
            depth,
            data: Vec::new(),
        };
        output.create_new_oct(0);

        output
    }

    pub fn create_new_oct(&mut self, mut header: u32) -> usize {
        self.data.reserve(9);
        let old_len = self.data.len();
        octree_header::set_header_tag(&mut header);

        unsafe {
            self.data.set_len(old_len + 9);
            self.data[old_len] = header;
            for i in 0..8 {
                self.data[old_len + 1 + i] = 69420420;
            }
        }
        old_len
    }

    pub fn contains_point(&self, node: &OctreePos) -> bool {
        let mut currnet_pointer: u32 = 0;
        let mut current_oct;
        let mut current_header;

        for d in 0..(node.depth + 1) {
            current_header = self.data[currnet_pointer as usize];
            current_oct = self.get_oct_inverted(node.coords, d) as u32;

            if !octree_header::get_exists(current_header, current_oct as u32) {
                return false;
            }
            if octree_header::get_final(current_header, current_oct as u32) {
                return true;
            }

            currnet_pointer = self.data[(currnet_pointer + 1 + current_oct) as usize];
        }
        false
    }

    pub fn contains_exact(&self, node: &OctreePos) -> bool {
        let mut currnet_pointer: u32 = 0;
        let mut current_oct;
        let mut current_header;

        for d in 0..node.depth {
            current_header = self.data[currnet_pointer as usize];
            current_oct = self.get_oct_inverted(node.coords, d) as u32;

            if !octree_header::get_exists(current_header, current_oct as u32) {
                return false;
            }
            if octree_header::get_final(current_header, current_oct as u32) {
                return false;
            }

            currnet_pointer = self.data[(currnet_pointer + 1 + current_oct) as usize];
        }

        current_header = self.data[currnet_pointer as usize];
        current_oct = self.get_oct_inverted(node.coords, node.depth) as u32;

        if octree_header::get_final(current_header, current_oct as u32) {
            return true;
        } else {
            false
        }
    }

    pub fn insert(&mut self, node: &OctreePos, value: image::Rgba<u8>) -> Option<u32> {
        if node.depth > self.depth {
            return None;
        }

        let mut current_pointer: u32 = 0;
        let mut current_oct = self.get_oct_inverted(node.coords, 0) as u32;
        let mut current_node = current_pointer + 1 + current_oct as u32;
        let mut inserted = true;

        for d in 0..node.depth {
            let current_header = self.data[current_pointer as usize];
            let next_oct = self.get_oct_inverted(node.coords, d + 1) as u32;

            current_pointer =
                if octree_header::get_exists(current_header, current_oct as u32) && inserted {
                    if octree_header::get_final(current_header, current_oct as u32) {
                        return None;
                    }

                    self.data[current_node as usize]
                } else {
                    let mut next_header = 0;
                    octree_header::set_exists(&mut next_header, next_oct as u32);
                    let next_pointer = self.create_new_oct(next_header) as u32;

                    octree_header::set_exists(
                        &mut self.data[current_pointer as usize],
                        current_oct as u32,
                    );
                    self.data[current_node as usize] = next_pointer;
                    inserted = false;

                    next_pointer
                };

            current_node = current_pointer + 1 + next_oct as u32;
            current_oct = next_oct;
        }

        let next_node = current_pointer + 1 + current_oct as u32;
        let current_header = self.data.get_mut(current_pointer as usize);

        let current_header = current_header.unwrap();

        if octree_header::get_exists(*current_header, current_oct as u32) && inserted {
            return None;
        }

        octree_header::set_exists(current_header, current_oct as u32);
        octree_header::set_final(current_header, current_oct as u32);

        self.data[next_node as usize] = octree_header::from_color(value);

        Some(next_node)
    }

    //replace with non recursive implementation
    fn collect_recursive(&self, nodes: &mut Vec<(OctreePos, u32)>, iter_level: IterStruct) {
        let header = self.data[iter_level.offset as usize];

        for i in 0..8 {
            if !octree_header::get_exists(header, i) {
                continue;
            }

            let scale = 1 << (self.depth - iter_level.cords.depth);
            let coords = OCT_PERMS[i as usize] * scale;
            let new_position = iter_level.cords.coords + coords;
            let offset = self.data[(iter_level.offset + 1 + i) as usize];

            if octree_header::get_final(header, i) {
                let cords = OctreePos {
                    coords: new_position,
                    depth: iter_level.cords.depth,
                };
                nodes.push((cords, offset));
            } else {
                let cords = OctreePos {
                    coords: new_position,
                    depth: iter_level.cords.depth + 1,
                };
                let new_iter = IterStruct { cords, offset };
                self.collect_recursive(nodes, new_iter);
            }
        }
    }

    pub fn collect_nodes(&self) -> Vec<(OctreePos, u32)> {
        let length = self.data.len() / 9;
        let mut collected: Vec<(OctreePos, u32)> = Vec::with_capacity(length);
        let cords = OctreePos {
            coords: IVec3::ZERO,
            depth: 0,
        };
        let first_iter = IterStruct { cords, offset: 0 };

        self.collect_recursive(&mut collected, first_iter);

        collected
    }
}
