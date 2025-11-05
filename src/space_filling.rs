use crate::octree::*;
use crate::*;
use std::collections::HashSet;

pub type CoordMap = HashSet<OctreePos>;

#[derive(Debug, Clone)]
pub struct MeshNode {
    pub cords: IVec3,
    pub dim: u8,
    pub positive: bool,
    pub depth: u8,
}

pub const fn bit_toggle(cords: IVec3, depth: u32, oct: u32) -> IVec3 {
    pub const fn thing(dim: i32, depth: u32, set: bool) -> i32 {
        if set {
            dim | (1 << depth)
        } else {
            dim & (!(1 << depth))
        }
    }

    IVec3::new(
        thing(cords.x, depth, ((oct >> 0) & 1) != 0),
        thing(cords.y, depth, ((oct >> 1) & 1) != 0),
        thing(cords.z, depth, ((oct >> 2) & 1) != 0),
    )
}

impl MeshNode {
    pub const fn to_square(&self, octree_depth: u8) -> [IVec3; 2] {
        let size = 1 << (octree_depth - self.depth);
        let mut base = self.cords;

        if self.positive {
            match self.dim {
                0 => base.x += size,
                1 => base.y += size,
                2 => base.z += size,
                _ => panic!(),
            }
        }

        let mut opposite = base;
        if self.dim != 0 {
            opposite.x += size
        }
        if self.dim != 1 {
            opposite.y += size
        }
        if self.dim != 2 {
            opposite.z += size
        }

        [base, opposite]
    }

    pub const fn to_vertices(&self, octree_depth: u8) -> [IVec3; 6] {
        let size = 1 << (octree_depth - self.depth);
        let [base, opposite] = self.to_square(octree_depth);

        let mut corner1 = base;
        if self.dim != 0 {
            corner1.x += size
        } else if self.dim != 1 {
            corner1.y += size
        }

        let mut corner2 = base;
        if self.dim != 2 {
            corner2.z += size
        } else if self.dim != 1 {
            corner2.y += size
        }

        [base, corner1, opposite, base, corner2, opposite]
    }
}

#[derive(Debug)]
pub struct FillSpaceData<'a, 'b> {
    pub next: &'a mut CoordMap,
    pub empty_tree: &'b mut Octree,
}

#[derive(Debug)]
pub struct FilledIterStruct {
    pub filled_offset: u32,
    pub empty_offset: u32,

    pub cords: OctreePos,
    pub side: u8,
}
