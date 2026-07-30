//! In-memory representation of the [`Scene`].
use std::ops::Range;

use crate::pipelines::{VertexData, VoxelPipeline};
use glam::Vec3;

/// A complete 3D scene with all the data required for voxelization.
pub struct Scene<P: VoxelPipeline> {
    /// All triangles contained within all instances of models of the scene.
    ///
    /// The scene does not distinguish models. If you have a model
    /// with multiple instances, you should just expand them all
    /// into different triangles.
    pub triangles: Vec<Triangle<P::Vertex>>,
    /// All materials contained within the scene
    pub materials: Vec<P::Material>,
    /// The bounding box of all triangles within the scene.
    ///
    /// During voxelization, the voxels (which should all be positioned
    /// within the bounding box of the scene) will be translated so
    /// that instead of starting at [`min`](BoundingBox::min) and ending at
    /// [`max`](BoundingBox::max), they will start at `0, 0, 0` and end at
    /// [`size`](BoundingBox::size) instead.
    pub bounds: BoundingBox,
}

/// A part of the scene.
pub struct SceneSlice<'a, P: VoxelPipeline> {
    /// The original, whole scene
    pub scene: &'a Scene<P>,
    /// The voxel range (in the scene's bounds!) that the scene
    /// spans over. Note that if you don't provide actual
    /// [`indices`](Self::indices) the voxelizer will still visit
    /// every triangle and discard most of it.
    pub range: Range<[i32; 3]>,
    /// The indices which the voxelizer should voxelize. Even if
    /// a triangle falls within the [`range`](Self::range), the
    /// voxelizer won't touch it. If no indices are provided,
    /// the voxelizer will visit every triangle in the scene, and
    /// discard most (if not all) of it.
    pub indices: Option<&'a [usize]>,
}

impl<P: VoxelPipeline> SceneSlice<'_, P> {
    pub fn for_each_triangle(&self, mut op: impl FnMut(Triangle<P::Vertex>)) {
        match self.indices {
            Some(indices) => {
                for &idx in indices {
                    op(self.scene.triangles[idx]);
                }
            }
            None => {
                for &tri in &self.scene.triangles {
                    op(tri);
                }
            }
        }
    }
}

/// Wrap mode of a texture
#[derive(Clone, Copy)]
pub enum WrapMode {
    /// Clamps every texture coordinates into the `[0, 1]` range.
    ClampToEdge = 1,
    /// Repeats the texture if UVs go out of the `[0, 1]` range,
    /// but mirrors it.
    MirroredRepeat,
    /// Repeats the texture if UVs go out of the `[0, 1]` range
    Repeat,
}

impl WrapMode {
    /// Applies the wrap mode onto a single coordinate `c`
    #[inline]
    #[must_use]
    pub fn apply(self, c: f32) -> f32 {
        match self {
            Self::ClampToEdge => c.clamp(0.0, 1.0),
            Self::Repeat => c.rem_euclid(1.0),
            Self::MirroredRepeat => {
                // we calculate as though UVs range from 0..2 and we just
                // flip the UVs in the 1..2 range to be mirrored
                let m = c.rem_euclid(2.0);
                if m > 1.0 { 2.0 - m } else { m }
            }
        }
    }
}

/// Triangle, defined by three vertices and a material that it uses.
#[derive(Clone, Copy)]
pub struct Triangle<V: VertexData> {
    /// The vertices of the triangle. Named `a, b, c` respectively
    /// in many parts of the code
    pub vertices: [V; 3],
    /// The material used by this triangle. This is
    /// an index into the scene's materials
    pub material_index: u32,
}

impl<V: VertexData> Triangle<V> {
    #[inline]
    #[must_use]
    pub(crate) fn unpack_vertices_to_glam(&self) -> [Vec3; 3] {
        self.vertices.map(|vertex| Vec3::from_array(vertex.pos()))
    }

    pub(crate) fn unpack<T>(&self, f: impl Fn(V) -> T) -> [T; 3] {
        self.vertices.map(|v| f(v))
    }
}

/// The bounding box of a scene.
///
/// During voxelization, the voxels (which should all be positioned within
/// the bounding box of the scene) will be translated so that instead of
/// starting at [`min`](Self::min) and ending at [`max`](Self::max), they
/// will start at `0, 0, 0` and end at [`size`](Self::size) instead.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// The smallest (minimum) point of the bounding box
    pub min: [f32; 3],
    /// The largest (maximum) point of the bounding box
    pub max: [f32; 3],
}

impl BoundingBox {
    /// Creates an empty bounding box with no volume.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        }
    }

    /// Returns true if no points have been added to this box.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0] || self.min[1] > self.max[1] || self.min[2] > self.max[2]
    }

    /// Extends the bounding box so that it contains the point `p`
    #[inline]
    pub const fn extend(&mut self, p: [f32; 3]) {
        self.min[0] = self.min[0].min(p[0]);
        self.min[1] = self.min[1].min(p[1]);
        self.min[2] = self.min[2].min(p[2]);
        self.max[0] = self.max[0].max(p[0]);
        self.max[1] = self.max[1].max(p[1]);
        self.max[2] = self.max[2].max(p[2]);
    }

    /// Returns the size of the bounding box (i.e. `max - min`).
    ///
    /// If the bounding box is [`empty`](Self::empty), returns [`Vec3::ZERO`]
    #[inline]
    #[must_use]
    pub fn size(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
}
