//! In-memory representation of the [`Scene`].
use std::{ops::Range, sync::Arc};

use crate::pipelines::{VertexData, VoxelPipeline};
use glam::Vec3;
use image::RgbaImage;

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
        self.unpack(|vertex| Vec3::from_array(vertex.pos()))
    }

    #[inline]
    #[must_use]
    pub fn unpack<T>(&self, f: impl Fn(&V) -> T) -> [T; 3] {
        let [a, b, c] = &self.vertices;

        [f(a), f(b), f(c)]
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

/// Data about the albedo texture of the material
pub struct MaterialTexturing {
    /// The actual texture
    pub texture: Arc<RgbaImage>,
    /// Wrap modes for `u, v` respectively
    pub wrap_mode: [WrapMode; 2],
}

pub struct TriangleTextureData<'a> {
    pub texture: &'a RgbaImage,
    pub uvs: [[f32; 2]; 3],
    pub wrap: [WrapMode; 2],
}

impl MaterialTexturing {
    #[inline]
    #[must_use]
    pub fn as_triangle(&self, uvs: [[f32; 2]; 3]) -> TriangleTextureData<'_> {
        TriangleTextureData {
            texture: &self.texture,
            uvs,
            wrap: self.wrap_mode,
        }
    }
}

pub trait Interpolate: Sized {
    #[must_use]
    fn interpolate(data: [Self; 3], barycentrics: [f32; 3]) -> Self;
}

impl<T: Interpolate + Copy, const N: usize> Interpolate for [T; N] {
    #[inline]
    fn interpolate(data: [Self; 3], bary: [f32; 3]) -> Self {
        std::array::from_fn(|i| T::interpolate([data[0][i], data[1][i], data[2][i]], bary))
    }
}

impl Interpolate for f32 {
    #[inline]
    fn interpolate(data: [f32; 3], bary: [f32; 3]) -> Self {
        data[0] * bary[0] + data[1] * bary[1] + data[2] * bary[2]
    }
}

impl Interpolate for u8 {
    #[inline]
    fn interpolate(data: [u8; 3], bary: [f32; 3]) -> u8 {
        let float_data = [data[0] as f32, data[1] as f32, data[2] as f32];

        f32::interpolate(float_data, bary).clamp(0.0, 255.0) as u8
    }
}

pub struct Channel<'a, T: Interpolate> {
    vert_colors: [T; 3],
    albedo_texture: Option<TriangleTextureData<'a>>,
}
