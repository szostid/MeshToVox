use glam::Vec3;

use crate::scene::Triangle;

pub mod pbrless;

/// Represents a vertex of a pipeline.
///
/// Needs to store a position.
pub trait VertexData: Copy {
    /// Returns the position of the vertex
    #[must_use]
    fn pos(&self) -> [f32; 3];
    /// Sets the position of the vertex

    fn set_pos(&mut self, pos: [f32; 3]);
}

/// A sampler of a triangle associated with a pipeline.
///
/// The sampler takes in the properties of multiple vertices, and the
/// material info, and blends them according to the barycentric coordinates.
pub trait TriangleSampler<'a> {
    /// The data that the sampler returns
    type VoxelData: Copy;

    /// Samples the data of the voxel at the provided barycentric coordinates.
    #[must_use]
    fn sample_from_bary(&self, bary: Vec3) -> Option<Self::VoxelData>;
}

/// A rasterization pipeline.
///
/// The rasterization pipeline takes in an input vertex format, and a bunch of
/// materials, and returns the output voxel format.
pub trait VoxelPipeline {
    /// Input data provided by the input format.
    ///
    /// Used by the [`Self::TriangleSampler`] to derive the properties of a
    /// given voxel.
    type Vertex: VertexData;

    /// Output data consumed by the output format.
    ///
    /// Returned by the [`Self::TriangleSampler`] given the vertices of a
    /// triangle.
    type VoxelData: Copy;

    /// The material type of this pipeline. It should store all the data required
    /// to derive the [`Self::VoxelData`] from a given triangle.
    type Material;

    /// The sampler takes in the properties of multiple vertices, and the
    /// material info, and blends them according to the barycentric coordinates.
    type TriangleSampler<'a>: TriangleSampler<'a, VoxelData = Self::VoxelData>;

    /// Prepares the sampler of a triangle which is made out of a given material.
    #[must_use]
    fn prepare_sampler<'a>(
        material: &'a Self::Material,
        triangle: &'a Triangle<Self::Vertex>,
    ) -> Self::TriangleSampler<'a>;
}