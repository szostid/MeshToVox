use glam::Vec3;

use crate::scene::Triangle;

pub mod pbrless;

pub trait VertexData: Copy {
    fn pos(&self) -> [f32; 3];
    fn set_pos(&mut self, pos: [f32; 3]);
}

pub trait VoxelPipeline {
    type Vertex: VertexData;
    type VoxelData: Copy;

    type Material;

    type TriangleData<'a>;

    fn prepare_triangle<'a>(
        material: &'a Self::Material,
        triangle: &'a Triangle<Self::Vertex>,
    ) -> Self::TriangleData<'a>;

    fn sample_from_bary(data: &Self::TriangleData<'_>, bary: Vec3) -> Option<Self::VoxelData>;
}
