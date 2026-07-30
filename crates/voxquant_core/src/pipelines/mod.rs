use glam::Vec3;

use crate::scene::Triangle;

pub mod pbrless;

pub trait VertexData: Copy {
    fn pos(&self) -> [f32; 3];
}

pub trait VoxelPipeline {
    type Vertex: VertexData;
    type Material;

    type TriangleData<'a>;

    fn prepare_triangle<'a>(
        material: &'a Self::Material,
        triangle: &'a Triangle<Self::Vertex>,
    ) -> Self::TriangleData<'a>;
}
