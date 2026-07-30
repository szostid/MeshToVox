use crate::pipelines::{TriangleSampler, VertexData, VoxelPipeline};
use crate::scene::{MaterialTexturing, Triangle, WrapMode};
use glam::{Vec2, Vec3, Vec4};

pub use image::RgbaImage;

/// A voxel with an associated color.
///
/// Used with the `pbrless` pipeline
#[derive(Clone, Copy)]
pub struct Voxel {
    pub color: [u8; 4],
}

/// A vertex with some associated color and UV (if present) data.
///
/// Used with the `pbrless` pipeline
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    /// Position of the vertex.
    ///
    /// The exact origin of the vertex space is unspecified.
    /// It will be translated by the voxelizer based on
    /// the [`BoundingBox`] of the scene.
    pub pos: [f32; 3],
    /// The base color of the vertex. If a texture is
    /// present, its color will be tinted by this field.
    pub color: [u8; 4],
    /// [`Vec2::NAN`] if UV's not present
    uv: [f32; 2],
}

impl VertexData for Vertex {
    #[inline]
    fn pos(&self) -> [f32; 3] {
        self.pos
    }

    fn set_pos(&mut self, pos: [f32; 3]) {
        self.pos = pos;
    }
}

impl Vertex {
    /// Creates a new vertex
    #[inline]
    #[must_use]
    pub fn new(pos: [f32; 3], uv: Option<[f32; 2]>, color: Option<[u8; 4]>) -> Self {
        Self {
            pos,
            uv: uv.unwrap_or([f32::NAN; 2]),
            color: color.unwrap_or([255, 255, 255, 255]),
        }
    }

    /// Returns the UV coordinates of this vertex, if they were provided
    /// when the vertex was created.
    #[inline]
    #[must_use]
    pub fn uv(&self) -> Option<[f32; 2]> {
        (!self.uv[0].is_nan()).then_some(self.uv)
    }
}

/// Determines how a mesh's color and emission are rendered.
pub struct Material {
    /// Data about the albedo texture of the material
    pub texturing: Option<MaterialTexturing>,
    /// The color alpha threshold below which any voxels should be
    /// discarded. If not present, no discarding will happen.
    pub alpha_threshold: Option<u8>,
    /// The base color of the material. If the material is emissive,
    /// this will be the color of its emissive texture.
    pub base_color: [u8; 4],
    /// Whether the material is emissive
    pub emissive: bool,
}

#[inline]
#[must_use]
fn interpolate_color(colors: [[u8; 4]; 3], bary: Vec3) -> [u8; 4] {
    let c0 = Vec4::from_array(colors[0].map(|c| c as f32));
    let c1 = Vec4::from_array(colors[1].map(|c| c as f32));
    let c2 = Vec4::from_array(colors[2].map(|c| c as f32));

    let final_color = c0 * bary.x + c1 * bary.y + c2 * bary.z;

    final_color.as_u8vec4().to_array()
}

#[inline]
#[must_use]
fn multiply_colors(c1: [u8; 4], c2: [u8; 4]) -> [u8; 4] {
    std::array::from_fn(|i| ((c1[i] as u16 * c2[i] as u16) / 255) as u8)
}

struct TriangleTextureData<'a> {
    pub texture: &'a RgbaImage,
    pub uvs: [Vec2; 3],
    pub wrap: [WrapMode; 2],
}

pub struct TriangleData<'a> {
    vert_colors: [[u8; 4]; 3],
    texture: Option<TriangleTextureData<'a>>,
    alpha_threshold: Option<u8>,
}

impl<'a> TriangleSampler<'a> for TriangleData<'a> {
    type VoxelData = Voxel;

    fn sample_from_bary(&self, mut bary: Vec3) -> Option<Voxel> {
        bary = bary.max(Vec3::ZERO);

        let sum = bary.x + bary.y + bary.z;
        if sum > f32::EPSILON {
            bary /= sum;
        }

        let mut color = interpolate_color(self.vert_colors, bary);

        if let Some(TriangleTextureData {
            texture,
            uvs,
            wrap: [wrap_u, wrap_v],
        }) = &self.texture
        {
            let mut uv = (uvs[0] * bary.x) + (uvs[1] * bary.y) + (uvs[2] * bary.z);
            uv.x = wrap_u.apply(uv.x);
            uv.y = wrap_v.apply(uv.y);

            let (w, h) = texture.dimensions();
            let x = (((w - 1) as f32) * uv.x) as u32;
            let y = (((h - 1) as f32) * uv.y) as u32;

            let tex_color = texture.get_pixel(x, y).0;
            color = multiply_colors(color, tex_color);
        }

        if let Some(threshold) = self.alpha_threshold
            && color[3] < threshold
        {
            return None;
        }

        Some(Voxel { color })
    }
}

pub struct Pipeline;

impl VoxelPipeline for Pipeline {
    type Vertex = Vertex;
    type VoxelData = Voxel;

    type Material = Material;

    type TriangleSampler<'a> = TriangleData<'a>;

    fn prepare_sampler<'a>(
        material: &'a Self::Material,
        triangle: &Triangle<Vertex>,
    ) -> Self::TriangleSampler<'a> {
        let texture = material.texturing.as_ref().map(|data| TriangleTextureData {
            texture: &data.texture,
            uvs: triangle
                .try_unpack(|v| v.uv().map(Vec2::from_array))
                .unwrap(),
            wrap: data.wrap_mode,
        });

        TriangleData {
            texture,
            vert_colors: triangle.unpack(|v| multiply_colors(v.color, material.base_color)),
            alpha_threshold: material.alpha_threshold,
        }
    }
}
