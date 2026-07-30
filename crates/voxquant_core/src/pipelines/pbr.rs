//! The PBR pipeline.
//!
//! Stores the metallic, roughness, normal and emissive data
//! for every voxel apart from the usual color data.
use crate::pipelines::{TriangleSampler, VertexData, VoxelPipeline};
use crate::scene::Interpolate;
use crate::scene::{MaterialTexturing, Triangle, TriangleTextureData};

pub use image::RgbaImage;

/// A voxel with an associated color.
///
/// Used with the `pbrless` pipeline
#[derive(Clone, Copy)]
pub struct Voxel {
    pub color: [u8; 4],
    pub normal: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
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
    pub normal: [f32; 3],
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
    pub fn new(
        pos: [f32; 3],
        normal: Option<[f32; 3]>,
        uv: Option<[f32; 2]>,
        color: Option<[u8; 4]>,
    ) -> Self {
        Self {
            pos,
            uv: uv.unwrap_or([f32::NAN; 2]),
            color: color.unwrap_or([255, 255, 255, 255]),
            normal: normal.unwrap_or([0.0, 1.0, 0.0]),
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
    /// glTF ORM map: Green = Roughness, Blue = Metallic
    pub metallic_roughness_texture: Option<MaterialTexturing>,
    /// The color alpha threshold below which any voxels should be
    /// discarded. If not present, no discarding will happen.
    pub alpha_threshold: Option<u8>,
    /// The base color of the material. If the material is emissive,
    /// this will be the color of its emissive texture.
    pub base_color: [u8; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    /// Whether the material is emissive
    pub emissive: bool,
}

#[inline]
#[must_use]
fn multiply_colors(c1: [u8; 4], c2: [u8; 4]) -> [u8; 4] {
    std::array::from_fn(|i| ((c1[i] as u16 * c2[i] as u16) / 255) as u8)
}

pub struct TriangleData<'a> {
    vert_colors: [[u8; 4]; 3],
    vert_normals: [[f32; 3]; 3],

    albedo_texture: Option<TriangleTextureData<'a>>,
    orm_texture: Option<TriangleTextureData<'a>>,

    metallic_factor: f32,
    roughness_factor: f32,
    alpha_threshold: Option<u8>,
}

impl<'a> TriangleSampler<'a> for TriangleData<'a> {
    type VoxelData = Voxel;

    fn sample_from_bary(&self, mut bary: [f32; 3]) -> Option<Voxel> {
        bary = bary.map(|b| f32::max(b, 0.0));

        let sum = bary[0] + bary[1] + bary[2];
        if sum > f32::EPSILON {
            bary = bary.map(|b| b / sum);
        }

        let mut color = Interpolate::interpolate(self.vert_colors, bary);

        let mut normal = Interpolate::interpolate(self.vert_normals, bary);
        // normal must be normalized (interpolation may shorten it)
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > f32::EPSILON {
            normal = normal.map(|n| n / length);
        }

        let mut metallic = self.metallic_factor;
        let mut roughness = self.roughness_factor;

        if let Some(TriangleTextureData {
            texture,
            uvs,
            wrap: [wrap_u, wrap_v],
        }) = &self.albedo_texture
        {
            let mut uv = Interpolate::interpolate(*uvs, bary);
            uv[0] = wrap_u.apply(uv[0]);
            uv[1] = wrap_v.apply(uv[1]);

            let (w, h) = texture.dimensions();
            let x = (((w - 1) as f32) * uv[0]) as u32;
            let y = (((h - 1) as f32) * uv[1]) as u32;

            let tex_color = texture.get_pixel(x, y).0;
            color = multiply_colors(color, tex_color);
        }

        if let Some(threshold) = self.alpha_threshold {
            if color[3] < threshold {
                return None;
            }
        }

        if let Some(TriangleTextureData {
            texture,
            uvs,
            wrap: [wrap_u, wrap_v],
        }) = &self.orm_texture
        {
            let mut uv = Interpolate::interpolate(*uvs, bary);
            uv[0] = wrap_u.apply(uv[0]);
            uv[1] = wrap_v.apply(uv[1]);

            let (w, h) = texture.dimensions();
            let x = (((w - 1) as f32) * uv[0]) as u32;
            let y = (((h - 1) as f32) * uv[1]) as u32;

            let orm_pixel = texture.get_pixel(x, y).0;

            roughness *= orm_pixel[1] as f32 / 255.0;
            metallic *= orm_pixel[2] as f32 / 255.0;
        }

        Some(Voxel {
            color,
            normal,
            metallic,
            roughness,
        })
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
        let uvs = triangle.try_unpack(Vertex::uv).unwrap();

        let albedo_texture = material
            .texturing
            .as_ref()
            .map(|data| data.as_triangle(uvs));

        let orm_texture = material
            .metallic_roughness_texture
            .as_ref()
            .map(|data| data.as_triangle(uvs));

        TriangleData {
            albedo_texture,
            orm_texture,
            vert_colors: triangle.unpack(|v| multiply_colors(v.color, material.base_color)),
            vert_normals: triangle.unpack(|v| v.normal),
            alpha_threshold: material.alpha_threshold,
            metallic_factor: material.metallic_factor,
            roughness_factor: material.roughness_factor,
        }
    }
}
