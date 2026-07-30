//! `glTF 2.0` input support for [`voxquant_core`] through the [`gltf`](https://docs.rs/gltf/latest/gltf/) crate
use glam::{Mat4, Vec3};
use image::RgbaImage;
use std::path::PathBuf;
use std::sync::Arc;
use voxquant_core::io::SceneReader;
use voxquant_core::pipelines::VoxelPipeline;
use voxquant_core::scene::{BoundingBox, MaterialTexturing, Triangle};
use voxquant_core::scene::{Scene, WrapMode};
use voxquant_core::{Format, InputFormat};

mod error;
pub use error::Error;

mod pbr;
mod pbrless;

trait GltfPipeline: VoxelPipeline {
    type MaterialExtras;

    const USES_NORMALS: bool;

    /// Builds the pipeline-specific material from a glTF material.
    fn parse_material(
        mat: &gltf::Material,
        image_data: &[Arc<RgbaImage>],
    ) -> Result<(Self::Material, Self::MaterialExtras)>;

    /// Provides a fallback material if parsing fails or an index is missing.
    fn fallback_material() -> (Self::Material, Self::MaterialExtras);

    fn get_uv_channel(extras: &Self::MaterialExtras) -> u32;

    /// Constructs the pipeline's vertex. The pipeline can choose to ignore
    /// normals or colors if it doesn't support them.
    fn create_vertex(
        pos: [f32; 3],
        normal: Option<[f32; 3]>,
        uv: Option<[f32; 2]>,
        color: Option<[u8; 4]>,
    ) -> Self::Vertex;
}

/// Result type for convenience.
pub type Result<T> = std::result::Result<T, Error>;

struct MeshInstance<'a> {
    mesh: gltf::Mesh<'a>,
    transform: Mat4,
}

fn collect_instances<'a>(
    node: &gltf::Node<'a>,
    parent_transform: Mat4,
    instances: &mut Vec<MeshInstance<'a>>,
) {
    let local_matrix = Mat4::from_cols_array_2d(&node.transform().matrix());

    let global_transform = parent_transform * local_matrix;

    if let Some(mesh) = node.mesh() {
        instances.push(MeshInstance {
            mesh,
            transform: global_transform,
        });
    }

    for child in node.children() {
        collect_instances(&child, global_transform, instances);
    }
}

#[profiling::function]
fn convert_image(data: gltf::image::Data) -> Result<Arc<RgbaImage>> {
    use bytemuck::Pod;
    use gltf::image::Format;
    use image::Pixel;
    use image::buffer::ConvertBuffer;
    use image::{ImageBuffer, Luma, LumaA, Rgb, Rgba, RgbaImage};

    /// Given `data`, converts the image (assumed to have
    /// the pixel format `P`) into an Rgba8 image
    fn convert_image<P: Pixel>(data: gltf::image::Data) -> Result<Arc<RgbaImage>>
    where
        P::Subpixel: Pod,
        ImageBuffer<P, Vec<P::Subpixel>>: ConvertBuffer<RgbaImage>,
    {
        fn convert_data<T: Pod>(data: Vec<u8>) -> Vec<T> {
            match bytemuck::try_cast_vec::<u8, T>(data) {
                Ok(data) => data,
                Err((_, data)) => bytemuck::pod_collect_to_vec(&data),
            }
        }

        let pixels = convert_data(data.pixels);

        ImageBuffer::from_vec(data.width, data.height, pixels)
            .ok_or(Error::InvalidImageDimensions)
            .map(|img| Arc::new(img.convert()))
    }

    match data.format {
        Format::R32G32B32FLOAT => convert_image::<Rgb<f32>>(data),
        Format::R32G32B32A32FLOAT => convert_image::<Rgba<f32>>(data),

        Format::R16 => convert_image::<Luma<u16>>(data),
        Format::R16G16 => convert_image::<LumaA<u16>>(data),
        Format::R16G16B16 => convert_image::<Rgb<u16>>(data),
        Format::R16G16B16A16 => convert_image::<Rgba<u16>>(data),

        Format::R8 => convert_image::<Luma<u8>>(data),
        Format::R8G8 => convert_image::<LumaA<u8>>(data),
        Format::R8G8B8 => convert_image::<Rgb<u8>>(data),
        Format::R8G8B8A8 => RgbaImage::from_vec(data.width, data.height, data.pixels)
            .ok_or(Error::InvalidImageDimensions)
            .map(Arc::new),
    }
}

#[profiling::function]
fn parse_image(image_data: &[Arc<RgbaImage>], texture: gltf::Texture) -> Result<Arc<RgbaImage>> {
    let image_index = texture.source().index();

    let image = image_data.get(image_index).ok_or(Error::OutOfBounds)?;

    Ok(Arc::clone(image))
}

struct GltfTexturingExtras {
    tex_coord: u32,
}

fn get_texture_data(
    texture_info: &gltf::texture::Info<'_>,
    image_data: &[Arc<RgbaImage>],
) -> Result<(MaterialTexturing, GltfTexturingExtras)> {
    const fn into_voxelization_mode(value: gltf::texture::WrappingMode) -> WrapMode {
        match value {
            gltf::texture::WrappingMode::ClampToEdge => WrapMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => WrapMode::MirroredRepeat,
            gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
        }
    }

    let texture_index = texture_info.texture().source().index();
    let texture = image_data.get(texture_index).ok_or(Error::OutOfBounds)?;

    Ok((
        MaterialTexturing {
            texture: Arc::clone(texture),
            wrap_mode: [
                into_voxelization_mode(texture_info.texture().sampler().wrap_s()),
                into_voxelization_mode(texture_info.texture().sampler().wrap_t()),
            ],
        },
        GltfTexturingExtras {
            tex_coord: texture_info.tex_coord(),
        },
    ))
}

#[derive(Default)]
struct MeshScratch {
    positions: Vec<[f32; 3]>,
    /// Normals are stored even if the pipeline doesn't
    /// [`GltfPipeline::USES_NORMALS`], but it will be
    /// empty in that case
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[u8; 4]>,
    indices: Vec<u32>,
}

#[profiling::function]
#[expect(
    clippy::cast_possible_truncation,
    reason = "safe to assume that neither material indices or triangle indices will be larger than usize"
)]
fn parse_mesh_instance<P: GltfPipeline>(
    instance: MeshInstance,
    bounds: &mut BoundingBox,
    materials: &[P::Material],
    material_extras: &[P::MaterialExtras],
    buffers: &[gltf::buffer::Data],
    triangles: &mut Vec<Triangle<P::Vertex>>,
    scratch: &mut MeshScratch,
) -> Result<()> {
    fn push_triangle<P: GltfPipeline>(
        [i1, i2, i3]: [u32; 3],
        triangles: &mut Vec<Triangle<P::Vertex>>,
        scratch: &MeshScratch,
        material_index: u32,
    ) {
        let i1 = i1 as usize;
        let i2 = i2 as usize;
        let i3 = i3 as usize;

        // check for malformed indices
        if i1 >= scratch.positions.len()
            || i2 >= scratch.positions.len()
            || i3 >= scratch.positions.len()
        {
            return;
        }

        triangles.push(Triangle {
            vertices: [
                P::create_vertex(
                    scratch.positions[i1],
                    scratch.normals.get(i1).copied(),
                    scratch.uvs.get(i1).copied(),
                    scratch.colors.get(i1).copied(),
                ),
                P::create_vertex(
                    scratch.positions[i2],
                    scratch.normals.get(i2).copied(),
                    scratch.uvs.get(i2).copied(),
                    scratch.colors.get(i2).copied(),
                ),
                P::create_vertex(
                    scratch.positions[i3],
                    scratch.normals.get(i3).copied(),
                    scratch.uvs.get(i3).copied(),
                    scratch.colors.get(i3).copied(),
                ),
            ],
            material_index,
        });
    }

    for primitive in instance.mesh.primitives() {
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let material_idx = primitive.material().index().unwrap_or(materials.len() - 1);

        let material_tex_coord = P::get_uv_channel(&material_extras[material_idx]);

        let positions = reader
            .read_positions()
            .ok_or(Error::PrimitiveWithNoPositions)?
            .map(|pos| {
                instance
                    .transform
                    .transform_point3(Vec3::from(pos))
                    .to_array()
            });

        scratch.positions.clear();
        scratch.positions.reserve(positions.len());

        for pos in positions {
            bounds.extend(pos);
            scratch.positions.push(pos);
        }

        scratch.uvs.clear();
        if let Some(uv_iter) = reader.read_tex_coords(material_tex_coord) {
            scratch.uvs.extend(uv_iter.into_f32());
        }

        scratch.colors.clear();
        if let Some(color_iter) = reader.read_colors(0) {
            scratch.colors.extend(color_iter.into_rgba_u8());
        }

        scratch.normals.clear();
        if P::USES_NORMALS
            && let Some(normal_iter) = reader.read_normals()
        {
            scratch.normals.extend(normal_iter);
        }

        scratch.indices.clear();
        if let Some(indices) = reader.read_indices() {
            scratch.indices.extend(indices.into_u32());
        } else {
            scratch.indices.extend(0..scratch.positions.len() as u32);
        }

        match primitive.mode() {
            gltf::mesh::Mode::Triangles => {
                let (triangle_indices, _) = scratch.indices.as_chunks::<3>();

                for &triangle in triangle_indices {
                    push_triangle::<P>(triangle, triangles, scratch, material_idx as u32);
                }
            }
            gltf::mesh::Mode::TriangleStrip => {
                for (i, window) in scratch.indices.windows(3).enumerate() {
                    let Ok([idx0, idx1, idx2]) = <[u32; 3]>::try_from(window) else {
                        unreachable!()
                    };

                    // winding order flips every odd triangle
                    if i.is_multiple_of(2) {
                        push_triangle::<P>(
                            [idx0, idx1, idx2],
                            triangles,
                            scratch,
                            material_idx as u32,
                        );
                    } else {
                        push_triangle::<P>(
                            [idx0, idx2, idx1],
                            triangles,
                            scratch,
                            material_idx as u32,
                        );
                    }
                }
            }
            gltf::mesh::Mode::TriangleFan => {
                if scratch.indices.len() >= 3 {
                    let idx0 = scratch.indices[0];

                    for window in scratch.indices[1..].windows(2) {
                        let Ok([idx1, idx2]) = <[u32; 2]>::try_from(window) else {
                            unreachable!()
                        };

                        push_triangle::<P>(
                            [idx0, idx1, idx2],
                            triangles,
                            scratch,
                            material_idx as u32,
                        );
                    }
                }
            }
            gltf::mesh::Mode::LineLoop | gltf::mesh::Mode::Lines | gltf::mesh::Mode::LineStrip => {
                eprintln!("line primitives are not supported");
            }
            gltf::mesh::Mode::Points => {
                eprintln!("point primitives are not supported");
            }
        }
    }

    Ok(())
}

/// A multithreaded version of `gltf::import`. Outputs `RgbaImage` instead of `gltf::image::Data`.
#[profiling::function]
fn import_gltf(
    reader: impl SceneReader,
) -> Result<(gltf::Document, Vec<gltf::buffer::Data>, Vec<Arc<RgbaImage>>)> {
    use rayon::prelude::*;

    // this cannot depend on `reader`'s lifetime because reader is entirely
    // consumed when loading the gltf; we map to pathbuf and deref to path
    let base = reader.base_path().map(PathBuf::from);
    let base = base.as_deref();

    let mut gltf = {
        profiling::scope!("gltf::load_document");

        gltf::Gltf::from_reader(reader)?
    };

    let buffers = {
        profiling::scope!("gltf::import_buffers");

        gltf::import_buffers(&gltf.document, base, gltf.blob.take())?
    };

    let images = gltf
        .images()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|image| {
            profiling::scope!("gltf::load_image");

            let texture_data = gltf::image::Data::from_source(image.source(), base, &buffers)?;

            convert_image(texture_data)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((gltf.document, buffers, images))
}

#[profiling::function]
fn load_gltf<P: GltfPipeline>(reader: impl SceneReader, root_transform: Mat4) -> Result<Scene<P>> {
    let (document, buffers, images) = import_gltf(reader)?;

    let (mut materials, mut material_extras) = document
        .materials()
        .map(|material| P::parse_material(&material, &images))
        .collect::<Result<(Vec<_>, Vec<_>)>>()?;

    let (fallback_material, fallback_extras) = P::fallback_material();
    materials.push(fallback_material);
    material_extras.push(fallback_extras);

    let mut instances = Vec::new();
    for scene in document.scenes() {
        for node in scene.nodes() {
            collect_instances(&node, root_transform, &mut instances);
        }
    }

    let total_triangles: usize = instances
        .iter()
        .map(|instance| {
            instance
                .mesh
                .primitives()
                .filter(|p| p.mode() == gltf::mesh::Mode::Triangles)
                .map(|p| p.indices().map_or(0, |a| a.count() / 3))
                .sum::<usize>()
        })
        .sum();

    let mut triangles = Vec::with_capacity(total_triangles);
    let mut bounds = BoundingBox::empty();

    let mut scratch = MeshScratch::default();

    for instance in instances {
        if let Err(e) = parse_mesh_instance::<P>(
            instance,
            &mut bounds,
            &materials,
            &material_extras,
            &buffers,
            &mut triangles,
            &mut scratch,
        ) {
            eprintln!("failed to parse mesh: {e}");
        }
    }

    Ok(Scene {
        triangles,
        materials,
        bounds,
    })
}

/// Config for the [`Gltf`] voxelizer.
#[derive(Debug)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "clap", command(next_help_heading = "glTF format options"))]
pub struct GltfConfig {
    /// The provided scale will be applied onto the model during importing
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 1.0))]
    pub base_scale: f32,
}

/// The definition of the input format.
///
/// NOTE: The format requires [`SceneReader::base_path`]
/// to return [`Some`] if using the `.gltf` file extension.
/// `.glb` does not use it.
pub struct Gltf;

impl Format for Gltf {
    // Y: up, -Z: forward, X: right
    const BASIS: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],  // X
        [0.0, 1.0, 0.0, 0.0],  // Y
        [0.0, 0.0, -1.0, 0.0], // -Z
        [0.0, 0.0, 0.0, 1.0],  // W
    ];
}

impl<P: GltfPipeline> InputFormat<P> for Gltf {
    type Config = GltfConfig;
    type Error = Error;

    fn read<R: SceneReader>(
        transform_matrix: [[f32; 4]; 4],
        reader: R,
        config: GltfConfig,
    ) -> Result<Scene<P>> {
        let root_transform = Mat4::from_cols_array_2d(&transform_matrix)
            * Mat4::from_scale(Vec3::splat(config.base_scale));

        load_gltf(reader, root_transform)
    }
}
