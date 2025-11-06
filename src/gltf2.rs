use crate::io::*;
use crate::*;
use image::ImageBuffer;
use image::Rgb;
use image::Rgba;
use image::buffer::ConvertBuffer;
use rayon::prelude::*;

/// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html 5.1.3. accessor.componentType
trait AccessorComponentType {
    const ACCESSOR_COMPONENT_TYPE: i32;
}

impl AccessorComponentType for i8 {
    const ACCESSOR_COMPONENT_TYPE: i32 = 5120;
}
impl AccessorComponentType for u8 {
    const ACCESSOR_COMPONENT_TYPE: i32 = 5121;
}
impl AccessorComponentType for i16 {
    const ACCESSOR_COMPONENT_TYPE: i32 = 5122;
}
impl AccessorComponentType for u16 {
    const ACCESSOR_COMPONENT_TYPE: i32 = 5123;
}
impl AccessorComponentType for i32 {
    const ACCESSOR_COMPONENT_TYPE: i32 = 5125;
}
impl AccessorComponentType for f32 {
    const ACCESSOR_COMPONENT_TYPE: i32 = 5126;
}

#[profiling::function]
fn convert_image(data: &gltf::image::Data) -> Result<image::RgbImage> {
    match data.format {
        gltf::image::Format::R32G32B32FLOAT => {
            let pixels: &[f32] = bytemuck::cast_slice(&data.pixels);

            ImageBuffer::<Rgb<f32>, _>::from_raw(data.width, data.height, pixels.to_vec())
                .context("image has invalid dimensions")
                .map(|img| img.convert())
        }

        gltf::image::Format::R16G16B16 => {
            let pixels: &[u16] = bytemuck::cast_slice(&data.pixels);

            ImageBuffer::<Rgb<u16>, _>::from_raw(data.width, data.height, pixels.to_vec())
                .context("image has invalid dimensions")
                .map(|img| img.convert())
        }

        gltf::image::Format::R8G8B8 => {
            let pixels = data.pixels.clone();

            ImageBuffer::<Rgb<u8>, _>::from_raw(data.width, data.height, pixels)
                .context("image has invalid dimensions")
        }

        gltf::image::Format::R32G32B32A32FLOAT => {
            let pixels: &[f32] = bytemuck::cast_slice(&data.pixels);

            ImageBuffer::<Rgba<f32>, _>::from_raw(data.width, data.height, pixels.to_vec())
                .context("image has invalid dimensions")
                .map(|img| img.convert())
        }

        gltf::image::Format::R16G16B16A16 => {
            let pixels: &[u16] = bytemuck::cast_slice(&data.pixels);

            ImageBuffer::<Rgba<u16>, _>::from_raw(data.width, data.height, pixels.to_vec())
                .context("image has invalid dimensions")
                .map(|img| img.convert())
        }

        gltf::image::Format::R8G8B8A8 => {
            let pixels = data.pixels.clone();

            ImageBuffer::<Rgba<u8>, _>::from_raw(data.width, data.height, pixels)
                .context("image has invalid dimensions")
                .map(|img| img.convert())
        }

        _ => bail!("format {:?} is unsupported", data.format),
    }
}

#[profiling::function]
fn parse_image(
    image_data: &[gltf::image::Data],
    texture: gltf::Texture,
    source_dir: &str,
) -> Result<image::RgbImage> {
    let source = texture.source().source();

    let data = match source {
        gltf::image::Source::Uri { uri, .. } => {
            let path = format!("{source_dir}/{uri}");

            let image = image::open(path.as_str())
                .with_context(|| format!("failed to fetch file `{path}` used by the mesh"))?;

            image.into_rgb8()
        }

        gltf::image::Source::View { .. } => {
            let image = image_data
                .get(texture.index())
                .context("failed to fetch image data (index is out of bounds)")?;

            convert_image(image).context("failed to convert image")?
        }
    };

    Ok(data)
}

#[profiling::function]
fn parse_material(
    mat: &gltf::Material,
    image_data: &[gltf::image::Data],
    source_dir: &str,
) -> Result<ImageOrColor> {
    if let Some(image) = mat
        .pbr_metallic_roughness()
        .base_color_texture()
        .map(|texture_info| texture_info.texture())
    {
        return parse_image(&image_data, image, source_dir)
            .context("failed to parse the color image used by the material")
            .map(ImageOrColor::Image);
    }

    if let Some(image) = mat
        .emissive_texture()
        .map(|texture_info| texture_info.texture())
    {
        return parse_image(&image_data, image, source_dir)
            .context("failed to parse the emissive image used by the material")
            .map(ImageOrColor::Image);
    }

    if let Some(image) = mat
        .pbr_specular_glossiness()
        .and_then(|spectral| spectral.diffuse_texture())
        .map(|texture_info| texture_info.texture())
    {
        return parse_image(&image_data, image, source_dir)
            .context("failed to parse the color image of the spectral material")
            .map(ImageOrColor::Image);
    }

    let base_color = mat.pbr_metallic_roughness().base_color_factor();

    let base_color = [
        (base_color[0] * 255.0) as u8,
        (base_color[1] * 255.0) as u8,
        (base_color[2] * 255.0) as u8,
    ];

    Ok(ImageOrColor::Color(base_color))
}

#[profiling::function]
fn parse_mesh(
    mesh: &gltf::Mesh,
    bounds: &mut BoundingBox,
    materials: &[ImageOrColor],
    buffers: &[gltf::buffer::Data],
    triangles: &mut Vec<[Vec3; 3]>,
    extras: &mut Vec<[VertexExtras; 3]>,
) -> Result<()> {
    #[inline]
    fn get_extras(
        idx: usize,
        normals: Option<&[Vec3]>,
        uvs: Option<&[Vec2]>,
        material_idx: u32,
    ) -> VertexExtras {
        let normal = normals
            .as_ref()
            .and_then(|normals| normals.get(idx))
            .copied();

        let uv = uvs.as_ref().and_then(|uvs| uvs.get(idx)).copied();

        VertexExtras::new(normal, uv, material_idx)
    }

    for primitive in mesh.primitives() {
        let mode = primitive.mode();

        if mode != gltf::mesh::Mode::Triangles {
            bail!("a mesh in the file uses non-triangle geometry");
        }

        let bound = primitive.bounding_box();
        bounds.extend(bound.min.into());
        bounds.extend(bound.max.into());

        let material_idx = primitive.material().index().unwrap_or(materials.len());

        let data = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        let mut indices = data
            .read_indices()
            .context("a mesh in the file has no vertex indices")?
            .into_u32();

        let vert_coords = data
            .read_positions()
            .context("a mesh in the file has no vertex positions")?
            .map(Vec3::from)
            .collect::<Vec<_>>();

        let normals = data
            .read_normals()
            .map(|normals| normals.map(Vec3::from).collect::<Vec<_>>());

        let uvs = data
            .read_tex_coords(0)
            .map(|uvs| uvs.into_f32().map(Vec2::from).collect::<Vec<_>>());

        loop {
            let i1 = indices.next();
            let i2 = indices.next();
            let i3 = indices.next();

            if i1.is_none() {
                break;
            }

            let (Some(i1), Some(i2), Some(i3)) = (i1, i2, i3) else {
                eprintln!("found a non-full triangle ({i1:?}, {i2:?}, {i3:?})");
                break;
            };

            triangles.push([
                vert_coords[i1 as usize],
                vert_coords[i2 as usize],
                vert_coords[i3 as usize],
            ]);

            extras.push([
                get_extras(
                    i1 as usize,
                    normals.as_deref(),
                    uvs.as_deref(),
                    material_idx as u32,
                ),
                get_extras(
                    i2 as usize,
                    normals.as_deref(),
                    uvs.as_deref(),
                    material_idx as u32,
                ),
                get_extras(
                    i3 as usize,
                    normals.as_deref(),
                    uvs.as_deref(),
                    material_idx as u32,
                ),
            ]);
        }
    }

    Ok(())
}

#[profiling::function]
pub fn load_gltf(path: &str) -> Result<Mesh> {
    let (document, buffers, images) = {
        profiling::scope!("gltf::import");
        gltf::import(path).context("failed to load the gltf file")
    }?;

    let folder = std::path::Path::new(path)
        .parent()
        .and_then(|file| file.as_os_str().to_str())
        .context("failed to read the parent folder of the file")?;

    let main_camera = document
        .cameras()
        .find(|cam| cam.index() != 0)
        .map(|camera| Camera::new(&camera.projection()));

    let model_view_projection = Mat4::from_cols_array_2d(
        &document
            .scenes()
            .next()
            .context("file has no scenes")?
            .nodes()
            .next()
            .context("scene has no root nodes")?
            .transform()
            .matrix(),
    );

    let view = View {
        camera: main_camera,
        model_view_projection,
    };

    let mut triangles = Vec::new();
    let mut triangle_extras = Vec::new();

    let mut materials = document
        .materials()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|material| parse_material(&material, &images, folder))
        .collect::<Result<Vec<_>, _>>()
        .context("failed to parse materials")?;

    // i.e. default material
    materials.push(ImageOrColor::Color([255, 255, 255]));

    let mut bounds = BoundingBox::max();

    for mesh in document.meshes() {
        parse_mesh(
            &mesh,
            &mut bounds,
            &materials,
            &buffers,
            &mut triangles,
            &mut triangle_extras,
        )?;
    }

    Ok(Mesh {
        materials,
        triangles,
        triangle_extras,
        bounds,
        view,
    })
}

#[profiling::function]
pub fn save_gltf(vertices: &[Vertex], gltf_path: &str, view: View, float: bool) -> Result<()> {
    let bb = BoundingBox::from_points(vertices.iter().map(|v| v.position));

    let size_of_vertices = if float {
        size_of::<FloatVertex>()
    } else {
        size_of::<Vertex>()
    };

    let num_bytes = vertices.len() * size_of_vertices;

    let buffer = json::object! {
        uri : "model.bin",
        byteLength : num_bytes,
    };

    let vertex_view = json::object! {
        buffer : 0,
        byteOffset : 0,
        byteLength : num_bytes,
        byteStride : size_of_vertices,
    };

    let byte_offset = if float {
        core::mem::offset_of!(FloatVertex, position)
    } else {
        core::mem::offset_of!(Vertex, position)
    };

    let position_accessor = json::object! {
        bufferView : 0,
        byteOffset : byte_offset,
        componentType : f32::ACCESSOR_COMPONENT_TYPE,
        count : vertices.len(),
        type : "VEC3",

        max : [bb.max.x, bb.max.y, bb.max.z],
        min : [bb.min.x, bb.min.y, bb.min.z],
    };

    let byte_offset = if float {
        core::mem::offset_of!(FloatVertex, color)
    } else {
        core::mem::offset_of!(Vertex, color)
    };
    let component_type = if float {
        f32::ACCESSOR_COMPONENT_TYPE
    } else {
        u8::ACCESSOR_COMPONENT_TYPE
    };
    let normalized = !float;

    let color_accessor = json::object! {
        bufferView : 0,
        byteOffset : byte_offset,
        componentType : component_type,
        normalized : normalized,
        count : vertices.len(),
        type : "VEC3",
    };

    let material = json::object! {
        doubleSided : true,
    };

    let mesh = json::object! {
        primitives : [{
            attributes : {
                POSITION : 0,
                COLOR_0 : 1,
            },

            material : 0
        }],
    };

    let gltf = json::object! {
        materials : [material],
        scenes : [ {nodes : [ 0 ]} ],
        nodes : [ {
            mesh : 0,
            matrix : mpv_to_json(&view.model_view_projection),
        }],

        meshes : [mesh],
        buffers : [buffer],
        bufferViews : [vertex_view],
        accessors : [position_accessor, color_accessor],
        asset : {version : "2.0" }
    };

    let folder = std::path::Path::new(gltf_path).parent().unwrap();
    let folder = folder.as_os_str().to_str().unwrap();

    std::fs::create_dir(folder)?;
    let bin_path = format!("{}/model.bin", folder);

    std::fs::write(gltf_path, gltf.dump())?;
    if float {
        let vertices = vertices
            .iter()
            .map(|vert| FloatVertex::from(vert.clone()))
            .collect::<Vec<_>>();
        std::fs::write(bin_path, bytemuck::cast_slice(&vertices))?;
    } else {
        std::fs::write(bin_path, bytemuck::cast_slice(vertices))?;
    }

    Ok(())
}
