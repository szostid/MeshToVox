//! `glTF 2.0` input support for [`voxquant_core`] through the [`gltf`](https://docs.rs/gltf/latest/gltf/) crate
use crate::{Error, GltfPipeline, Result};
use image::RgbaImage;
use std::sync::Arc;
use voxquant_core::pipelines::pbrless;
use voxquant_core::scene::MaterialTexturing;
use voxquant_core::scene::WrapMode;

impl GltfPipeline for pbrless::Pipeline {
    type MaterialExtras = GltfMaterialExtras;

    const USES_NORMALS: bool = false;

    fn create_vertex(
        pos: [f32; 3],
        _normal: Option<[f32; 3]>,
        uv: Option<[f32; 2]>,
        color: Option<[u8; 4]>,
    ) -> pbrless::Vertex {
        pbrless::Vertex::new(pos, uv, color)
    }

    fn fallback_material() -> (pbrless::Material, GltfMaterialExtras) {
        (
            pbrless::Material {
                texturing: None,
                alpha_threshold: None,
                base_color: [255, 255, 255, 255],
                emissive: false,
            },
            GltfMaterialExtras { texturing: None },
        )
    }

    fn parse_material(
        mat: &gltf::Material,
        image_data: &[Arc<RgbaImage>],
    ) -> crate::Result<(pbrless::Material, GltfMaterialExtras)> {
        let alpha_threshold = match mat.alpha_mode() {
            gltf::material::AlphaMode::Opaque => None,
            gltf::material::AlphaMode::Mask => {
                let cutoff = mat.alpha_cutoff().unwrap_or(0.5);
                Some((cutoff * 255.0) as u8)
            }
            // we cannot handle transparency yet, so we do a very high alpha threshold.
            // basically everything that's not opaque is not voxelized at all
            //
            // NOTE: don't use 255 here, i've found that (i guess due to precision issues?)
            // some stuff can become a swiss cheese with too high of a threashold
            gltf::material::AlphaMode::Blend => Some(250),
        };

        let emissive = mat.emissive_factor().into_iter().any(|c| c > 0.0);

        let base_color = if emissive {
            let [r, g, b] = mat.emissive_factor().map(|r| (r * 255.0) as u8);

            [r, g, b, 255]
        } else {
            mat.pbr_metallic_roughness()
                .base_color_factor()
                .map(|r| (r * 255.0) as u8)
        };

        let (texturing, texturing_extras) = match get_material_texture_data(mat, image_data)? {
            Some((texturing, extras)) => (Some(texturing), Some(extras)),
            None => (None, None),
        };

        Ok((
            pbrless::Material {
                texturing,
                alpha_threshold,
                base_color,
                emissive,
            },
            GltfMaterialExtras {
                texturing: texturing_extras,
            },
        ))
    }

    fn get_uv_channel(extras: &Self::MaterialExtras) -> u32 {
        extras.texturing.as_ref().map_or(0, |tex| tex.tex_coord)
    }
}

struct GltfTexturingExtras {
    tex_coord: u32,
}

pub struct GltfMaterialExtras {
    /// If the material has some [`texturing`](Material::texturing),
    /// this will contain the texturing extras
    texturing: Option<GltfTexturingExtras>,
}

fn get_material_texture_data(
    mat: &gltf::Material,
    image_data: &[Arc<RgbaImage>],
) -> Result<Option<(MaterialTexturing, GltfTexturingExtras)>> {
    fn with_material_texture<R>(
        mat: &gltf::Material,
        f: impl FnOnce(gltf::texture::Info<'_>) -> R,
    ) -> Option<R> {
        if let Some(info) = mat.emissive_texture() {
            return Some(f(info));
        }

        if let Some(info) = mat.pbr_metallic_roughness().base_color_texture() {
            return Some(f(info));
        }

        if let Some(info) = mat
            .pbr_specular_glossiness()
            .and_then(|spectral| spectral.diffuse_texture())
        {
            return Some(f(info));
        }

        None
    }

    const fn into_voxelization_mode(value: gltf::texture::WrappingMode) -> WrapMode {
        match value {
            gltf::texture::WrappingMode::ClampToEdge => WrapMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => WrapMode::MirroredRepeat,
            gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
        }
    }

    with_material_texture(mat, |texture_info| {
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
    })
    .map_or(Ok(None), |f| f.map(Some))
}
