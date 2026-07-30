use crate::{GltfPipeline, GltfTexturingExtras, get_texture_data};
use image::RgbaImage;
use std::sync::Arc;
use voxquant_core::pipelines::pbr;

impl GltfPipeline for pbr::Pipeline {
    type MaterialExtras = GltfMaterialExtras;

    const USES_NORMALS: bool = true;

    fn create_vertex(
        pos: [f32; 3],
        normal: Option<[f32; 3]>,
        uv: Option<[f32; 2]>,
        color: Option<[u8; 4]>,
    ) -> pbr::Vertex {
        pbr::Vertex::new(pos, normal, uv, color)
    }

    fn fallback_material() -> (pbr::Material, GltfMaterialExtras) {
        (
            pbr::Material {
                texturing: None,
                metallic_roughness_texture: None,
                alpha_threshold: None,
                base_color: [255, 255, 255, 255],
                metallic_factor: 1.0,
                roughness_factor: 1.0,
                emissive: false,
            },
            GltfMaterialExtras {
                texturing: None,
                orm_texturing: None,
            },
        )
    }

    fn parse_material(
        mat: &gltf::Material,
        image_data: &[Arc<RgbaImage>],
    ) -> crate::Result<(pbr::Material, GltfMaterialExtras)> {
        let alpha_threshold = match mat.alpha_mode() {
            gltf::material::AlphaMode::Opaque => None,
            gltf::material::AlphaMode::Mask => {
                let cutoff = mat.alpha_cutoff().unwrap_or(0.5);
                Some((cutoff * 255.0) as u8)
            }
            gltf::material::AlphaMode::Blend => Some(250),
        };

        let emissive = mat.emissive_factor().into_iter().any(|c| c > 0.0);
        let pbr = mat.pbr_metallic_roughness();

        let base_color = if emissive {
            let [r, g, b] = mat.emissive_factor().map(|r| (r * 255.0) as u8);
            [r, g, b, 255]
        } else {
            pbr.base_color_factor().map(|r| (r * 255.0) as u8)
        };

        let metallic_factor = pbr.metallic_factor();
        let roughness_factor = pbr.roughness_factor();

        let albedo_info = mat
            .emissive_texture()
            .or_else(|| pbr.base_color_texture())
            .or_else(|| {
                mat.pbr_specular_glossiness()
                    .and_then(|s| s.diffuse_texture())
            });

        let (texturing, texturing_extras) = if let Some(info) = albedo_info {
            let (tex, ext) = get_texture_data(&info, image_data)?;
            (Some(tex), Some(ext))
        } else {
            (None, None)
        };

        let (metallic_roughness_texture, orm_extras) =
            if let Some(info) = pbr.metallic_roughness_texture() {
                let (tex, ext) = get_texture_data(&info, image_data)?;
                (Some(tex), Some(ext))
            } else {
                (None, None)
            };

        Ok((
            pbr::Material {
                texturing,
                metallic_roughness_texture,
                alpha_threshold,
                base_color,
                metallic_factor,
                roughness_factor,
                emissive,
            },
            GltfMaterialExtras {
                texturing: texturing_extras,
                orm_texturing: orm_extras,
            },
        ))
    }

    fn get_uv_channel(extras: &Self::MaterialExtras) -> u32 {
        extras
            .texturing
            .as_ref()
            .or(extras.orm_texturing.as_ref())
            .map_or(0, |tex| tex.tex_coord)
    }
}

pub struct GltfMaterialExtras {
    /// If the material has some [`texturing`](Material::texturing),
    /// this will contain the texturing extras
    texturing: Option<GltfTexturingExtras>,
    orm_texturing: Option<GltfTexturingExtras>,
}
