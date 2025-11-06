use std::collections::HashMap;

use crate::octree::*;
use crate::*;
use bytemuck::Pod;
use bytemuck::Zeroable;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct Vertex {
    pub position: Vec3,
    pub color: [u8; 3],
    pub _p: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct FloatVertex {
    pub position: Vec3,
    pub color: [f32; 3],
}

impl From<Vertex> for FloatVertex {
    fn from(value: Vertex) -> Self {
        let color = value.color.map(|c| c as f32 / 255.0);
        Self {
            position: value.position,
            color,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct VertexExtras {
    normal: Vec3,
    uv: Vec2,

    pub material_idx: u32,
}

impl VertexExtras {
    pub fn new(normal: Option<Vec3>, uv: Option<Vec2>, material_idx: u32) -> Self {
        Self {
            normal: normal.unwrap_or(Vec3::NAN),
            uv: uv.unwrap_or(Vec2::NAN),
            material_idx,
        }
    }

    #[inline]
    #[must_use]
    pub fn normal(&self) -> Option<Vec3> {
        (self.normal != Vec3::NAN).then_some(self.normal)
    }

    #[inline]
    #[must_use]
    pub fn uv(&self) -> Option<Vec2> {
        (self.uv != Vec2::NAN).then_some(self.uv)
    }
}

#[derive(Debug, Clone)]
pub enum ImageOrColor {
    Image(image::RgbImage),
    Color([u8; 3]),
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub triangles: Vec<[Vec3; 3]>,
    pub triangle_extras: Vec<[VertexExtras; 3]>,
    pub materials: Vec<ImageOrColor>,

    pub bounds: BoundingBox,
    pub view: View,
}

#[derive(Debug, Clone)]
pub struct PerspectiveCamera {
    pub yfov: f32,
    pub znear: f32,

    pub zfar: Option<f32>,
    pub aspect_ratio: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct OrthographicCamera {
    pub xmag: f32,
    pub ymag: f32,
    pub zfar: f32,
    pub znear: f32,
}

impl PerspectiveCamera {
    pub fn new(value: &gltf::camera::Perspective<'_>) -> Self {
        Self {
            yfov: value.yfov(),
            znear: value.znear(),
            zfar: value.zfar(),
            aspect_ratio: value.aspect_ratio(),
        }
    }

    pub fn to_json(&self) -> json::JsonValue {
        json::object! {
            "type" : "perspective",
            perspective : {
                yfov : self.yfov,
                znear : self.znear,
                zfar : self.zfar,
                aspect_ratio : self.aspect_ratio,
            }
        }
    }
}
impl OrthographicCamera {
    pub fn new(value: &gltf::camera::Orthographic<'_>) -> Self {
        Self {
            xmag: value.xmag(),
            ymag: value.ymag(),
            zfar: value.zfar(),
            znear: value.znear(),
        }
    }

    pub fn to_json(&self) -> json::JsonValue {
        json::object! {
            "type": "orthographic",
            orthographic : {
                xmag : self.xmag,
                ymag : self.ymag,
                zfar : self.zfar,
                znear : self.znear,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Camera {
    PerspectiveCamera(PerspectiveCamera),
    OrthographiCamera(OrthographicCamera),
}

impl Camera {
    pub fn new(cam: &gltf::camera::Projection<'_>) -> Self {
        match cam {
            gltf::camera::Projection::Orthographic(ort) => {
                Camera::OrthographiCamera(OrthographicCamera::new(ort))
            }
            gltf::camera::Projection::Perspective(per) => {
                Camera::PerspectiveCamera(PerspectiveCamera::new(per))
            }
        }
    }
    pub fn to_json(&self) -> json::JsonValue {
        match self {
            Self::PerspectiveCamera(ort) => ort.to_json(),
            Self::OrthographiCamera(per) => per.to_json(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct View {
    pub camera: Option<Camera>,
    pub model_view_projection: Mat4,
}

pub fn mpv_to_json(mvp: &Mat4) -> json::JsonValue {
    let mut output: Vec<json::JsonValue> = Vec::with_capacity(16);

    let mvp = mvp.to_cols_array();

    for i in 0..16 {
        output.push(mvp[i].into());
    }

    json::JsonValue::Array(output)
}

mod magica {
    pub const fn encode(color: image::Rgb<u8>) -> u8 {
        let color = color.0;
        (color[0] >> 5) | ((color[1] >> 5) << 3) | ((color[2] >> 6) << 6)
    }

    pub const fn decode(byte: u8) -> image::Rgb<u8> {
        let mask3 = (1 << 3) - 1;
        let mask2 = (1 << 2) - 1;

        let r = (byte & mask3) << 5;
        let g = ((byte >> 3) & mask3) << 5;
        let b = ((byte >> 6) & mask2) << 6;

        image::Rgb([r, g, b])
    }

    #[cfg(test)]
    pub const fn _gather() {
        let mut counter = 0;
        loop {
            if encode(decode(counter)) != counter {
                panic!()
            }
            if counter == u8::MAX {
                break;
            }
            counter += 1;
        }
    }

    #[cfg(test)]
    pub const _: () = _gather();
}

impl Octree {
    pub fn save_as_magica_voxel(&self, file_path: &str, size: u32) -> Result<()> {
        use dot_vox::*;

        const CHUNK_SIZE: i32 = 256;

        let nodes = self.collect_nodes();

        let mut chunks = HashMap::<IVec3, Vec<dot_vox::Voxel>>::new();

        for (coords, color) in nodes {
            let color = octree_header::to_color(color);
            let color_idx = magica::encode(color);

            let chunk = coords.coords / CHUNK_SIZE;
            let local_coords = (coords.coords % CHUNK_SIZE).as_u8vec3();

            chunks.entry(chunk).or_default().push(dot_vox::Voxel {
                x: local_coords.x,
                y: local_coords.z,
                z: local_coords.y,
                i: color_idx,
            });
        }

        let mut palette = Vec::with_capacity(256);

        for index in 0..u8::MAX {
            let color = magica::decode(index);
            palette.push(dot_vox::Color {
                r: color.0[0],
                g: color.0[1],
                b: color.0[2],
                a: 255,
            });
        }

        let mut models = Vec::new();
        let mut nodes = Vec::new();

        nodes.push(SceneNode::Transform {
            attributes: Default::default(),
            frames: vec![Frame {
                attributes: Default::default(),
            }],
            child: 1,
            layer_id: 0,
        });

        nodes.push(SceneNode::Group {
            attributes: Default::default(),
            children: Vec::new(),
        });

        for (chunk, voxels) in chunks {
            let model_id = models.len() as u32;

            models.push(Model {
                size: Size {
                    x: CHUNK_SIZE as u32,
                    y: CHUNK_SIZE as u32,
                    z: CHUNK_SIZE as u32,
                },
                voxels,
            });

            let transform_index = nodes.len() as u32;
            let shape_index = transform_index + 1;

            nodes.push(SceneNode::Transform {
                attributes: Default::default(),
                frames: vec![Frame {
                    attributes: [(
                        "_t".to_string(),
                        format!(
                            "{} {} {}",
                            chunk.x * CHUNK_SIZE,
                            chunk.z * CHUNK_SIZE,
                            chunk.y * CHUNK_SIZE
                        ),
                    )]
                    .into(),
                }],
                child: shape_index,
                layer_id: 0,
            });

            nodes.push(SceneNode::Shape {
                attributes: Default::default(),
                models: vec![ShapeModel {
                    model_id,
                    attributes: Default::default(),
                }],
            });

            let SceneNode::Group { children, .. } = &mut nodes[1] else {
                unreachable!()
            };

            children.push(transform_index);
        }

        // Construct the scene
        let data = dot_vox::DotVoxData {
            version: 150,
            models,
            palette,
            materials: Vec::new(),
            layers: Vec::new(),
            scenes: nodes,
        };

        // Write the file
        let mut file = std::fs::File::create(file_path)?;

        data.write_vox(&mut file)?;

        Ok(())
    }

    pub fn save_as_gltf(
        &self,
        gltf_path: &str,
        view: View,
        sparse: bool,
        size: u32,
        float: bool,
    ) -> Result<()> {
        let max_size = size - 1;

        let mesh = if sparse {
            self.fill_space(max_size)
        } else {
            let nodes = self.collect_nodes();
            let mut tris: Vec<Vertex> = Vec::with_capacity(nodes.len() * 36);
            for (node, color) in &nodes {
                let color = octree_header::to_color(*color).0;
                for i in 0..6 {
                    let node = crate::space_filling::MeshNode {
                        cords: node.coords,
                        dim: i / 2,
                        positive: (i % 2) == 0,
                        depth: node.depth as u8,
                    };
                    let verts = node.to_vertices(self.depth as u8);

                    let verts = verts.map(|vert| {
                        let position =
                            ((vert + IVec3::NEG_ONE).as_dvec3() / max_size as f64).as_vec3();
                        let position = position.mul_add(Vec3::splat(2.0), Vec3::NEG_ONE);

                        Vertex {
                            position,
                            color,
                            _p: 0,
                        }
                    });

                    for vert in verts {
                        tris.push(vert);
                    }
                }
            }

            tris
        };

        gltf2::save_gltf(&mesh, gltf_path, view, float)
    }
}
