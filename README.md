# MeshToVox
A Command line ultility to convert triangle meshes into voxels.

The utility supports loading `.gltf`/`.glb` files and outputting `.gltf` (untested) and `.vox` files

## CLI Usage
Usage: `mesh_to_vox [OPTIONS] --input <INPUT> --output <OUTPUT>`

Options:
  `-i, --input <INPUT>`    The input file that will be voxelized
  `-o, --output <OUTPUT>`  The output file after voxelization
      `--dim <DIM>`        The resolution of the output model [default: 1022]
      `--sparse <SPARSE>`  [default: true] [possible values: true, false]
  `-h, --help`             Print help
  `-V, --version`          Print version

## Installation
[Cargo](https://www.rust-lang.org/tools/install 'Cargo') is requried for installation. Clone the repo and run with `cargo run --release -- (your argument)`