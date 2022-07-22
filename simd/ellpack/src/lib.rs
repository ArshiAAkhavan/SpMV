#![feature(ptr_const_cast)]
#![feature(stdsimd)]

mod ellpack;
mod matrix;

const BYTE_LEN: usize = 8;
const AVX512_VLEN: usize = 512;
const AVX256_VLEN: usize = 256;
const PACK_SIZE: usize = AVX512_VLEN / (BYTE_LEN * std::mem::size_of::<Input>());

pub type Input = f32;

pub use ellpack::Ellpack;
pub use matrix::Matrix;
