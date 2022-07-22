use ellpack::Ellpack;
use ellpack::Matrix;

fn main() {
    let mut m = Matrix::new(9, 9);
    // m.fill_sparce(0.1);

    m[0][0] = 1_f32;
    m[0][3] = 2_f32;
    m[1][1] = 3_f32;
    m[1][2] = 4_f32;
    m[2][6] = 5_f32;
    m[3][1] = 6_f32;
    m[3][3] = 7_f32;
    m[4][1] = 8_f32;
    m[4][3] = 9_f32;
    m[4][4] = 10_f32;
    m[5][0] = 11_f32;
    m[5][2] = 12_f32;
    m[6][0] = 13_f32;
    m[6][5] = 14_f32;
    m[7][3] = 15_f32;
    m[7][5] = 16_f32;
    m[7][7] = 17_f32;
    println!("{m}");
    let e = Ellpack::from(m);
    // let v:Vec<f32>=(0..8).map(|x| x as f32).collect();
    let v = vec![1f32; 8];
    let out = e.mul_simd(v);
    println!("{e}");
    println!("{out:?}");
}
