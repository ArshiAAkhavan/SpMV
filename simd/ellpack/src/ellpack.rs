use std::fmt::Display;

use super::Input;
use super::Matrix;
use super::PACK_SIZE;

#[derive(Default, Debug)]
pub struct Ellpack<T> {
    cap: usize,
    vals: Vec<T>,
    cols: Vec<i32>,
    pack_ptr: Vec<usize>,
    rows: Vec<(usize, usize)>,
}

impl Ellpack<f32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn mul_simd(&self, v: Vec<f32>) -> Vec<f32> {
        #[cfg(any(target_arch = "x86_64"))]
        use core::arch::x86_64 as vp;

        #[cfg(any(target_arch = "x86"))]
        use core::arch::x86 as vp;

        let n_all_rows = (self.pack_ptr.len() - 1) * PACK_SIZE;
        let mut out: Vec<f32> = vec![Default::default(); n_all_rows];
        let mut rows: Vec<i32> = self.rows.iter().map(|x| x.0 as i32).collect();
        rows.extend((rows.len()..n_all_rows).map(|x| x as i32));

        for (pck_idx, _) in self.pack_ptr[..self.pack_ptr.len() - 1].iter().enumerate() {
            let row_ofs = (pck_idx * PACK_SIZE) as isize;
            let odx512;
            let mut out512;
            unsafe {
                odx512 = vp::_mm512_loadu_epi32(rows.as_ptr().offset(row_ofs));
                out512 =
                    vp::_mm512_i32gather_ps::<4>(odx512, out.as_ptr().offset(row_ofs).cast::<u8>());
            }
            for pck_ofs in (self.pack_ptr[pck_idx]..self.pack_ptr[pck_idx + 1]).step_by(PACK_SIZE) {
                let pck_ofs: isize = pck_ofs.try_into().unwrap();

                unsafe {
                    let idx512 = vp::_mm512_loadu_epi32(self.cols.as_ptr().offset(pck_ofs));
                    let val512 = vp::_mm512_load_ps(self.vals.as_ptr().offset(pck_ofs));
                    let vec512 = vp::_mm512_i32gather_ps::<4>(idx512, v.as_ptr().cast::<u8>());

                    out512 = vp::_mm512_fmadd_ps(vec512, val512, out512);
                }
            }
            unsafe {
                vp::_mm512_i32scatter_ps::<4>(
                    out.as_mut_ptr().offset(row_ofs).cast::<u8>(),
                    odx512,
                    out512,
                );
            }
        }

        out[..self.rows.len()].to_vec()
    }
}

impl From<Matrix<Input>> for Ellpack<Input> {
    fn from(matrix: Matrix<Input>) -> Self {
        // calucate nnz for each row so that we can sort rows
        // based on their nnz value
        let mut rows = Vec::with_capacity(matrix.row_count() as usize);
        for (index, row) in matrix.into_iter().enumerate() {
            let nnz = row.iter().filter(|x| **x != Default::default()).count();
            rows.push((index, nnz));
        }
        rows.sort_by_key(|x| x.1);

        // calculate total number of values that ellpack holds
        let cap: usize = rows
            .iter()
            .skip((PACK_SIZE - 1).into())
            .step_by(PACK_SIZE.into())
            .map(|x| x.1)
            .sum();
        let cap = cap
            + rows
                .iter()
                .last()
                .map(|x| x.1)
                .unwrap_or(Default::default());
        let cap = cap * PACK_SIZE;

        // calculate pack_ptr
        let mut pack_ptr =
            Vec::with_capacity((matrix.row_count() as f32 / PACK_SIZE as f32).ceil() as usize);
        let mut ptr = 0;
        pack_ptr.push(ptr);
        for nnz in rows
            .iter()
            .skip((PACK_SIZE - 1).into())
            .step_by(PACK_SIZE.into())
            .map(|x| x.1)
        {
            ptr += nnz * PACK_SIZE;
            pack_ptr.push(ptr);
        }
        pack_ptr.push(
            ptr + rows
                .iter()
                .last()
                .map(|x| x.1)
                .unwrap_or(Default::default())
                * PACK_SIZE,
        );

        let mut vals = vec![Default::default(); cap];
        let mut cols = vec![0i32; cap];
        let mut curr_pack_idx = 0;

        // for each pack do the following:
        //    for each row in the pack:
        //        for each nz value in this row, place the val and its coll_idx
        //        in curr_pack_ptr* PACK_SIZE * nnz + row_ext where:
        //            curr_pack_ptr: start index of the current pack in vals
        //            nnz: nnz of the values currenctly seen in this row
        //            row_ext: offset of this value in its pack's row_slice
        // note:
        //    for every iteration of the `while` loop, curr_pack_ptr must be
        //    incremented by `PACK_SIZE` * max nnz seen in rows of curresponding
        //    pack, but since rows are sorted increasing, the last row in the pack
        //    has no less nnz (if not more) that other rows, hence, we can easy
        //    increment `curr_pack_ptr` by `PACK_SIZE` * nnz of last row in pack.
        //
        while curr_pack_idx * PACK_SIZE < rows.len() {
            for row_ofs in 0..PACK_SIZE {
                if curr_pack_idx * PACK_SIZE + row_ofs >= rows.len() {
                    break;
                }

                let row_idx: usize = rows[curr_pack_idx * PACK_SIZE + row_ofs].0;
                let mut nnz = 0;
                for col_idx in 0..matrix.col_count() {
                    if matrix[row_idx][col_idx as usize] != Default::default() {
                        let curr_pack_ptr = pack_ptr[curr_pack_idx];
                        let idx = curr_pack_ptr + PACK_SIZE * nnz + row_ofs;

                        vals[idx] = matrix[row_idx][col_idx as usize];
                        cols[idx] = col_idx as i32;
                        nnz += 1;
                    }
                }
            }
            curr_pack_idx += 1;
        }
        Self {
            cap,
            cols,
            vals,
            rows,
            pack_ptr,
        }
    }
}

impl<T: std::fmt::Debug> Display for Ellpack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ellpack:{}", self.cap)?;
        writeln!(f, "\tvals:{:?}", self.vals)?;
        writeln!(f, "\tcols:{:?}", self.cols)?;
        Ok(())
    }
}
