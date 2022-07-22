use rand;
use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

use super::Input;

#[derive(Debug)]
pub struct Matrix<T> {
    nrows: u32,
    ncols: u32,
    data: Vec<Vec<T>>,
}

impl Matrix<Input>
// impl<T> Matrix<T>
// where
//     T: Clone + Default,
{
    pub fn new(nrows: u32, ncols: u32) -> Self {
        let mut data = Vec::with_capacity(nrows as usize);
        for _ in 0..nrows {
            data.push(vec![Default::default(); ncols as usize]);
        }
        Self { nrows, ncols, data }
    }

    pub fn fill_sparce(&mut self, sparsity_factor: f32) {

        for row in &mut self.data {
            let nnz = ((self.ncols as f32) * sparsity_factor).round() as usize + 1;
            for _ in 0..nnz {
                // row[rand::random::<usize>() % self.nrows] = rand::random::<Input>();
                row[(rand::random::<u32>() % self.nrows) as usize] = (rand::random::<u32>() % 5) as f32;
            }
        }
    }

    pub fn row_count(&self) -> u32 {
        return self.nrows;
    }
    pub fn col_count(&self) -> u32 {
        return self.ncols;
    }
}

impl<T: Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "matrix[{}][{}]\n", self.nrows, self.ncols)?;
        for row in self {
            for cell in row {
                write!(f, "{cell} ")?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl<'a, T> IntoIterator for &'a Matrix<T> {
    type Item = <&'a Vec<Vec<T>> as IntoIterator>::Item;

    type IntoIter = <&'a Vec<Vec<T>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&self.data).into_iter()
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = Vec<T>;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.data[index];
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.data[index];
    }
}
