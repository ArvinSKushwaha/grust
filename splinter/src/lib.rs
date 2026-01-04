use std::{
    array::TryFromSliceError,
    ops::{Add, Mul},
};

pub mod cubic;
pub mod linear;
pub mod quadratic;

pub use crate::{
    cubic::{
        cubic_bezier_f32, cubic_bezier_f64, cubic_degree_decrease_f32, cubic_degree_decrease_f64,
        cubic_split_f32, cubic_split_f64,
    },
    linear::{lerp_f32, lerp_f64},
    quadratic::{
        QuadraticBezierF32, quadratic_bezier_f32, quadratic_bezier_f64,
        quadratic_degree_decrease_f32, quadratic_degree_decrease_f64,
        quadratic_degree_increase_f32, quadratic_degree_increase_f64, quadratic_split_f32,
        quadratic_split_f64,
    },
};

pub trait Lerpable<T, M>: Copy + Add<T, Output = T> + Mul<M, Output = T> {}

impl<M, T: Copy + Add<T, Output = T> + Mul<M, Output = T>> Lerpable<T, M> for T {}

pub fn bezier_eval_f32<T: Lerpable<T, f32>>(t: f32, nodes: &[T]) -> T {
    let n = nodes.len() - 1;
    if t <= 0. {
        return nodes[0];
    } else if t >= 1. {
        return nodes[n];
    }

    match n {
        1 => nodes[0],
        2 => quadratic_bezier_f32(t, nodes.try_into().unwrap()),
        3 => cubic_bezier_f32(t, nodes.try_into().unwrap()),
        n => {
            let mut space = nodes.to_vec();

            for k in 0..n {
                for i in 0..(n - k) {
                    space[i] = lerp_f32(t, [space[i], space[i + 1]].into());
                }
            }
            space[0]
        }
    }
}

pub fn bezier_eval_f64<T: Lerpable<T, f64>>(t: f64, nodes: &[T]) -> T {
    let n = nodes.len() - 1;
    if t <= 0. {
        return nodes[0];
    } else if t >= 1. {
        return nodes[n];
    }

    match n {
        1 => nodes[0],
        2 => quadratic_bezier_f64(t, nodes.try_into().unwrap()),
        3 => cubic_bezier_f64(t, nodes.try_into().unwrap()),
        n => {
            let mut space = nodes.to_vec();

            for k in 0..n {
                for i in 0..(n - k) {
                    space[i] = lerp_f64(t, BezierKnownF64([space[i], space[i + 1]]));
                }
            }
            space[0]
        }
    }
}

pub fn bezier_split_f32<T: Lerpable<T, f32>>(t: f32, nodes: &[T]) -> [Vec<T>; 2] {
    let n = nodes.len() - 1;
    if t <= 0. || t >= 1. {
        panic!("Poorly defined");
    }

    match n {
        1 => [vec![nodes[0]], vec![nodes[0]]],
        2 => quadratic_split_f32(t, nodes.try_into().unwrap()).map(|i| i.0.into()),
        3 => cubic_split_f32(t, nodes.try_into().unwrap()).map(|i| i.0.into()),
        n => {
            let mut space = nodes.to_vec();
            let mut left = Vec::with_capacity(n + 1);
            let mut right = Vec::with_capacity(n + 1);

            for k in 0..n {
                left.push(space[0]);
                right.push(space[n - k]);

                for i in 0..(n - k) {
                    space[i] = lerp_f32(t, BezierKnownF32([space[i], space[i + 1]]));
                }
            }

            left.push(space[0]);

            right.push(space[0]);
            right.reverse();

            [left, right]
        }
    }
}

pub fn bezier_split_f64<T: Lerpable<T, f64>>(t: f64, nodes: &[T]) -> [BezierF64<T>; 2] {
    let n = nodes.len() - 1;
    if t <= 0. || t >= 1. {
        panic!("Poorly defined");
    }

    match n {
        1 => [BezierF64(vec![nodes[0]]), BezierF64(vec![nodes[0]])],
        2 => quadratic_split_f64(t, nodes.try_into().unwrap()).map(|i| BezierF64(i.0.into())),
        3 => cubic_split_f64(t, nodes.try_into().unwrap()).map(|i| i.0.into()),
        n => {
            let mut space = nodes.to_vec();
            let mut left = Vec::with_capacity(n + 1);
            let mut right = Vec::with_capacity(n + 1);

            for k in 0..n {
                left.push(space[0]);
                right.push(space[n - k]);

                for i in 0..(n - k) {
                    space[i] = lerp_f64(t, BezierKnownF64([space[i], space[i + 1]]));
                }
            }

            left.push(space[0]);

            right.push(space[0]);
            right.reverse();

            [BezierF64(left), BezierF64(right)]
        }
    }
}

pub fn bezier_degree_increase_f32<T: Lerpable<T, f32>>(nodes: &[T]) -> BezierF32<T> {
    let n = nodes.len() - 1;

    match n {
        1 => BezierKnownF32([nodes[0], nodes[0] * 0.5 + nodes[1] * 0.5, nodes[1]]).into(),
        2 => quadratic_degree_increase_f32(nodes.try_into().unwrap()).into(),
        n => {
            let mut new_nodes = vec![nodes[0]; nodes.len() + 1];
            new_nodes[0] = nodes[0];
            new_nodes[nodes.len()] = nodes[n];

            let k = n + 1;
            for i in 1..k {
                new_nodes[i] =
                    (nodes[i] * (k - i) as f32 + nodes[i - 1] * i as f32) * (k as f32).recip();
            }

            BezierF32(new_nodes)
        }
    }
}

pub fn bezier_degree_increase_f64<T: Lerpable<T, f64>>(nodes: &[T]) -> BezierF64<T> {
    let n = nodes.len() - 1;

    match n {
        1 => BezierKnownF64([nodes[0], nodes[0] * 0.5 + nodes[1] * 0.5, nodes[1]]).into(),
        2 => quadratic_degree_increase_f64(nodes.try_into().unwrap()).into(),
        n => {
            let mut new_nodes = vec![nodes[0]; nodes.len() + 1];
            new_nodes[0] = nodes[0];
            new_nodes[nodes.len()] = nodes[n];

            let k = n + 1;
            for i in 1..k {
                new_nodes[i] =
                    (nodes[i] * (k - i) as f64 + nodes[i - 1] * i as f64) * (k as f64).recip();
            }

            BezierF64(new_nodes)
        }
    }
}

pub fn bezier_degree_decrease_f32<T: Lerpable<T, f32>>(nodes: &[T]) -> BezierF32<T> {
    let n = nodes.len() - 1;

    match n {
        2 => quadratic_degree_decrease_f32(nodes.try_into().unwrap()).into(),
        3 => cubic_degree_decrease_f32(nodes.try_into().unwrap()).into(),
        n => {
            let mut mat = nalgebra::DMatrix::<f32>::zeros(n + 1, n);

            for i in 0..n {
                mat[(i, i)] = (n - i) as f32 / (n as f32);
                mat[(i + 1, i)] = (i + 1) as f32 / (n as f32);
            }

            let transformation = mat.pseudo_inverse(1e-5).unwrap();

            let mut output = vec![nodes[0]; n];

            for k in 0..n {
                for i in 0..n + 1 {
                    output[k] = output[k] + nodes[i] * transformation[(k, i)];
                }
            }

            let offset = nodes[0] + output[0] * -1.;
            output.iter_mut().for_each(|o| *o = *o + offset);

            BezierF32(output)
        }
    }
}

pub fn bezier_degree_decrease_f64<T: Lerpable<T, f64>>(nodes: &[T]) -> BezierF64<T> {
    let n = nodes.len() - 1;

    match n {
        2 => quadratic_degree_decrease_f64(nodes.try_into().unwrap()).into(),
        3 => cubic_degree_decrease_f64(nodes.try_into().unwrap()).into(),
        n => {
            let mut mat = nalgebra::DMatrix::<f64>::zeros(n + 1, n);

            for i in 0..n {
                mat[(i, i)] = (n - i) as f64 / (n as f64);
                mat[(i + 1, i)] = (i + 1) as f64 / (n as f64);
            }

            let transformation = mat.pseudo_inverse(1e-5).unwrap();

            let mut output = vec![nodes[0]; n];

            for k in 0..n {
                for i in 0..n + 1 {
                    output[k] = output[k] + nodes[i] * transformation[(k, i)];
                }
            }

            let offset = nodes[0] + output[0] * -1.;
            output.iter_mut().for_each(|o| *o = *o + offset);

            BezierF64(output)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BezierKnownF32<T: Lerpable<T, f32>, const N: usize>([T; N]);

#[derive(Debug, Clone, Copy)]
pub struct BezierKnownF64<T: Lerpable<T, f64>, const N: usize>([T; N]);

#[derive(Debug, Clone)]
pub struct BezierF32<T: Lerpable<T, f32>>(Vec<T>);

#[derive(Debug, Clone)]
pub struct BezierF64<T: Lerpable<T, f64>>(Vec<T>);

impl<T: Lerpable<T, f32>, const N: usize> BezierKnownF32<T, N> {
    pub fn eval(&self, t: f32) -> T {
        bezier_eval_f32(t, &self.0[..])
    }

    pub fn split(&self, t: f32) -> [BezierKnownF32<T, N>; 2] {
        let [left, right] = bezier_split_f32(t, &self.0[..]);

        [
            BezierKnownF32::try_from(left).unwrap(),
            BezierKnownF32::try_from(right).unwrap(),
        ]
    }
}

impl<T: Lerpable<T, f64>, const N: usize> BezierKnownF64<T, N> {
    pub fn eval(&self, t: f64) -> T {
        bezier_eval_f64(t, &self.0[..])
    }

    pub fn split(&self, t: f64) -> [BezierKnownF64<T, N>; 2] {
        let [left, right] = bezier_split_f64(t, &self.0[..]);

        [
            BezierKnownF64::try_from(left).unwrap(),
            BezierKnownF64::try_from(right).unwrap(),
        ]
    }
}

impl<T: Lerpable<T, f32>, const N: usize> From<[T; N]> for BezierKnownF32<T, N> {
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T: Lerpable<T, f64>, const N: usize> From<[T; N]> for BezierKnownF64<T, N> {
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T: Lerpable<T, f32>, const N: usize> TryFrom<&[T]> for BezierKnownF32<T, N> {
    type Error = TryFromSliceError;

    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        Ok(Self(<[T; N]>::try_from(value)?))
    }
}

impl<T: Lerpable<T, f64>, const N: usize> TryFrom<&[T]> for BezierKnownF64<T, N> {
    type Error = TryFromSliceError;

    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        Ok(Self(<[T; N]>::try_from(value)?))
    }
}

impl<T: Lerpable<T, f32>, const N: usize> TryFrom<Vec<T>> for BezierKnownF32<T, N> {
    type Error = TryFromSliceError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self(<[T; N]>::try_from(&value[..])?))
    }
}

impl<T: Lerpable<T, f64>, const N: usize> TryFrom<Vec<T>> for BezierKnownF64<T, N> {
    type Error = TryFromSliceError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        Ok(Self(<[T; N]>::try_from(&value[..])?))
    }
}

impl<T: Lerpable<T, f32>, const N: usize> TryFrom<BezierF32<T>> for BezierKnownF32<T, N> {
    type Error = TryFromSliceError;

    fn try_from(value: BezierF32<T>) -> Result<Self, Self::Error> {
        Self::try_from(value.0)
    }
}

impl<T: Lerpable<T, f64>, const N: usize> TryFrom<BezierF64<T>> for BezierKnownF64<T, N> {
    type Error = TryFromSliceError;

    fn try_from(value: BezierF64<T>) -> Result<Self, Self::Error> {
        Self::try_from(value.0)
    }
}

impl<T: Lerpable<T, f32>, const N: usize> From<[T; N]> for BezierF32<T> {
    fn from(value: [T; N]) -> Self {
        Self(value.into())
    }
}

impl<T: Lerpable<T, f64>, const N: usize> From<[T; N]> for BezierF64<T> {
    fn from(value: [T; N]) -> Self {
        Self(value.into())
    }
}

impl<T: Lerpable<T, f32>, const N: usize> From<BezierKnownF32<T, N>> for BezierF32<T> {
    fn from(value: BezierKnownF32<T, N>) -> Self {
        Self(value.0.into())
    }
}

impl<T: Lerpable<T, f64>, const N: usize> From<BezierKnownF64<T, N>> for BezierF64<T> {
    fn from(value: BezierKnownF64<T, N>) -> Self {
        Self(value.0.into())
    }
}

impl<T: Lerpable<T, f32>> From<&[T]> for BezierF32<T> {
    fn from(value: &[T]) -> Self {
        Self(value.to_vec())
    }
}

impl<T: Lerpable<T, f64>> From<&[T]> for BezierF64<T> {
    fn from(value: &[T]) -> Self {
        Self(value.to_vec())
    }
}

impl<T: Lerpable<T, f32>> From<Vec<T>> for BezierF32<T> {
    fn from(value: Vec<T>) -> Self {
        Self(value)
    }
}

impl<T: Lerpable<T, f64>> From<Vec<T>> for BezierF64<T> {
    fn from(value: Vec<T>) -> Self {
        Self(value)
    }
}

impl<T: Lerpable<T, f64>> From<BezierF64<T>> for Vec<T> {
    fn from(value: BezierF64<T>) -> Self {
        value.0
    }
}

impl<T: Lerpable<T, f32>> From<BezierF32<T>> for Vec<T> {
    fn from(value: BezierF32<T>) -> Self {
        value.0
    }
}
