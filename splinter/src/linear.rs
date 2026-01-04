use crate::{BezierKnownF32, BezierKnownF64, Lerpable};

pub fn lerp_f32<T: Lerpable<T, f32>>(t: f32, nodes: LinearBezierF32<T>) -> T {
    let nodes = nodes.0;

    if t <= 0. {
        nodes[0]
    } else if t >= 1. {
        nodes[1]
    } else {
        nodes[0] * (1. - t) + nodes[1] * t
    }
}

pub fn lerp_f64<T: Lerpable<T, f64>>(t: f64, nodes: LinearBezierF64<T>) -> T {
    let nodes = nodes.0;

    if t <= 0. {
        nodes[0]
    } else if t >= 1. {
        nodes[1]
    } else {
        nodes[0] * (1. - t) + nodes[1] * t
    }
}

pub type LinearBezierF32<T> = BezierKnownF32<T, 2>;
pub type LinearBezierF64<T> = BezierKnownF64<T, 2>;

pub type LinearBezier1D = LinearBezierF32<f32>;
pub type LinearBezier2D = LinearBezierF32<glam::Vec2>;
pub type LinearBezier3D = LinearBezierF32<glam::Vec3>;
pub type LinearBezier4D = LinearBezierF32<glam::Vec4>;

pub type DLinearBezier1D = LinearBezierF64<f64>;
pub type DLinearBezier2D = LinearBezierF64<glam::DVec2>;
pub type DLinearBezier3D = LinearBezierF64<glam::DVec3>;
pub type DLinearBezier4D = LinearBezierF64<glam::DVec4>;
