use crate::{
    BezierKnownF32, BezierKnownF64, Lerpable,
    cubic::{CubicBezierF32, CubicBezierF64},
    lerp_f64,
    linear::{LinearBezierF32, LinearBezierF64, lerp_f32},
};

pub fn quadratic_bezier_f32<T: Lerpable<T, f32>>(t: f32, nodes: QuadraticBezierF32<T>) -> T {
    let nodes = nodes.0;
    if t <= 0. {
        return nodes[0];
    } else if t >= 1. {
        return nodes[2];
    }

    let rev_t = 1. - t;
    nodes[0] * rev_t * rev_t + nodes[1] * 2. * rev_t * t + nodes[2] * t * t
}

pub fn quadratic_bezier_f64<T: Lerpable<T, f64>>(t: f64, nodes: QuadraticBezierF64<T>) -> T {
    let nodes = nodes.0;
    if t <= 0. {
        return nodes[0];
    } else if t >= 1. {
        return nodes[2];
    }

    let rev_t = 1. - t;
    nodes[0] * rev_t * rev_t + nodes[1] * 2. * rev_t * t + nodes[2] * t * t
}

pub fn quadratic_split_f32<T: Lerpable<T, f32>>(
    t: f32,
    nodes: QuadraticBezierF32<T>,
) -> [QuadraticBezierF32<T>; 2] {
    let res = quadratic_bezier_f32(t, nodes);

    let nodes = nodes.0;
    [
        BezierKnownF32([
            nodes[0],
            lerp_f32(t, BezierKnownF32([nodes[0], nodes[1]])),
            res,
        ]),
        BezierKnownF32([
            res,
            lerp_f32(t, BezierKnownF32([nodes[1], nodes[2]])),
            nodes[2],
        ]),
    ]
}

pub fn quadratic_split_f64<T: Lerpable<T, f64>>(
    t: f64,
    nodes: QuadraticBezierF64<T>,
) -> [QuadraticBezierF64<T>; 2] {
    let res = quadratic_bezier_f64(t, nodes);

    let nodes = nodes.0;
    [
        BezierKnownF64([
            nodes[0],
            lerp_f64(t, BezierKnownF64([nodes[0], nodes[1]])),
            res,
        ]),
        BezierKnownF64([
            res,
            lerp_f64(t, BezierKnownF64([nodes[1], nodes[2]])),
            nodes[2],
        ]),
    ]
}

pub fn quadratic_degree_increase_f32<T: Lerpable<T, f32>>(
    nodes: QuadraticBezierF32<T>,
) -> CubicBezierF32<T> {
    let nodes = nodes.0;

    BezierKnownF32([
        nodes[0],
        (nodes[0] * (1. / 3.) + nodes[1] * (2. / 3.)),
        (nodes[1] * (2. / 3.) + nodes[2] * (1. / 3.)),
        nodes[2],
    ])
}

pub fn quadratic_degree_increase_f64<T: Lerpable<T, f64>>(
    nodes: QuadraticBezierF64<T>,
) -> CubicBezierF64<T> {
    let nodes = nodes.0;

    BezierKnownF64([
        nodes[0],
        (nodes[0] * (1. / 3.) + nodes[1] * (2. / 3.)),
        (nodes[1] * (2. / 3.) + nodes[2] * (1. / 3.)),
        nodes[2],
    ])
}

pub fn quadratic_degree_decrease_f32<T: Lerpable<T, f32>>(
    nodes: QuadraticBezierF32<T>,
) -> LinearBezierF32<T> {
    let nodes = nodes.0;

    let m = nalgebra::Matrix3x2::<f32>::new(
        1., 0., //
        1. / 2., 1. / 2., //
        0., 1., //
    );

    let transform = m.pseudo_inverse(1e-5).unwrap();

    let outnodes = [
        nodes[0] * transform[(0,0)] + nodes[1] * transform[(0,1)] + nodes[2] * transform[(0,2)],
        nodes[0] * transform[(1,0)] + nodes[1] * transform[(1,1)] + nodes[2] * transform[(1,2)],
    ];

    BezierKnownF32(outnodes)
}

pub fn quadratic_degree_decrease_f64<T: Lerpable<T, f64>>(
    nodes: QuadraticBezierF64<T>,
) -> LinearBezierF64<T> {
    let nodes = nodes.0;

    let m = nalgebra::Matrix3x2::<f64>::new(
        1., 0., //
        1. / 2., 1. / 2., //
        0., 1., //
    );

    let transform = m.pseudo_inverse(1e-5).unwrap();

    let outnodes = [
        nodes[0] * transform[(0,0)] + nodes[1] * transform[(0,1)] + nodes[2] * transform[(0,2)],
        nodes[0] * transform[(1,0)] + nodes[1] * transform[(1,1)] + nodes[2] * transform[(1,2)],
    ];

    BezierKnownF64(outnodes)
}

pub type QuadraticBezierF32<T> = BezierKnownF32<T, 3>;
pub type QuadraticBezierF64<T> = BezierKnownF64<T, 3>;

pub type QuadraticBezier1D = QuadraticBezierF32<f32>;
pub type QuadraticBezier2D = QuadraticBezierF32<glam::Vec2>;
pub type QuadraticBezier3D = QuadraticBezierF32<glam::Vec3>;
pub type QuadraticBezier4D = QuadraticBezierF32<glam::Vec4>;

pub type DQuadraticBezier1D = QuadraticBezierF64<f64>;
pub type DQuadraticBezier2D = QuadraticBezierF64<glam::DVec2>;
pub type DQuadraticBezier3D = QuadraticBezierF64<glam::DVec3>;
pub type DQuadraticBezier4D = QuadraticBezierF64<glam::DVec4>;

impl QuadraticBezier2D {
    pub fn sdf(pos: glam::Vec2) -> f32 {
        use glam::*;

        todo!();
    }
}
