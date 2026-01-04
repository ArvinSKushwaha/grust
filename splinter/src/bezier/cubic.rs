use crate::bezier::{
    BezierKnownF32, BezierKnownF64, Lerpable, lerp_f32, lerp_f64,
    quadratic::{QuadraticBezierF32, QuadraticBezierF64},
    quadratic_bezier_f32, quadratic_bezier_f64,
};

pub fn cubic_bezier_f32<T: Lerpable<T, f32>>(t: f32, nodes: CubicBezierF32<T>) -> T {
    let nodes = nodes.0;

    if t <= 0. {
        return nodes[0];
    } else if t >= 1. {
        return nodes[3];
    }

    let rev_t = 1. - t;
    nodes[0] * rev_t * rev_t * rev_t
        + nodes[1] * 3. * rev_t * rev_t * t
        + nodes[2] * 3. * rev_t * t * t
        + nodes[3] * t * t * t
}

pub fn cubic_bezier_f64<T: Lerpable<T, f64>>(t: f64, nodes: CubicBezierF64<T>) -> T {
    let nodes = nodes.0;

    if t <= 0. {
        return nodes[0];
    } else if t >= 1. {
        return nodes[3];
    }

    let rev_t = 1. - t;
    nodes[0] * rev_t * rev_t * rev_t
        + nodes[1] * 3. * rev_t * rev_t * t
        + nodes[2] * 3. * rev_t * t * t
        + nodes[3] * t * t * t
}

pub fn cubic_split_f32<T: Lerpable<T, f32>>(
    t: f32,
    nodes: CubicBezierF32<T>,
) -> [CubicBezierF32<T>; 2] {
    let res = cubic_bezier_f32(t, nodes);

    let nodes = nodes.0;
    [
        BezierKnownF32([
            nodes[0],
            lerp_f32(t, BezierKnownF32([nodes[0], nodes[1]])),
            quadratic_bezier_f32(t, BezierKnownF32([nodes[0], nodes[1], nodes[2]])),
            res,
        ]),
        BezierKnownF32([
            res,
            quadratic_bezier_f32(t, BezierKnownF32([nodes[1], nodes[2], nodes[3]])),
            lerp_f32(t, BezierKnownF32([nodes[2], nodes[3]])),
            nodes[3],
        ]),
    ]
}
pub fn cubic_split_f64<T: Lerpable<T, f64>>(
    t: f64,
    nodes: CubicBezierF64<T>,
) -> [CubicBezierF64<T>; 2] {
    let res = cubic_bezier_f64(t, nodes);

    let nodes = nodes.0;
    [
        BezierKnownF64([
            nodes[0],
            lerp_f64(t, BezierKnownF64([nodes[0], nodes[1]])),
            quadratic_bezier_f64(t, BezierKnownF64([nodes[0], nodes[1], nodes[2]])),
            res,
        ]),
        BezierKnownF64([
            res,
            quadratic_bezier_f64(t, BezierKnownF64([nodes[1], nodes[2], nodes[3]])),
            lerp_f64(t, BezierKnownF64([nodes[2], nodes[3]])),
            nodes[3],
        ]),
    ]
}

pub fn cubic_degree_decrease_f32<T: Lerpable<T, f32>>(
    nodes: CubicBezierF32<T>,
) -> QuadraticBezierF32<T> {
    let nodes = nodes.0;

    let m = nalgebra::Matrix4x3::<f32>::new(
        1.,
        0.,
        0., //
        1. / 3.,
        2. / 3.,
        0., //
        0.,
        2. / 3.,
        1. / 3., //
        0.,
        0.,
        1., //
    );

    let transform = m.pseudo_inverse(1e-5).unwrap();

    let outnodes = [
        nodes[0] * transform[(0, 0)]
            + nodes[1] * transform[(0, 1)]
            + nodes[2] * transform[(0, 2)]
            + nodes[3] * transform[(0, 3)],
        nodes[0] * transform[(1, 0)]
            + nodes[1] * transform[(1, 1)]
            + nodes[2] * transform[(1, 2)]
            + nodes[3] * transform[(1, 3)],
        nodes[0] * transform[(2, 0)]
            + nodes[1] * transform[(2, 1)]
            + nodes[2] * transform[(2, 2)]
            + nodes[3] * transform[(2, 3)],
    ];

    BezierKnownF32(outnodes)
}

pub fn cubic_degree_decrease_f64<T: Lerpable<T, f64>>(
    nodes: CubicBezierF64<T>,
) -> QuadraticBezierF64<T> {
    let nodes = nodes.0;

    let m = nalgebra::Matrix4x3::<f64>::new(
        1.,
        0.,
        0., //
        1. / 3.,
        2. / 3.,
        0., //
        0.,
        2. / 3.,
        1. / 3., //
        0.,
        0.,
        1., //
    );

    let transform = m.pseudo_inverse(1e-5).unwrap();

    let outnodes = [
        nodes[0] * transform[(0, 0)]
            + nodes[1] * transform[(0, 1)]
            + nodes[2] * transform[(0, 2)]
            + nodes[3] * transform[(0, 3)],
        nodes[0] * transform[(1, 0)]
            + nodes[1] * transform[(1, 1)]
            + nodes[2] * transform[(1, 2)]
            + nodes[3] * transform[(1, 3)],
        nodes[0] * transform[(2, 0)]
            + nodes[1] * transform[(2, 1)]
            + nodes[2] * transform[(2, 2)]
            + nodes[3] * transform[(2, 3)],
    ];

    BezierKnownF64(outnodes)
}

pub type CubicBezierF32<T> = BezierKnownF32<T, 4>;
pub type CubicBezierF64<T> = BezierKnownF64<T, 4>;

pub type CubicBezier1D = CubicBezierF32<f32>;
pub type CubicBezier2D = CubicBezierF32<glam::Vec2>;
pub type CubicBezier3D = CubicBezierF32<glam::Vec3>;
pub type CubicBezier4D = CubicBezierF32<glam::Vec4>;

pub type DCubicBezier1D = CubicBezierF64<f64>;
pub type DCubicBezier2D = CubicBezierF64<glam::DVec2>;
pub type DCubicBezier3D = CubicBezierF64<glam::DVec3>;
pub type DCubicBezier4D = CubicBezierF64<glam::DVec4>;

pub fn num_quadratics(bezier: CubicBezier2D, tolerance: f32) -> usize {
    let [from, ctrl1, ctrl2, to] = bezier.0;

    let x = from + ctrl1 * -3. + ctrl2 * 3. + to * -1.;

    let err = x.length_squared() / (432.0 * tolerance * tolerance);

    // Avoid computing powf(1/6).ceil() using a lookup table that contains
    // i^6 for i in 1..25.
    const MAX_QUADS: usize = 16;
    const LUT: [f32; MAX_QUADS] = [
        1.0, 64.0, 729.0, 4096.0, 15625.0, 46656.0, 117649.0, 262144.0, 531441.0, 1000000.0,
        1771561.0, 2985984.0, 4826809.0, 7529536.0, 11390625.0, 16777216.0,
    ];

    if err <= 16777216.0 {
        #[allow(clippy::needless_range_loop)]
        for i in 0..MAX_QUADS {
            if err <= LUT[i] {
                return i + 1;
            }
        }
    }

    // If the number of quads does not fit in the LUT, fall back to the
    // expensive computation.
    err.powf(1. / 6.).ceil().max(1.) as usize
}
