use crate::bezier::{
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
        1.,
        0., //
        1. / 2.,
        1. / 2., //
        0.,
        1., //
    );

    let transform = m.pseudo_inverse(1e-5).unwrap();

    let outnodes = [
        nodes[0] * transform[(0, 0)] + nodes[1] * transform[(0, 1)] + nodes[2] * transform[(0, 2)],
        nodes[0] * transform[(1, 0)] + nodes[1] * transform[(1, 1)] + nodes[2] * transform[(1, 2)],
    ];

    BezierKnownF32(outnodes)
}

pub fn quadratic_degree_decrease_f64<T: Lerpable<T, f64>>(
    nodes: QuadraticBezierF64<T>,
) -> LinearBezierF64<T> {
    let nodes = nodes.0;

    let m = nalgebra::Matrix3x2::<f64>::new(
        1.,
        0., //
        1. / 2.,
        1. / 2., //
        0.,
        1., //
    );

    let transform = m.pseudo_inverse(1e-5).unwrap();

    let outnodes = [
        nodes[0] * transform[(0, 0)] + nodes[1] * transform[(0, 1)] + nodes[2] * transform[(0, 2)],
        nodes[0] * transform[(1, 0)] + nodes[1] * transform[(1, 1)] + nodes[2] * transform[(1, 2)],
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
    pub fn sdf(&self, pos: glam::Vec2) -> f32 {
        use glam::*;

        let [node1, node2, node3] = self.0;

        let a = node2 - node1;
        let b = node1 - node2 * 2.0 + node3;
        let c = a * 2.0;

        let d = node1 - pos;

        let kk = 1.0 / b.dot(b);
        let kx = a.dot(b) * kk;
        let ky = (2.0 * a.dot(a) + d.dot(b)) * kk / 3.0;
        let kz = kk * d.dot(a);
        let p = ky - kx * kx;

        let p3 = p * p * p;
        let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
        let mut h = q * q + 4.0 * p3;

        let res = if h >= 0.0 {
            h = h.sqrt();
            let x = (vec2(h, -h) - q) / 2.0;
            let uv = x.signum() * x.abs().powf(1.0 / 3.0);
            let t = (uv.x + uv.y - kx).clamp(0., 1.);
            (d + (c + b * t) * t).length_squared()
        } else {
            let z = (-p).sqrt();
            let v = (q / (p * z * 2.0)).acos() / 3.0;
            let m = v.cos();
            let n = v.sin() * (3.0f32).sqrt();
            let t = (vec3(m + m, -n - m, n - m) * z - kx).clamp(Vec3::ZERO, Vec3::ONE);
            (d + (c + b * t.x) * t.x)
                .length_squared()
                .min((d + (c + b * t.y) * t.y).length_squared())
        };

        res.sqrt()
    }
}

// Quadratic Bezier - exact   (https://www.shadertoy.com/view/MlKcDD)
//
// float sdBezier( in vec2 pos, in vec2 A, in vec2 B, in vec2 C )
// {
//     vec2 a = B - A;
//     vec2 b = A - 2.0*B + C;
//     vec2 c = a * 2.0;
//     vec2 d = A - pos;
//     float kk = 1.0/dot(b,b);
//     float kx = kk * dot(a,b);
//     float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
//     float kz = kk * dot(d,a);
//     float res = 0.0;
//     float p = ky - kx*kx;
//     float p3 = p*p*p;
//     float q = kx*(2.0*kx*kx-3.0*ky) + kz;
//     float h = q*q + 4.0*p3;
//     if( h >= 0.0)
//     {
//         h = sqrt(h);
//         vec2 x = (vec2(h,-h)-q)/2.0;
//         vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
//         float t = clamp( uv.x+uv.y-kx, 0.0, 1.0 );
//         res = dot2(d + (c + b*t)*t);
//     }
//     else
//     {
//         float z = sqrt(-p);
//         float v = acos( q/(p*z*2.0) ) / 3.0;
//         float m = cos(v);
//         float n = sin(v)*1.732050808;
//         vec3  t = clamp(vec3(m+m,-n-m,n-m)*z-kx,0.0,1.0);
//         res = min( dot2(d+(c+b*t.x)*t.x),
//                    dot2(d+(c+b*t.y)*t.y) );
//         // the third root cannot be the closest
//         // res = min(res,dot2(d+(c+b*t.z)*t.z));
//     }
//     return sqrt( res );
// }
