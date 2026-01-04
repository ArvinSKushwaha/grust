use std::sync::OnceLock;

use eframe::egui::{self, Color32, ColorImage};
use egui_plot::{
    Line, Plot, PlotImage, PlotItem, PlotPoint, PlotPoints, PlotUi, Points, Polygon,
    color_from_strength,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use splinter::bezier::{
    bezier_degree_decrease_f32, bezier_degree_increase_f32, bezier_eval_f32, bezier_split_f32,
    cubic::{CubicBezier2D, num_quadratics},
    cubic_degree_decrease_f32, cubic_split_f32,
    quadratic::QuadraticBezier2D,
};

fn main() {
    let mut points = vec![vec![
        glam::vec2(-1., -0.5),
        glam::vec2(1., -1.), //
        glam::vec2(-1., 0.5),
    ]];

    let mut durations = std::collections::VecDeque::new();
    durations.push_back(std::time::Instant::now());

    let mut add_queue = vec![];
    let res = ((200usize * 16) / 9, 200usize);
    let mut grid = vec![0; res.0 * res.1];
    let mut texture_handle = OnceLock::new();

    eframe::run_simple_native(
        "Spliter Example: Toy",
        eframe::NativeOptions::default(),
        move |ctx, _| {
            durations.push_back(std::time::Instant::now());

            if durations.len() > 100 {
                durations.pop_front();
            }

            egui::TopBottomPanel::top("control-buttons").show(ctx, |ui| {
                ui.add_space(2.);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(points[0].len() > 2, egui::Button::new("Remove Node"))
                        .clicked()
                    {
                        points[0] = bezier_degree_decrease_f32(&points[0]).into();
                    }

                    if ui.add(egui::Button::new("Add Node")).clicked() {
                        points[0] = bezier_degree_increase_f32(&points[0]).into();
                    }

                    if durations.len() > 5 {
                        let dur = (0..durations.len() - 1)
                            .map(|i| durations[i + 1].duration_since(durations[i]).as_secs_f64())
                            .sum::<f64>()
                            / (durations.len() as f64 - 1.0);
                        ui.label(format!("FPS: {:.2} Hz", dur.recip()));
                    }
                });
                ui.add_space(2.);
            });

            egui::Window::new("debug").show(ctx, |ui| {
                ctx.texture_ui(ui);
            });

            egui::CentralPanel::default().show(ctx, |ui| {
                let mut plot = Plot::new("plot")
                    .data_aspect(1.0)
                    .show_grid(false)
                    .allow_drag(false)
                    .cursor_color(Color32::TRANSPARENT)
                    .show_x(false)
                    .show_y(false);

                if ctx.cumulative_pass_nr() == 0 {
                    plot = plot.default_x_bounds(-2., 2.).default_y_bounds(-2., 2.);
                }

                let response = plot.show(ui, |plot_ui| {
                    let pointer = plot_ui
                        .pointer_coordinate()
                        .map(|i| glam::dvec2(i.x, i.y).as_vec2());
                    let drag = {
                        let i = plot_ui.pointer_coordinate_drag_delta();
                        glam::vec2(i.x, i.y)
                    };

                    let bounds = plot_ui.plot_bounds();
                    let (&x_min, &x_max) = (bounds.range_x().start(), bounds.range_x().end());
                    let (&y_min, &y_max) = (bounds.range_y().start(), bounds.range_y().end());

                    let dx = (
                        (x_max - x_min) / (res.0 as f64),
                        (y_max - y_min) / (res.1 as f64),
                    );

                    let subdivided = points
                        .par_iter()
                        .flat_map(|i| {
                            if i.len() == 3 {
                                QuadraticBezier2D::try_from(i.as_slice())
                                    .into_iter()
                                    .collect::<Vec<_>>()
                            } else if i.len() == 4 {
                                let mut cb = CubicBezier2D::try_from(i.as_slice()).unwrap();
                                let num = num_quadratics(cb, 1e-2);

                                let mut bz = vec![];
                                for i in 0..num {
                                    let [left, right] =
                                        cubic_split_f32(1. / (num - i + 1) as f32, cb);
                                    bz.push(cubic_degree_decrease_f32(left));
                                    cb = right;
                                }

                                bz.push(cubic_degree_decrease_f32(cb));

                                bz
                            } else {
                                vec![]
                            }
                        })
                        .collect::<Vec<_>>();

                    grid.par_iter_mut().enumerate().for_each(|(idx, p)| {
                        let [i, j] = [idx % res.0, idx / res.0];

                        let (x, y) = ((i as f32 / res.0 as f32), (j as f32 / res.1 as f32));
                        let (x, y) = (
                            (x_max as f32 - x_min as f32) * x + x_min as f32,
                            (y_max as f32 - y_min as f32) * y + y_min as f32,
                        );

                        if let Some(dist) = subdivided
                            .par_iter()
                            .map(|i| i.sdf(glam::vec2(x, y)))
                            .min_by(f32::total_cmp)
                        {
                            *p = ((30. / (x_max - x_min) as f32 * dist).sin().powi(2) * 0.1 * 255.)
                                as u8;
                        }
                    });

                    let _ = texture_handle.get_or_init(|| {
                        ctx.load_texture(
                            "sdf",
                            ColorImage::from_gray([res.0, res.1], &vec![0; res.0 * res.1]),
                            egui::TextureOptions {
                                magnification: egui::TextureFilter::Linear,
                                minification: egui::TextureFilter::Linear,
                                wrap_mode: egui::TextureWrapMode::ClampToEdge,
                                mipmap_mode: None,
                            },
                        )
                    });

                    if let Some(handle) = texture_handle.get_mut() {
                        handle.set(
                            ColorImage::from_gray([res.0, res.1], &grid),
                            egui::TextureOptions {
                                magnification: egui::TextureFilter::Linear,
                                minification: egui::TextureFilter::Linear,
                                wrap_mode: egui::TextureWrapMode::ClampToEdge,
                                mipmap_mode: None,
                            },
                        );
                    }

                    // for (x, y, dist) in grid {
                    //     let box_shape = vec![
                    //         [x as f64, y as f64],
                    //         [x as f64 + dx.0, y as f64],
                    //         [x as f64 + dx.0, y as f64 + dx.1],
                    //         [x as f64, y as f64 + dx.1],
                    //     ];
                    //
                    //     let color = if dist != f32::INFINITY {
                    //         Color32::from(egui::epaint::Rgba::from_gray(dist))
                    //     } else {
                    //         Color32::DARK_RED
                    //     };
                    //
                    //     let poly = Polygon::new(format!("{x}a{y}"), box_shape)
                    //         .fill_color(color)
                    //         .stroke((1., egui::epaint::Rgba::from_gray(dist)));
                    //     plot_ui.polygon(poly);
                    // }

                    let mut closests = Vec::new();
                    points.append(&mut add_queue);

                    let mut dragged_point = false;
                    for points in &mut points {
                        for point in points.iter_mut() {
                            if let Some(pointer) = pointer
                                && pointer.distance_squared(*point) < (0.0005)
                            {
                                *point += drag;
                                dragged_point = true;
                            }
                        }

                        let vec_points = points
                            .iter()
                            .map(|a| a.as_dvec2().to_array())
                            .collect::<Vec<_>>();

                        if let Some(handle) = texture_handle.get() {
                            plot_ui.image(
                                PlotImage::new(
                                    "sdf_image",
                                    handle,
                                    PlotPoint::new((x_min + x_max) / 2., (y_min + y_max) / 2.),
                                    [(x_max - x_min) as f32, (y_max - y_min) as f32],
                                )
                                .uv([egui::pos2(0., 1.), egui::pos2(1., 0.)]),
                            );
                        }

                        plot_ui.line(
                            Line::new("Control Line", vec_points.clone())
                                .color(Color32::ORANGE.linear_multiply(0.5)),
                        );

                        plot_ui.points(
                            Points::new("Control Points", vec_points.clone())
                                .shape(egui_plot::MarkerShape::Circle)
                                .color(Color32::ORANGE.linear_multiply(0.5))
                                .radius(5.),
                        );

                        const N: usize = 1000;
                        let mut ts = Vec::with_capacity(N);
                        let mut discretized = Vec::with_capacity(N);

                        for i in 0..N {
                            let t = (i as f32) / (N as f32 - 1.);
                            let p = bezier_eval_f32(t, points);

                            discretized.push(p.as_dvec2().to_array());
                            ts.push(t);
                        }

                        let closest_point = pointer
                            .iter()
                            .flat_map(|&pointer| {
                                discretized
                                    .iter()
                                    .enumerate()
                                    .map(|(i, p)| {
                                        (
                                            i,
                                            pointer.distance_squared(glam::vec2(
                                                p[0] as f32,
                                                p[1] as f32,
                                            )),
                                        )
                                    })
                                    .min_by(|i, j| i.1.total_cmp(&j.1))
                                    .map(|(i, dist)| (i, ts[i], discretized[i], dist))
                            })
                            .next();

                        if let Some((_, t, _, dist)) = closest_point
                            && dist < 0.05
                        {
                            let n = points.len() - 1;
                            let mut space = points.clone();

                            for k in 0..n {
                                for i in 0..(n - k) {
                                    space[i] = space[i] * (1. - t) + space[i + 1] * t;
                                }

                                plot_ui.line(
                                    Line::new(
                                        format!("Control Line depth {k}"),
                                        space[..(n - k)]
                                            .iter()
                                            .map(|i| i.as_dvec2().to_array())
                                            .collect::<Vec<_>>(),
                                    )
                                    .color(Color32::GREEN.linear_multiply(0.5)),
                                );
                            }
                        }

                        let bezier_line =
                            Line::new("bezier", discretized).stroke((2.0, Color32::BLUE));

                        plot_ui.add(bezier_line);

                        closests.push(closest_point);
                    }

                    if !dragged_point {
                        plot_ui.translate_bounds(-egui::vec2(drag.x, drag.y));
                    }

                    closests
                });

                if response.response.double_clicked() {
                    for (i, closest) in response.inner.iter().enumerate() {
                        if let &Some((_idx, t, p, dist)) = closest
                            && dist < 0.01
                        {
                            let [left, right] = bezier_split_f32(t, &points[i][..]);

                            points[i] = left;
                            add_queue.push(right);
                        }
                    }
                }
            });
        },
    )
    .unwrap();
}
