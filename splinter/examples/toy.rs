use eframe::egui::{self, Color32};
use egui_plot::{Line, Plot, PlotItem, PlotPoint, PlotPoints, Points};
use splinter::{bezier_degree_decrease_f32, bezier_degree_increase_f32, bezier_eval_f32, bezier_split_f32};

fn main() {
    let mut points = vec![vec![
        glam::vec2(-1., -0.5),
        glam::vec2(1., 1.),  //
        glam::vec2(1., -1.), //
        glam::vec2(-1., 0.5),
    ]];

    let mut add_queue = vec![];

    eframe::run_simple_native(
        "Spliter Example: Toy",
        eframe::NativeOptions::default(),
        move |ctx, _| {
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
                });
                ui.add_space(2.);
            });

            egui::CentralPanel::default().show(ctx, |ui| {
                let response = Plot::new("plot")
                    .data_aspect(1.0)
                    .show_grid(false)
                    .allow_drag(false)
                    .cursor_color(Color32::TRANSPARENT)
                    .auto_bounds(false)
                    .show_x(false)
                    .show_y(false)
                    .default_x_bounds(-2., 2.)
                    .default_y_bounds(-2., 2.)
                    .show(ui, |plot_ui| {
                        let pointer = plot_ui
                            .pointer_coordinate()
                            .map(|i| glam::dvec2(i.x, i.y).as_vec2());
                        let drag = {
                            let i = plot_ui.pointer_coordinate_drag_delta();
                            glam::vec2(i.x, i.y)
                        };

                        let mut closests = Vec::new();
                        points.append(&mut add_queue);

                        for points in &mut points {
                            for point in points.iter_mut() {
                                if let Some(pointer) = pointer
                                    && pointer.distance_squared(*point) < (0.001)
                                {
                                    *point += drag;
                                }
                            }

                            let vec_points = points
                                .iter()
                                .map(|a| a.as_dvec2().to_array())
                                .collect::<Vec<_>>();

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
                                let p = bezier_eval_f32(t, &points);

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
                        
                        closests
                    });

                if response.response.clicked() {
                    for (i, closest) in response.inner.iter().enumerate() {
                        if let &Some((_idx, t, p, dist)) = closest && dist < 0.01 {
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
