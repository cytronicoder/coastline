"""Example script demonstrating validation utilities."""

from pathlib import Path

from coastline_analysis import (
    aggregate_counts,
    boxcount_series,
    generate_noise_curve,
    generate_straight_line,
    plot_geometry,
    plot_loglog,
    run_sanity_checks,
)

EPS_LIST = [512, 256, 128, 64, 32, 16]
OFFSETS = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
ROTATIONS = [0.0, 15.0, 30.0, 45.0]


def main():
    straight = generate_straight_line()
    noise = generate_noise_curve()

    # Plot geometries
    plot_geometry(straight, title="Straight Line Geometry", save_path=Path("straight_geometry.png"))
    plot_geometry(noise, title="Noise Curve Geometry", save_path=Path("noise_geometry.png"))

    # Compute box counts and plot log-log
    straight_counts = boxcount_series(straight, EPS_LIST, OFFSETS, ROTATIONS)
    straight_aggregated = aggregate_counts(straight_counts)
    plot_loglog(straight_aggregated, title="Straight Line Log-Log Plot", save_path=Path("straight_loglog.png"))

    noise_counts = boxcount_series(noise, EPS_LIST, OFFSETS, ROTATIONS)
    noise_aggregated = aggregate_counts(noise_counts)
    plot_loglog(noise_aggregated, title="Noise Curve Log-Log Plot", save_path=Path("noise_loglog.png"))

    straight_summary = run_sanity_checks(straight, EPS_LIST, OFFSETS, ROTATIONS)
    noise_summary = run_sanity_checks(noise, EPS_LIST, OFFSETS, ROTATIONS)

    print("Straight line summary:\n", straight_summary)
    print("Noise curve summary:\n", noise_summary)


if __name__ == "__main__":
    main()
