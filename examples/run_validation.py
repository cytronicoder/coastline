"""Example script demonstrating validation utilities."""

from coastline_analysis import (
    generate_noise_curve,
    generate_straight_line,
    run_sanity_checks,
)

EPS_LIST = [512, 256, 128, 64, 32, 16]
OFFSETS = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
ROTATIONS = [0.0, 15.0, 30.0, 45.0]


def main():
    straight = generate_straight_line()
    noise = generate_noise_curve()

    straight_summary = run_sanity_checks(straight, EPS_LIST, OFFSETS, ROTATIONS)
    noise_summary = run_sanity_checks(noise, EPS_LIST, OFFSETS, ROTATIONS)

    print("Straight line summary:\n", straight_summary)
    print("Noise curve summary:\n", noise_summary)


if __name__ == "__main__":
    main()
