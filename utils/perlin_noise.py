import argparse
import numpy as np
import matplotlib.pyplot as plt


def fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def gradient(h: int, x: float, y: float) -> float:
    """Convert hash to one of 4 gradient directions."""
    vectors = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    g = vectors[h % 4]
    return g[..., 0] * x + g[..., 1] * y


def perlin(x: np.ndarray, y: np.ndarray, seed: int=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Build a permutation table
    p = np.arange(256, dtype=int)
    rng.shuffle(p)
    p = np.stack([p, p]).flatten()

    xi = x.astype(int) & 255
    yi = y.astype(int) & 255
    xf = x - x.astype(int)
    yf = y - y.astype(int)

    u = fade(xf)
    v = fade(yf)

    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)

    x_lerp = lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
    return x_lerp


def generate_perlin_image(grid_size: int, resolution: int=1024, seed: int=42) -> np.ndarray:
    lin = np.linspace(0, grid_size, resolution, endpoint=False)
    x, y = np.meshgrid(lin, lin)
    noise = perlin(x, y, seed=seed)
    return noise


def main():
    parser = argparse.ArgumentParser(description="Generate a 2D Perlin noise image.")
    parser.add_argument(
        "--grid_size",
        type=float,
        default=4,
        help="Fineness of the noise grid (higher = more features per image)"
             "Small values (e.g. 2) give large smooth blobs; large values (e.g. 10) give fine-grained texture.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Output image resolution in pixels (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save image to this file path instead of displaying it",
    )
    args = parser.parse_args()

    noise = generate_perlin_image(args.grid_size, args.resolution, args.seed)

    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(noise, cmap="YlOrBr", origin="lower")
    # ax.imshow(noise, cmap="YlOrBr", origin="lower")
    ax.imshow(noise, cmap="Reds", origin="lower")
    ax.axis("off")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=1024)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
