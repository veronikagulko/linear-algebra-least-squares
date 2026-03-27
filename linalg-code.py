import numpy as np
from PIL import Image

def load_and_binarize_image(image_path):
    img = Image.open(image_path).convert("L")
    gray = np.array(img)
    mask = gray < 250
    return gray, mask

def extract_boundary(mask):
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not mask[y, x]:
                continue

            if (
                not mask[y - 1, x]
                or not mask[y + 1, x]
                or not mask[y, x - 1]
                or not mask[y, x + 1]
            ):
                boundary[y, x] = True

    ys, xs = np.where(boundary)
    y_cartesian = h - 1 - ys
    return np.column_stack((xs.astype(float), y_cartesian.astype(float)))

def fit_circle_least_squares(points: np.ndarray):
    x = points[:, 0]
    y = points[:, 1]

    # M matrix: [x, y, 1]
    M = np.column_stack((x, y, np.ones_like(x)))
    # b vector: -(x^2 + y^2)
    b = -(x**2 + y**2)

    # Calculate coefficients using (M^T * M)^-1 * M^T * b
    M_T = M.T
    M_T_M_inv = np.linalg.inv(M_T @ M)
    coeffs = M_T_M_inv @ M_T @ b
    
    A, B, C = coeffs

    h = -A / 2
    k = -B / 2
    r = np.sqrt(h**2 + k**2 - C)

    distances = np.sqrt((x - h)**2 + (y - k)**2)
    residuals = distances - r

    sse = np.sum(residuals**2)
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)

    return {
        "A": A,
        "B": B,
        "C": C,
        "center": (h, k),
        "radius": r,
        "residuals": residuals,
        "sse": sse,
        "mse": mse,
        "rmse": rmse,
        "relative_rmse": rmse / r
    }

def analyze_image(image_path: str):
    gray, mask = load_and_binarize_image(image_path)
    points = extract_boundary(mask)
    fit = fit_circle_least_squares(points)

    A = fit["A"]
    B = fit["B"]
    C = fit["C"]
    h, k = fit["center"]
    r = fit["radius"]
    
    # rad(MSE) is the Root Mean Squared Error (RMSE)
    relative_error = fit["rmse"] / r

    print(f"A: {A:.4f}")
    print(f"B: {B:.4f}")
    print(f"C: {C:.4f}")
    print(f"Center (h,k): ({h:.4f}, {k:.4f})")
    print(f"Radius: {r:.4f}")
    
    # Standard circle equation: (x - h)^2 + (y - k)^2 = r^2
    print(f"Equation: (x - {h:.4f})^2 + (y - {k:.4f})^2 = {r**2:.4f}")
    
    print(f"sqrt(MSE) / radius: {relative_error:.4f}")
    
    # Final Decision
    if relative_error < 0.05:
        print("Final Decision: Pass (Good fit)")
    else:
        print("Final Decision: Fail (Poor fit)")

if __name__ == "__main__":
    analyze_image("heart.png")