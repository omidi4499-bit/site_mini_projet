from __future__ import annotations

import base64
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from urllib import request
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
EPS = 1e-9

Point = Tuple[float, float]
Vector3 = Tuple[float, float, float]

HF_MODEL_URL = os.getenv("HF_DEPTH_API_URL", "https://api-inference.huggingface.co/models/Intel/dpt-hybrid-midas")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "3"))

UNIT_FACTORS = {
    "cm": 1.0,
    "mm": 10.0,
    "m": 0.01,
    "in": 1.0 / 2.54,
}


@dataclass
class DepthMap:
    width: int
    height: int
    values: List[float]
    model_name: str


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(vec: Vector3) -> float:
    return math.sqrt(dot(vec, vec))


def normalize(vec: Vector3) -> Vector3:
    length = norm(vec)
    if length < EPS:
        return (0.0, 0.0, 1.0)
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def parse_points(raw_points: Any, expected: int, label: str) -> List[Point]:
    if not isinstance(raw_points, list) or len(raw_points) != expected:
        raise ValueError(f"{label}: {expected} points requis.")

    points: List[Point] = []
    for idx, point in enumerate(raw_points):
        if not isinstance(point, dict):
            raise ValueError(f"{label}: format invalide au point {idx + 1}.")

        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label}: coordonnees invalides au point {idx + 1}.") from exc

        points.append((x, y))

    return points


def distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def distance3(a: Vector3, b: Vector3) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def decode_data_url_image(data_url: Any) -> bytes:
    if not isinstance(data_url, str) or not data_url:
        raise ValueError("image_data_url manquant.")

    payload = data_url
    if "," in data_url:
        payload = data_url.split(",", 1)[1]

    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("image_data_url invalide.") from exc


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)

    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_to_grayscale_values(png_bytes: bytes) -> Tuple[int, int, List[float]]:
    signature = b"\x89PNG\r\n\x1a\n"
    if not png_bytes.startswith(signature):
        raise ValueError("Reponse image non exploitable.")

    offset = len(signature)
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    idat_chunks: List[bytes] = []

    while offset + 8 <= len(png_bytes):
        length = struct.unpack(">I", png_bytes[offset:offset + 4])[0]
        offset += 4
        chunk_type = png_bytes[offset:offset + 4]
        offset += 4
        chunk_data = png_bytes[offset:offset + length]
        offset += length
        offset += 4

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", chunk_data
            )
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError("Image depth non supportee.")
            if bit_depth != 8:
                raise ValueError("Image depth non supportee.")
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width <= 0 or height <= 0 or not idat_chunks:
        raise ValueError("Image depth incomplete.")

    if color_type == 0:
        channels = 1
    elif color_type == 2:
        channels = 3
    elif color_type == 6:
        channels = 4
    else:
        raise ValueError("Format image non supporte.")

    raw = zlib.decompress(b"".join(idat_chunks))
    stride = width * channels

    expected_size = height * (1 + stride)
    if len(raw) < expected_size:
        raise ValueError("Image depth tronquee.")

    values: List[float] = [0.0] * (width * height)
    prev_scanline = [0] * stride
    raw_offset = 0

    for row in range(height):
        filter_type = raw[raw_offset]
        raw_offset += 1

        scanline = list(raw[raw_offset:raw_offset + stride])
        raw_offset += stride

        if filter_type == 1:
            for i in range(stride):
                left = scanline[i - channels] if i >= channels else 0
                scanline[i] = (scanline[i] + left) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                scanline[i] = (scanline[i] + prev_scanline[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = scanline[i - channels] if i >= channels else 0
                up = prev_scanline[i]
                scanline[i] = (scanline[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = scanline[i - channels] if i >= channels else 0
                up = prev_scanline[i]
                up_left = prev_scanline[i - channels] if i >= channels else 0
                scanline[i] = (scanline[i] + _paeth_predictor(left, up, up_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError("Filtre image inconnu.")

        for col in range(width):
            idx = row * width + col
            px_offset = col * channels

            if channels == 1:
                gray = scanline[px_offset]
            else:
                r = scanline[px_offset]
                g = scanline[px_offset + 1]
                b = scanline[px_offset + 2]
                gray = int((r + g + b) / 3)

            values[idx] = float(gray)

        prev_scanline = scanline

    return width, height, values


def percentile(sorted_values: Sequence[float], ratio: float) -> float:
    if not sorted_values:
        raise ValueError("Depth map vide.")

    ratio = clamp(ratio, 0.0, 1.0)
    position = ratio * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))

    if low == high:
        return float(sorted_values[low])

    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def depth_map_from_hf(image_bytes: bytes) -> DepthMap:
    if not HF_TOKEN:
        raise RuntimeError("Service indisponible.")

    req = request.Request(HF_MODEL_URL, data=image_bytes, method="POST")
    req.add_header("Authorization", f"Bearer {HF_TOKEN}")
    req.add_header("Content-Type", "application/octet-stream")
    req.add_header("Accept", "image/png,application/json")

    with request.urlopen(req, timeout=HF_TIMEOUT) as response:
        response_bytes = response.read()
        content_type = (response.headers.get("Content-Type") or "").lower()

    if "application/json" in content_type:
        payload = json.loads(response_bytes.decode("utf-8", errors="ignore"))
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError("Service indisponible.")

        encoded = None
        if isinstance(payload, dict):
            encoded = payload.get("predicted_depth") or payload.get("depth") or payload.get("image")

        if not encoded:
            raise RuntimeError("Service indisponible.")

        if isinstance(encoded, dict) and "data" in encoded:
            encoded = encoded["data"]

        if not isinstance(encoded, str):
            raise RuntimeError("Service indisponible.")

        if "," in encoded:
            encoded = encoded.split(",", 1)[1]

        try:
            response_bytes = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise RuntimeError("Service indisponible.") from exc

    width, height, values = decode_png_to_grayscale_values(response_bytes)

    return DepthMap(
        width=width,
        height=height,
        values=values,
        model_name="modele-profondeur",
    )


def get_depth_map(image_bytes: bytes) -> DepthMap:
    try:
        return depth_map_from_hf(image_bytes)
    except Exception as exc:
        return fallback_depth_map()


def fallback_depth_map() -> DepthMap:
    values = [128.0]
    return DepthMap(
        width=1,
        height=1,
        values=values,
        model_name="modele-local",
    )


def map_image_point_to_depth_coords(
    point: Point,
    image_width: int,
    image_height: int,
    depth_width: int,
    depth_height: int,
) -> Tuple[int, int]:
    px = clamp(point[0], 0.0, float(max(0, image_width - 1)))
    py = clamp(point[1], 0.0, float(max(0, image_height - 1)))

    x = int(round(px * max(0, depth_width - 1) / max(1, image_width - 1)))
    y = int(round(py * max(0, depth_height - 1) / max(1, image_height - 1)))

    x = int(clamp(x, 0, max(0, depth_width - 1)))
    y = int(clamp(y, 0, max(0, depth_height - 1)))
    return x, y


def sample_depth_stats(
    depth_map: DepthMap,
    point: Point,
    image_width: int,
    image_height: int,
    radius: int = 2,
) -> Tuple[float, float]:
    center_x, center_y = map_image_point_to_depth_coords(
        point,
        image_width,
        image_height,
        depth_map.width,
        depth_map.height,
    )

    samples: List[float] = []
    for y in range(max(0, center_y - radius), min(depth_map.height, center_y + radius + 1)):
        row_offset = y * depth_map.width
        for x in range(max(0, center_x - radius), min(depth_map.width, center_x + radius + 1)):
            samples.append(depth_map.values[row_offset + x])

    if not samples:
        idx = center_y * depth_map.width + center_x
        samples = [depth_map.values[idx]]

    mean = sum(samples) / len(samples)
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    return mean, math.sqrt(variance)


def normalize_depth_to_cm(raw_depth: float, depth_map: DepthMap) -> float:
    if depth_map.width * depth_map.height <= 1:
        return 90.0

    sorted_values = sorted(depth_map.values)
    p10 = percentile(sorted_values, 0.10)
    p90 = percentile(sorted_values, 0.90)

    if abs(p90 - p10) < EPS:
        return 150.0

    normalized = (raw_depth - p10) / (p90 - p10)
    normalized = clamp(normalized, 0.0, 1.0)
    return 40.0 + normalized * 260.0


def build_camera_intrinsics(image_width: int) -> Tuple[float, float, float]:
    fov = math.radians(60.0)
    focal_px = (image_width / 2.0) / math.tan(fov / 2.0)
    cx = image_width / 2.0
    cy = None
    return focal_px, cx, cy


def project_pixel_to_3d(point: Point, depth_cm: float, image_width: int, image_height: int) -> Vector3:
    focal_px = (image_width / 2.0) / math.tan(math.radians(60.0) / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0
    x = (point[0] - cx) * depth_cm / focal_px
    y = (point[1] - cy) * depth_cm / focal_px
    return (x, y, depth_cm)


def sample_segment_points(a: Point, b: Point, count: int = 12) -> List[Point]:
    points: List[Point] = []
    for i in range(count):
        t = i / max(1, count - 1)
        points.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return points


def plane_normal_from_params(params: Sequence[float]) -> Vector3:
    theta, phi = params[0], params[1]
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def plane_residuals(params: Sequence[float], points: Sequence[Vector3]) -> List[float]:
    normal = plane_normal_from_params(params)
    d = params[2]
    return [dot(normal, point) + d for point in points]


def numeric_jacobian(params: Sequence[float], points: Sequence[Vector3], eps: float = 1e-6) -> Tuple[List[List[float]], List[float]]:
    base = plane_residuals(params, points)
    jacobian = [[0.0 for _ in params] for _ in base]

    for j in range(len(params)):
        shifted = list(params)
        shifted[j] += eps
        shifted_residual = plane_residuals(shifted, points)
        for i in range(len(base)):
            jacobian[i][j] = (shifted_residual[i] - base[i]) / eps

    return jacobian, base


def mat_transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(column) for column in zip(*matrix)]


def mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)

    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            for j in range(cols):
                result[i][j] += aik * b[k][j]

    return result


def mat_vec_mul(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(m_ij * v_j for m_ij, v_j in zip(row, vector)) for row in matrix]


def solve_linear_system(a: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    n = len(a)
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        raise ValueError("Systeme lineaire invalide.")

    augmented = [list(row) + [b[idx]] for idx, row in enumerate(a)]

    for column in range(n):
        pivot_row = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        pivot_value = augmented[pivot_row][column]

        if abs(pivot_value) < EPS:
            raise ValueError("Systeme lineaire singulier.")

        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]

        pivot_value = augmented[column][column]
        for j in range(column, n + 1):
            augmented[column][j] /= pivot_value

        for row in range(n):
            if row == column:
                continue

            factor = augmented[row][column]
            if factor == 0.0:
                continue

            for j in range(column, n + 1):
                augmented[row][j] -= factor * augmented[column][j]

    return [augmented[i][n] for i in range(n)]


def fit_plane_gn(points: Sequence[Vector3], max_iter: int = 20) -> Tuple[Vector3, float, float]:
    if len(points) < 3:
        raise ValueError("Points insuffisants pour ajuster un plan.")

    base_normal = normalize(cross(
        (points[1][0] - points[0][0], points[1][1] - points[0][1], points[1][2] - points[0][2]),
        (points[2][0] - points[0][0], points[2][1] - points[0][1], points[2][2] - points[0][2]),
    ))
    theta0 = math.acos(clamp(base_normal[2], -1.0, 1.0))
    phi0 = math.atan2(base_normal[1], base_normal[0])
    d0 = -dot(base_normal, points[0])

    params = [theta0, phi0, d0]

    for _ in range(max_iter):
        jacobian, residual = numeric_jacobian(params, points)
        jacobian_t = mat_transpose(jacobian)
        normal = mat_mul(jacobian_t, jacobian)
        for i in range(len(normal)):
            normal[i][i] += 1e-6

        rhs = [-value for value in mat_vec_mul(jacobian_t, residual)]
        step = solve_linear_system(normal, rhs)

        params = [p + s for p, s in zip(params, step)]
        if math.sqrt(sum(s * s for s in step)) < 1e-6:
            break

    normal_vec = plane_normal_from_params(params)
    # Correction : recalculer les residus avec les parametres finaux
    residual = plane_residuals(params, points)
    plane_rmse = math.sqrt(sum(r * r for r in residual) / max(1, len(residual)))
    return normalize(normal_vec), params[2], plane_rmse

def rotation_matrix_from_vectors(a: Vector3, b: Vector3) -> List[List[float]]:
    a_n = normalize(a)
    b_n = normalize(b)
    v = cross(a_n, b_n)
    c = clamp(dot(a_n, b_n), -1.0, 1.0)
    s = norm(v)

    if s < EPS:
        if c > 0.0:
            return [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        axis = normalize(cross(a_n, (1.0, 0.0, 0.0)))
        if norm(axis) < EPS:
            axis = normalize(cross(a_n, (0.0, 1.0, 0.0)))
        v = axis
        s = norm(v)
        c = -1.0

    vx, vy, vz = v
    k = [
        [0.0, -vz, vy],
        [vz, 0.0, -vx],
        [-vy, vx, 0.0],
    ]

    k2 = mat_mul(k, k)
    factor = (1.0 - c) / max(EPS, s * s)

    return [
        [1.0 + k[0][0] + k2[0][0] * factor, k[0][1] + k2[0][1] * factor, k[0][2] + k2[0][2] * factor],
        [k[1][0] + k2[1][0] * factor, 1.0 + k[1][1] + k2[1][1] * factor, k[1][2] + k2[1][2] * factor],
        [k[2][0] + k2[2][0] * factor, k[2][1] + k2[2][1] * factor, 1.0 + k[2][2] + k2[2][2] * factor],
    ]


def rotate_point(matrix: Sequence[Sequence[float]], point: Vector3) -> Vector3:
    x = matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2]
    y = matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2]
    z = matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2]
    return (x, y, z)


def precision_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "Elevee"
    if confidence >= 0.6:
        return "Moyenne"
    return "Faible"


def convert_from_cm(value_cm: float, unit: str) -> float:
    factor = UNIT_FACTORS.get(unit, 1.0)
    return value_cm * factor


def compute_measurements(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload JSON invalide.")

    image_data_url = payload.get("image_data_url")
    image_width = int(float(payload.get("image_width", 0)))
    image_height = int(float(payload.get("image_height", 0)))
    unit = str(payload.get("unit", "cm")).strip().lower()

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Dimensions image invalides.")

    if unit not in UNIT_FACTORS:
        unit = "cm"

    reference_points = parse_points(payload.get("reference_points"), 2, "reference_points")
    try:
        reference_length_cm = float(payload.get("reference_length_cm", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("reference_length_cm invalide.") from exc

    if reference_length_cm <= 0:
        raise ValueError("reference_length_cm invalide.")

    width_points = parse_points(payload.get("width_points"), 2, "width_points")
    height_points = parse_points(payload.get("height_points"), 2, "height_points")

    image_bytes = decode_data_url_image(image_data_url)
    depth_map = get_depth_map(image_bytes)

    width_px = distance(width_points[0], width_points[1])
    height_px = distance(height_points[0], height_points[1])
    reference_px = distance(reference_points[0], reference_points[1])

    if width_px <= EPS or height_px <= EPS:
        raise ValueError("Les segments largeur/hauteur doivent etre non nuls.")

    if reference_px <= EPS:
        raise ValueError("Le segment reference doit etre non nul.")

    def depth_cm_at(point: Point) -> Tuple[float, float]:
        raw_mean, raw_std = sample_depth_stats(depth_map, point, image_width, image_height)
        depth_cm = normalize_depth_to_cm(raw_mean, depth_map)
        return depth_cm, raw_std

    width_depth_a, width_std_a = depth_cm_at(width_points[0])
    width_depth_b, width_std_b = depth_cm_at(width_points[1])
    height_depth_a, height_std_a = depth_cm_at(height_points[0])
    height_depth_b, height_std_b = depth_cm_at(height_points[1])
    reference_depth_a, reference_std_a = depth_cm_at(reference_points[0])
    reference_depth_b, reference_std_b = depth_cm_at(reference_points[1])

    sample_points: List[Vector3] = []
    for point in sample_segment_points(reference_points[0], reference_points[1]):
        depth_cm, _ = depth_cm_at(point)
        sample_points.append(project_pixel_to_3d(point, depth_cm, image_width, image_height))

    for point in sample_segment_points(width_points[0], width_points[1]):
        depth_cm, _ = depth_cm_at(point)
        sample_points.append(project_pixel_to_3d(point, depth_cm, image_width, image_height))

    for point in sample_segment_points(height_points[0], height_points[1]):
        depth_cm, _ = depth_cm_at(point)
        sample_points.append(project_pixel_to_3d(point, depth_cm, image_width, image_height))

    plane_normal, _, plane_rmse = fit_plane_gn(sample_points)
    rotation = rotation_matrix_from_vectors(plane_normal, (0.0, 0.0, 1.0))

    width_point_a = project_pixel_to_3d(width_points[0], width_depth_a, image_width, image_height)
    width_point_b = project_pixel_to_3d(width_points[1], width_depth_b, image_width, image_height)
    height_point_a = project_pixel_to_3d(height_points[0], height_depth_a, image_width, image_height)
    height_point_b = project_pixel_to_3d(height_points[1], height_depth_b, image_width, image_height)
    reference_point_a = project_pixel_to_3d(reference_points[0], reference_depth_a, image_width, image_height)
    reference_point_b = project_pixel_to_3d(reference_points[1], reference_depth_b, image_width, image_height)

    width_point_a = rotate_point(rotation, width_point_a)
    width_point_b = rotate_point(rotation, width_point_b)
    height_point_a = rotate_point(rotation, height_point_a)
    height_point_b = rotate_point(rotation, height_point_b)
    reference_point_a = rotate_point(rotation, reference_point_a)
    reference_point_b = rotate_point(rotation, reference_point_b)

    width_cm = distance3(width_point_a, width_point_b)
    height_cm = distance3(height_point_a, height_point_b)
    reference_cm = distance3(reference_point_a, reference_point_b)

    if reference_cm <= EPS:
        raise ValueError("Reference invalide.")

    scale_factor = reference_length_cm / reference_cm
    width_cm *= scale_factor
    height_cm *= scale_factor
    area_cm2 = width_cm * height_cm

    depth_variation = (width_std_a + width_std_b + height_std_a + height_std_b + reference_std_a + reference_std_b) / 6.0
    mean_depth = (width_depth_a + width_depth_b + height_depth_a + height_depth_b + reference_depth_a + reference_depth_b) / 6.0

    plane_factor = clamp(plane_rmse / max(1.0, mean_depth), 0.0, 1.0)
    confidence = clamp(0.9 - plane_factor * 2.2 - depth_variation * 0.01, 0.2, 0.95)

    width_out = convert_from_cm(width_cm, unit)
    height_out = convert_from_cm(height_cm, unit)
    area_out = area_cm2 * UNIT_FACTORS.get(unit, 1.0) ** 2

    note = "Resultat estime. La reference doit etre nette et sur le meme plan que l'objet."

    return {
        "unit": unit,
        "width": width_out,
        "height": height_out,
        "area": area_out,
        "precision_label": precision_label(confidence),
        "note": note,
    }


class MeasurementHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.path = "/index.html"
            super().do_GET()
            return

        if parsed.path == "/api/ping":
            self.send_json(200, {"status": "ok"})
            return

        if parsed.path == "/api/measure":
            self.send_json(405, {"error": "Utilise POST sur /api/measure."})
            return

        super().do_GET()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/measure":
            self.send_json(404, {"error": "Route introuvable."})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self.send_json(400, {"error": "Corps JSON manquant."})
            return

        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json(400, {"error": "JSON invalide."})
            return

        try:
            result = compute_measurements(payload)
            self.send_json(200, result)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            self.send_json(500, {"error": "Service indisponible. Reessaie plus tard."})

    def send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(host: str = "127.0.0.1", port: int = 5000) -> None:
    handler = partial(MeasurementHandler, directory=str(BASE_DIR))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serveur demarre sur http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()



