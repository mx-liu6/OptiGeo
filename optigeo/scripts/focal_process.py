import os
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import shutil
from typing import Dict, Tuple

import click
import numpy as np
import utils3d
from tqdm import tqdm

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

from optigeo.utils.io import read_depth, read_image, read_meta, write_depth, write_image, write_meta
from optigeo.utils.geometry_numpy import depth_occlusion_edge_numpy


def build_target_intrinsics(width: int, height: int, target_focal_px: float) -> np.ndarray:
    return np.array(
        [
            [target_focal_px / width, 0.0, 0.5],
            [0.0, target_focal_px / height, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def sample_matches_target_focal(intrinsics: np.ndarray, width: int, height: int, target_focal_px: float, atol: float = 1e-3) -> bool:
    fx_px = float(intrinsics[0, 0]) * width
    fy_px = float(intrinsics[1, 1]) * height
    return abs(fx_px - target_focal_px) <= atol and abs(fy_px - target_focal_px) <= atol


def _require_cv2():
    if cv2 is None:
        raise ModuleNotFoundError('cv2 is required for focal processing; please install opencv-python or opencv-python-headless.')


def remap_to_target_focal(
    image: np.ndarray,
    depth: np.ndarray,
    depth_mask: np.ndarray,
    depth_mask_inf: np.ndarray,
    src_intrinsics: np.ndarray,
    tgt_intrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproject an RGB/depth pair from `src_intrinsics` to `tgt_intrinsics`
    under the same camera pose and image resolution.

    Returns:
    - remapped_image: uint8 RGB image of shape (H, W, 3)
    - remapped_depth: float32 depth map of shape (H, W), with NaN/Inf preserved
    """
    height, width = depth.shape

    transform = src_intrinsics @ np.linalg.inv(tgt_intrinsics)
    uv_tgt = utils3d.numpy.image_uv(width=width, height=height)
    pts = np.concatenate([uv_tgt, np.ones((height, width, 1), dtype=np.float32)], axis=-1) @ transform.T
    uv_src = pts[..., :2] / (pts[..., 2:3] + 1e-12)
    pixel_src = utils3d.numpy.uv_to_pixel(uv_src, width=width, height=height).astype(np.float32)

    map_x = pixel_src[..., 0]
    map_y = pixel_src[..., 1]

    remapped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    edge_mask = depth_occlusion_edge_numpy(depth, mask=depth_mask, thickness=2, tol=0.01)
    finite_depth = np.where(depth_mask, depth, 1.0).astype(np.float32)
    inv_depth = np.where(depth_mask, 1.0 / np.maximum(finite_depth, 1e-12), 0.0).astype(np.float32)

    remap_valid_nearest = cv2.remap(depth_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    remap_inf = cv2.remap(depth_mask_inf.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    remap_edge = cv2.remap(edge_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0

    remap_valid_linear = cv2.remap(depth_mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    remap_depth_nearest = cv2.remap(finite_depth, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
    remap_inv_depth_linear = cv2.remap(inv_depth, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    remap_depth_linear = np.where(remap_inv_depth_linear > 1e-12, 1.0 / remap_inv_depth_linear, 1.0).astype(np.float32)

    use_linear = (remap_valid_linear > 0.999) & (~remap_edge)
    remapped_depth = np.where(use_linear, remap_depth_linear, remap_depth_nearest).astype(np.float32)
    remapped_depth[~remap_valid_nearest] = np.nan
    remapped_depth[remap_inf] = np.inf
    return remapped_image, remapped_depth


def process_one_sample(
    src_root: Path,
    dst_root: Path,
    rel_path: str,
    target_focal_px: float,
    overwrite: bool,
) -> str:
    _require_cv2()
    src_dir = src_root / rel_path
    dst_dir = dst_root / rel_path

    image_path = src_dir / 'image.jpg'
    depth_path = src_dir / 'depth.png'
    meta_path = src_dir / 'meta.json'

    if not image_path.exists() or not depth_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f'Missing required file under {src_dir}')

    if dst_dir.exists() and not overwrite and all((dst_dir / name).exists() for name in ['image.jpg', 'depth.png', 'meta.json']):
        return 'skipped_existing'

    meta = read_meta(meta_path)
    intrinsics = np.array(meta['intrinsics'], dtype=np.float32)
    image = read_image(image_path)
    depth, depth_unit = read_depth(depth_path)
    height, width = depth.shape

    target_intrinsics = build_target_intrinsics(width, height, target_focal_px)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if sample_matches_target_focal(intrinsics, width, height, target_focal_px):
        shutil.copy2(image_path, dst_dir / 'image.jpg')
        shutil.copy2(depth_path, dst_dir / 'depth.png')
        meta['intrinsics'] = target_intrinsics.tolist()
        write_meta(dst_dir / 'meta.json', meta)
        return 'copied'

    depth_mask = np.isfinite(depth)
    depth_mask_inf = np.isinf(depth)
    remapped_image, remapped_depth = remap_to_target_focal(
        image=image,
        depth=depth,
        depth_mask=depth_mask,
        depth_mask_inf=depth_mask_inf,
        src_intrinsics=intrinsics,
        tgt_intrinsics=target_intrinsics,
    )

    write_image(dst_dir / 'image.jpg', remapped_image)
    write_depth(dst_dir / 'depth.png', remapped_depth, unit=depth_unit)
    meta['intrinsics'] = target_intrinsics.tolist()
    write_meta(dst_dir / 'meta.json', meta)
    return 'processed'


@click.command()
@click.option('--src_root', type=click.Path(path_type=Path), default='data/train/OptiGeo', show_default=True, help='Source dataset root.')
@click.option('--index_path', type=click.Path(path_type=Path), default='data/train/OptiGeo/index_metric.txt', show_default=True, help='Index file listing relative sample directories.')
@click.option('--dst_root', type=click.Path(path_type=Path), default='data/train/OptiGeo_fixfocal', show_default=True, help='Output dataset root.')
@click.option('--target_focal_px', type=float, default=1320.0, show_default=True, help='Target focal length in pixel units.')
@click.option('--overwrite', is_flag=True, help='Overwrite existing outputs.')
def main(
    src_root: Path,
    index_path: Path,
    dst_root: Path,
    target_focal_px: float,
    overwrite: bool,
):
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    index_path = index_path.resolve()

    rel_paths = [line.strip() for line in index_path.read_text().splitlines() if line.strip()]
    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / index_path.name).write_text('\n'.join(rel_paths) + '\n')

    counts: Dict[str, int] = {
        'processed': 0,
        'copied': 0,
        'skipped_existing': 0,
        'failed': 0,
    }

    for rel_path in tqdm(rel_paths, desc=f'Fix focal -> {target_focal_px:g}px'):
        try:
            status = process_one_sample(
                src_root=src_root,
                dst_root=dst_root,
                rel_path=rel_path,
                target_focal_px=target_focal_px,
                overwrite=overwrite,
            )
            counts[status] += 1
        except Exception as exc:
            counts['failed'] += 1
            print(f'[ERROR] {rel_path}: {exc}', flush=True)

    print('Done.')
    for key, value in counts.items():
        print(f'{key}: {value}')
    print(f'output_index: {dst_root / index_path.name}')


if __name__ == '__main__':
    main()
