# scripts/mesh_export.py
"""
Export FreeCAD bodies to STL files for URDF/MJCF use.

Each body is exported with its Placement stripped, so the mesh vertices
are in the body's local frame. Vertex coordinates are in mm (FreeCAD's
default); the URDF should scale by 0.001 at load time via the mesh
scale attribute.

Typical usage in FreeCAD's Python console:

    from pathlib import Path
    SCRIPTS_PATH = Path('/path/to/scripts')
    MESHES_PATH = Path('/path/to/meshes')

    import sys
    sys.path.insert(0, str(SCRIPTS_PATH))
    import mesh_export

    # Option 1: Export a single body
    mesh_export.export_body('chassis', MESHES_PATH / 'chassis.stl')

    # Option 2: Export all bodies at once
    mesh_export.export_bodies(
        labels=['chassis', 'wheel_left', 'wheel_right'],
        output_dir=MESHES_PATH,
    )

    # Verify mesh export
    print(mesh_export.get_mesh_bounds(MESHES_PATH / 'chassis.stl'))

To reload after editing:
    import importlib
    importlib.reload(mesh_export)
"""

from pathlib import Path
import FreeCAD as App
import Mesh
import MeshPart

# Deviation controls tessellation fineness (in mm).
DEFAULT_LINEAR_DEVIATION  = 0.1   # mm
DEFAULT_ANGULAR_DEVIATION = 0.5   # radians

def export_body(
    label, 
    output_path,
    rpy_degrees=None,
    strip_placement=True,
    linear_deviation=DEFAULT_LINEAR_DEVIATION,
    angular_deviation=DEFAULT_ANGULAR_DEVIATION
):
    """Export a single body to an STL file, with Placement stripped.

    Parameters
    ----------
    label : str
        Body label in the active FreeCAD document.
    output_path : str or pathlib.Path
        Destination .stl file path. Parent directory must exist.
    rpy_degrees : (roll, pitch, yaw) tuple of floats, optional
        Rotation applied to the geometry before export, in degrees.
        Uses URDF's fixed-axis (extrinsic) XYZ convention:
            roll  = rotation about world X
            pitch = rotation about world Y
            yaw   = rotation about world Z
        Applied in order: X first, then Y, then Z.
        If None, no rotation is applied.
        Example: rpy_degrees=(0, 0, -90) remaps FreeCAD's Y-forward,
        X-axle frame to ROS's X-forward, Y-left frame.
    strip_placement : bool
        If True, move the body to the local frame's origin
    linear_deviation : float
        Max deviation (mm) of triangle from true surface. Default 0.1.
    angular_deviation : float
        Max angular deviation (radians) between adjacent facets. Default 0.5.

    Returns
    -------
    dict with keys:
        label       : body label
        path        : output file path (as Path)
        vertices    : number of vertices in the mesh
        triangles   : number of triangles
        size_bytes  : output file size
    """
    output_path = Path(output_path)

    objs = App.ActiveDocument.getObjectsByLabel(label)
    if not objs:
        raise ValueError(
            f"No object with label '{label}' in active document."
        )
    obj = objs[0]

    # Strip Placement: work on a copy with identity Placement so the
    # exported mesh is in the body's local frame.
    shape_local = obj.Shape.copy()
    if strip_placement:
        shape_local.Placement = App.Placement()

    # Apply axis-remap rotation. URDF RPY convention: extrinsic XYZ
    if rpy_degrees is not None:
        roll_deg, pitch_deg, yaw_deg = rpy_degrees
        rotation = _make_rpy_rotation(roll_deg, pitch_deg, yaw_deg)
        placement = App.Placement(App.Vector(0, 0, 0), rotation)
        shape_local.transformShape(placement.toMatrix())

    # Tessellate: convert B-rep to triangle mesh with controlled accuracy.
    mesh = MeshPart.meshFromShape(
        Shape=shape_local,
        LinearDeflection=linear_deviation,
        AngularDeflection=angular_deviation,
        Relative=False,
    )

    # Make sure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write STL (binary format, smaller than ASCII)
    mesh.write(Filename=str(output_path))

    return {
        'label'      : label,
        'path'       : output_path,
        'vertices'   : mesh.CountPoints,
        'triangles'  : mesh.CountFacets,
        'size_bytes' : output_path.stat().st_size,
    }


def export_bodies(labels=None, output_dir=None,
                  label_to_path=None, **kwargs):
    """Export multiple bodies in one call.

    Two options for calling:

           export_bodies(labels=['chassis', 'wheel_left'],
                         output_dir=Path('meshes'))

           export_bodies({'chassis': Path('meshes/chassis.stl'), ...})

    Parameters
    ----------
    labels : iterable of str, optional
        Body labels to export. Used with output_dir.
    output_dir : str or pathlib.Path, optional
        Common destination directory. Used with labels.
    label_to_path : dict, optional
        Mapping of body label → output STL path (str or Path).
        If given, takes precedence over labels/output_dir.
    **kwargs :
        Passed through to export_body (e.g., linear_deviation).

    Returns
    -------
    list of result dicts from export_body, with 'error' key set on failure.
    """
    # Build the label_to_path mapping if not given directly
    if label_to_path is None:
        if labels is None or output_dir is None:
            raise ValueError(
                "Provide either label_to_path={...} or both "
                "labels=[...] and output_dir=Path(...)."
            )
        output_dir = Path(output_dir)
        label_to_path = {label: output_dir / f'{label}.stl'
                         for label in labels}

    results = []
    for label, path in label_to_path.items():
        path = Path(path)
        try:
            r = export_body(label, path, **kwargs)
            results.append(r)
            size_kb = r['size_bytes'] / 1024
            print(f"Exported {r['path']}")
            print(f"    {r['triangles']} triangles, {size_kb:.1f} KB")
        except Exception as e:
            print(f"Error: {label:20s}  FAILED: {e}")
            results.append({'label': label, 'path': path, 'error': str(e)})

    return results

def get_mesh_bounds(stl_path):
    """Return the bounding box of an STL file as a dict.

    Useful for sanity-checking that exported meshes have the expected
    dimensions and are centered on the right point.

    Parameters
    ----------
    stl_path : str or pathlib.Path

    Returns
    -------
    dict with keys: xmin, ymin, zmin, xmax, ymax, zmax,
                    xsize, ysize, zsize  (all in mm)
    """
    stl_path = Path(stl_path)
    m = Mesh.Mesh(str(stl_path))
    bb = m.BoundBox
    return {
        'xmin': bb.XMin, 'ymin': bb.YMin, 'zmin': bb.ZMin,
        'xmax': bb.XMax, 'ymax': bb.YMax, 'zmax': bb.ZMax,
        'xsize': bb.XLength, 'ysize': bb.YLength, 'zsize': bb.ZLength,
    }

def _make_rpy_rotation(roll_deg, pitch_deg, yaw_deg):
    """Build an App.Rotation from URDF-style RPY (extrinsic XYZ, degrees).

    URDF defines rpy as fixed-axis rotations applied in the order:
    roll about world X, then pitch about world Y, then yaw about world Z.

    The resulting orientation is equivalent to the matrix product:
        R = Rz(yaw) * Ry(pitch) * Rx(roll)
    (since extrinsic rotations compose right-to-left when applied to a
    point.)

    We construct this by composing three single-axis rotations, which
    is unambiguous regardless of FreeCAD's internal Euler conventions.
    """
    rx = App.Rotation(App.Vector(1, 0, 0), roll_deg)
    ry = App.Rotation(App.Vector(0, 1, 0), pitch_deg)
    rz = App.Rotation(App.Vector(0, 0, 1), yaw_deg)

    # Extrinsic XYZ means apply X first, then Y, then Z to the body.
    return rz.multiply(ry).multiply(rx)