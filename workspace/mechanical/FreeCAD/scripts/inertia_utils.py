# scripts/inertia_utils.py
"""
BodyInertial: Extract URDF/MJCF-ready inertial properties from FreeCAD bodies.

Usage
---
Typical usage in FreeCAD's Python console:

    from pathlib import Path
    SCRIPTS_PATH = Path('/path/to/scripts')

    import sys
    sys.path.insert(0, str(SCRIPTS_PATH))
    import inertia_utils

    # Option 1: Use the Shape's material density and FreeCAD's geometric CoM
    chassis = inertia_utils.BodyInertial('chassis')

    # Option 2: Provide measured mass (kg) and measured center of mass (m)
    chassis = inertia_utils.BodyInertial('chassis',
                                         measured_mass=0.125,
                                         measured_com=(0.0, 0.0, 0.018))

    # Print results
    chassis.summary()
    print(chassis.to_urdf_inertial())
    print(chassis.to_mjcf_inertial())

To reload after editing:
    import importlib
    importlib.reload(inertia_utils)

Notes
---
FreeCAD always provides:
  - geometric centroid (CoM under uniform-density assumption)
  - inertia tensor about that geometric centroid (also under uniform density)

The user can override two things:
  - measured_mass: rescales the tensor proportionally
  - measured_com:  changes the point the tensor is expressed about
                   (requires a parallel-axis shift from geometric to measured)

The uniform-density assumption can fail to match reality (a dense battery at
the bottom will shift the real CoM downward, for example). Providing a
measured CoM addresses the biggest source of sim-to-real error.
"""

import math
import FreeCAD as App


class BodyInertial:
    """Inertial properties of a FreeCAD body for URDF/MJCF export.

    Output frame:
     * If rpy_degrees is provided, all output values (CoM, inertia tensor,
        placement) are returned in a frame rotated from FreeCAD's by the
        given Euler angles. This should match the rotation applied to the
        exported mesh (mesh_export.export_body's rpy_degrees) so the URDF
        sees consistent geometry and inertia.
     * The measured_com argument, if given, is interpreted in FreeCAD's
        original frame (before rotation), since that's the frame in which
        the user typically measures and creates a model.

    Parameters
    ----------
    label : str
        The body's label in the active FreeCAD document.
    measured_mass : float, optional
        Empirically measured mass in kg. Overrides material-derived mass.
    measured_com : (x, y, z) tuple, optional
        Empirically measured CoM in meters, in the body's local frame (i.e. from
        the local body's origin point in FreeCAD). Overrides the geometric 
        centroid.
    rpy_degrees : (roll, pitch, yaw) tuple, optional
        Rotation applied to all output values to remap from FreeCAD's frame
        to a target frame. Uses URDF's extrinsic XYZ convention, in degrees.
        Must match the rotation passed to mesh_export.export_body to keep
        meshes and inertia consistent.
        Example: rpy_degrees=(0, 0, -90) remaps FreeCAD (X=axle, Y=fwd) to
        ROS (X=fwd, Y=left).
    """
    def __init__(
        self, 
        label, 
        measured_mass=None, 
        measured_com=None, 
        rpy_degrees=None
    ):
        """Constructor"""
        self._label        = label
        self._measured_com = tuple(measured_com) if measured_com else None
        self._rpy_degrees  = tuple(rpy_degrees) if rpy_degrees else None

        # Build a 3x3 rotation matrix for output transforms (or None)
        if self._rpy_degrees is not None:
            self._R = self._build_rotation_matrix(*self._rpy_degrees)
        else:
            self._R = None

        self._extract_from_freecad()
        self._resolve_mass(measured_mass)
        self._compute()

    def _extract_from_freecad(self):
        """Pull raw geometric quantities from FreeCAD (pre-override state)."""
        objs = App.ActiveDocument.getObjectsByLabel(self._label)
        if not objs:
            raise ValueError(
                f"No object with label '{self._label}' in active document."
            )
        self._obj = objs[0]

        # Work in the body's local frame (strip Placement)
        shape_local = self._obj.Shape.copy()
        shape_local.Placement = App.Placement()

        # Volume, mm^3 to m^3
        self._volume = shape_local.Volume * 1e-9

        # Geometric CoM in local frame, mm to m
        com_mm = shape_local.CenterOfMass
        self._geometric_com = self._clean_com(
            (com_mm.x * 1e-3, com_mm.y * 1e-3, com_mm.z * 1e-3)
        )

        # FreeCAD's inertia tensor: g*mm^2 at unit density (1 g/mm^3),
        # computed ABOUT THE GEOMETRIC CoM, with axes parallel to local frame.
        M = shape_local.MatrixOfInertia
        self._raw_moi_matrix = [
            [M.A11, M.A12, M.A13],
            [M.A21, M.A22, M.A23],
            [M.A31, M.A32, M.A33],
        ]

        # Placement (body origin in parent frame), mm to m
        p = self._obj.Placement.Base
        self._placement = (p.x * 1e-3, p.y * 1e-3, p.z * 1e-3)

    def _resolve_mass(self, measured_mass):
        """Choose mass source: measurement or material density × volume."""
        if measured_mass is not None:
            self._mass           = float(measured_mass)
            self._mass_source    = 'measured'
            self._density        = None
            self._density_source = None
            return

        detected = self._detect_density(self._obj)
        if detected is None:
            raise ValueError(
                f"Cannot determine mass for '{self._label}'. Either pass "
                f"measured_mass= (in kg), or assign a material with a "
                f"Density property to the body in FreeCAD."
            )
        self._density, self._density_source = detected
        self._mass        = self._volume * self._density
        self._mass_source = f'density × volume ({self._density_source})'

    @staticmethod
    def _detect_density(obj):
        """Read density (kg/m^3) from the body's ShapeMaterial, or None."""
        if not hasattr(obj, 'ShapeMaterial'):
            return None
        mat = obj.ShapeMaterial
        if mat is None:
            return None

        try:
            if not mat.hasPhysicalProperty('Density'):
                return None
        except Exception:
            pass  # older/newer API — try getPhysicalValue anyway

        try:
            d = mat.getPhysicalValue('Density')
        except Exception:
            return None
        if d is None:
            return None

        try:
            kgm3 = float(d.getValueAs('kg/m^3'))
        except AttributeError:
            try:
                kgm3 = float(d)
            except (TypeError, ValueError):
                return None

        if kgm3 <= 0:
            return None
        return (kgm3, f'ShapeMaterial "{mat.Name}"')

    def _compute(self):
        """Build final inertia tensor consistent with the chosen mass & CoM.

        Pipeline:
          1. Scale FreeCAD's raw tensor (g*mm^2 at unit density, about the
             geometric CoM) to SI units at the chosen mass.
          2. If measured_com was given, shift the tensor from the geometric
             CoM to the measured CoM using the parallel axis theorem.
          3. Also compute the tensor about the local origin (diagnostic).
        """
        # Pick the claimed CoM
        if self._measured_com is not None:
            self._local_com  = tuple(float(c) for c in self._measured_com)
            self._com_source = 'measured'
        else:
            self._local_com  = self._geometric_com
            self._com_source = 'geometric'

        # Scale raw tensor to SI at the chosen mass.
        # Unit derivation:
        #   FreeCAD raw: g*mm^2 at unit density 1 g/mm^3
        #   To get kg*m^2 at real mass, we scale by (real_mass_g / V_mm^3)
        #   and convert units (g*mm^2 to kg*m^2, factor 1e-9).
        #   Combined: scale = mass_kg * 1e-15 / V_m^3
        scale = self._mass * 1e-15 / self._volume
        tensor_at_geom_com = [[v * scale for v in row]
                              for row in self._raw_moi_matrix]

        # If the claimed CoM differs from the geometric CoM, shift
        # the tensor to be expressed about the claimed (measured) CoM.
        # The parallel axis theorem in "away from CoM" direction:
        #   I_P = I_com + m * (|d|^2 I_3 − d dᵀ)   where d = P − CoM
        delta = tuple(self._local_com[i] - self._geometric_com[i]
                      for i in range(3))
        if any(abs(d) > 1e-9 for d in delta):
            tensor_at_claimed_com = self._parallel_axis(
                tensor_at_geom_com, self._mass, delta, direction=+1
            )
        else:
            tensor_at_claimed_com = tensor_at_geom_com

        self._moi_com = self._clean_matrix(tensor_at_claimed_com)

        # For diagnostic purposes, also compute I about the local
        # origin. Shift from the claimed CoM to the origin.
        self._moi_origin = self._parallel_axis(
            self._moi_com, self._mass, self._local_com, direction=+1
        )
        self._moi_origin = self._clean_matrix(self._moi_origin)

        # Apply output frame rotation if requested. geometric_com is not rotated
        # as it is used for diagnostics (in FreeCAD frame of reference)
        if self._R is not None:
            self._local_com  = self._rotate_vector(self._R, self._local_com)
            self._placement  = self._rotate_vector(self._R, self._placement)
            self._moi_com    = self._rotate_tensor(self._R, self._moi_com)
            self._moi_origin = self._rotate_tensor(self._R, self._moi_origin)
            
            # Re-clean noise introduced by the rotation arithmetic
            self._local_com  = self._clean_com(self._local_com)
            self._placement  = self._clean_com(self._placement)
            self._moi_com    = self._clean_matrix(self._moi_com)
            self._moi_origin = self._clean_matrix(self._moi_origin)

    # --------------------------------------------------------------------------
    # Math helpers

    @staticmethod
    def _parallel_axis(I_ref, mass, d, direction=+1):
        """Parallel axis theorem, unified for both directions.

        direction = +1  (away from CoM, inertia increases):
            Input:  I about the CoM
            d:      vector from CoM to the target point P
            Output: I about P
        direction = −1  (toward CoM, inertia decreases):
            Input:  I about some point P
            d:      vector from CoM to P
            Output: I about the CoM
        """
        dx, dy, dz = d
        d2 = dx*dx + dy*dy + dz*dz

        # m * (|d|^2 I_3 − d dᵀ)
        shift = [
            [mass * (d2 - dx*dx), mass * (-dx*dy),      mass * (-dx*dz)     ],
            [mass * (-dy*dx),     mass * (d2 - dy*dy),  mass * (-dy*dz)     ],
            [mass * (-dz*dx),     mass * (-dz*dy),      mass * (d2 - dz*dz) ],
        ]
        return [[I_ref[i][j] + direction * shift[i][j] for j in range(3)]
                for i in range(3)]

    @staticmethod
    def _clean_com(com, tol=1e-9):
        """Zero out sub-nanometer noise."""
        return tuple(0.0 if abs(c) < tol else c for c in com)

    @staticmethod
    def _clean_matrix(M, tol=1e-12):
        """Zero out sub-picokgm^2 noise."""
        def clean(x):
            return 0.0 if abs(x) < tol else x
        return [[clean(v) for v in row] for row in M]

    @staticmethod
    def _print_matrix(M, indent=""):
        for row in M:
            print(f"{indent}[{row[0]:12.9f}  {row[1]:12.9f}  {row[2]:12.9f}]")

    @staticmethod
    def _build_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
        """Build a 3x3 rotation matrix from URDF-style RPY (extrinsic XYZ).

        URDF convention: rotate about world X by roll, then world Y by pitch,
        then world Z by yaw. As a matrix product applied to a point:
            R = Rz * Ry * Rx
        so the final transform on a point p is Rz(Ry(Rx(p))).
        """
        cr = math.cos(math.radians(roll_deg))
        sr = math.sin(math.radians(roll_deg))
        cp = math.cos(math.radians(pitch_deg))
        sp = math.sin(math.radians(pitch_deg))
        cy = math.cos(math.radians(yaw_deg))
        sy = math.sin(math.radians(yaw_deg))

        Rx = [[1,  0,   0 ],
            [0,  cr, -sr],
            [0,  sr,  cr]]
        Ry = [[ cp, 0, sp],
            [ 0,  1, 0 ],
            [-sp, 0, cp]]
        Rz = [[cy, -sy, 0],
            [sy,  cy, 0],
            [0,   0,  1]]

        return BodyInertial._matmul(Rz, BodyInertial._matmul(Ry, Rx))

    @staticmethod
    def _matmul(A, B):
        """Multiply two 3x3 matrices (as list-of-lists)."""
        return [[sum(A[i][k] * B[k][j] for k in range(3))
                for j in range(3)]
                for i in range(3)]

    @staticmethod
    def _rotate_vector(R, v):
        """Apply rotation R to 3-vector v: returns R * v as a tuple."""
        return tuple(sum(R[i][k] * v[k] for k in range(3)) for i in range(3))

    @staticmethod
    def _rotate_tensor(R, T):
        """Transform a 3x3 second-rank tensor under rotation R: R * T * Rᵀ.

        This is the standard tensor transformation rule. Vectors get one R;
        tensors get an R on each side because they have two "directional"
        indices that both need to transform.
        """
        Rt = [[R[j][i] for j in range(3)] for i in range(3)]
        return BodyInertial._matmul(BodyInertial._matmul(R, T), Rt)

    def __repr__(self):
        return (f"BodyInertial(label='{self._label}', "
                f"mass={self._mass:.4f}kg [{self._mass_source}], "
                f"com={self._local_com} [{self._com_source}])")

    # --------------------------------------------------------------------------
    # Public methods

    def get_label(self):
        """Body label in the FreeCAD document."""
        return self._label

    def get_volume(self):
        """Volume in m^3 (pure geometry)."""
        return self._volume

    def get_density(self):
        """Density in kg/m^3, or None if on the measured-mass path."""
        return self._density

    def get_mass(self):
        """Final mass in kg (measured or material-derived)."""
        return self._mass

    def get_local_com(self):
        """Final CoM in the body's local frame, (x,y,z) in meters."""
        return self._local_com

    def get_geometric_com(self):
        """Geometric centroid from FreeCAD, ignoring any override."""
        return self._geometric_com

    def get_global_com(self):
        """CoM in parent frame (local CoM + Placement)."""
        return tuple(lc + p for lc, p in
                     zip(self._local_com, self._placement))

    def get_placement(self):
        """Body origin in parent frame, (x,y,z) in meters."""
        return self._placement

    def get_com_moi(self):
        """Inertia tensor about the CoM, kg*m^2. What URDF/MJCF wants."""
        return [row[:] for row in self._moi_com]

    def get_origin_moi(self):
        """Inertia tensor about the body's local origin, kg*m^2."""
        return [row[:] for row in self._moi_origin]

    def get_mass_source(self):
        return self._mass_source

    def get_com_source(self):
        return self._com_source

    def get_density_source(self):
        return self._density_source

    def get_radii_of_gyration(self):
        """Radii of gyration (kx, ky, kz) in meters. Sanity-check lengths."""
        m = self._mass
        if m <= 0:
            return (0.0, 0.0, 0.0)
        I = self._moi_com
        return tuple(math.sqrt(abs(I[i][i]) / m) for i in range(3))

    def summary(self):
        """Print a human-readable summary of all properties."""
        print(f"=== BodyInertial: {self._label} ===")
        if self._rpy_degrees is not None:
            print(f"  output frame: rotated by RPY (deg) = {self._rpy_degrees}")
        print(f"  volume     = {self._volume*1e9:.2f} mm^3  "
              f"({self._volume:.9f} m^3)")

        if self._density is not None:
            print(f"  density    = {self._density:.1f} kg/m^3  "
                  f"[{self._density_source}]")

        print(f"  mass       = {self._mass:.6f} kg "
              f"({self._mass*1000:.2f} g)  [{self._mass_source}]")

        # If measured_mass was used but a material exists, show the comparison
        if (self._mass_source == 'measured'
                and hasattr(self._obj, 'ShapeMaterial')
                and self._obj.ShapeMaterial is not None):
            detected = self._detect_density(self._obj)
            if detected is not None:
                d_kgm3, d_src = detected
                geom_mass = self._volume * d_kgm3
                ratio = self._mass / geom_mass if geom_mass > 0 else 0
                print(f"               ({d_src} would give "
                      f"{geom_mass*1000:.2f} g at {d_kgm3:.0f} kg/m^3; "
                      f"ratio measured/material = {ratio:.3f})")

        # Determine frame labels for clarity
        if self._rpy_degrees is not None:
            output_frame  = "rotated frame"
            freecad_frame = "FreeCAD frame"
        else:
            output_frame = freecad_frame = "FreeCAD frame"

        lcx, lcy, lcz = self._local_com
        print(f"  local CoM  = ({lcx:.6f}, {lcy:.6f}, {lcz:.6f}) m  "
            f"[{self._com_source}, in {output_frame}]")
        if self._com_source == 'measured':
            gx, gy, gz = self._geometric_com
            # Note: the parallel-axis distance is computed in FreeCAD frame because
            # _geometric_com is in FreeCAD frame. For Z-axis rotations this gives
            # the same magnitude as in the rotated frame; for general rotations
            # the magnitude is preserved (rotations don't change distances).
            delta = [self._measured_com[i] - self._geometric_com[i]
                    for i in range(3)]
            dist_mm = math.sqrt(sum(d*d for d in delta)) * 1000
            print(f"               (geometric would be "
                f"({gx:.6f}, {gy:.6f}, {gz:.6f}) m, in {freecad_frame}; "
                f"measured is {dist_mm:.2f}mm away)")

        gcx, gcy, gcz = self.get_global_com()
        print(f"  global CoM = ({gcx:.6f}, {gcy:.6f}, {gcz:.6f}) m  "
              f"(local + placement)")

        px, py, pz = self._placement
        print(f"  placement  = ({px:.6f}, {py:.6f}, {pz:.6f}) m  "
            f"(body origin in parent frame, in {output_frame})")

        print(f"  MoI about CoM (kg*m^2, in {output_frame}):")
        self._print_matrix(self._moi_com, indent="    ")

        kx, ky, kz = self.get_radii_of_gyration()
        print(f"  radii of gyration: "
              f"kx={kx*1000:.2f} mm, ky={ky*1000:.2f} mm, kz={kz*1000:.2f} mm")
        print()

    def to_urdf_inertial(self, indent="  "):
        """Format as a URDF <inertial> XML block."""
        I = self._moi_com
        ixx, iyy, izz = I[0][0], I[1][1], I[2][2]
        ixy, ixz, iyz = I[0][1], I[0][2], I[1][2]
        cx, cy, cz = self._local_com
        return "\n".join([
            f'{indent}<inertial>',
            f'{indent}  <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" '
            f'rpy="0 0 0"/>',
            f'{indent}  <mass value="{self._mass:.6f}"/>',
            f'{indent}  <inertia ixx="{ixx:.9f}" iyy="{iyy:.9f}" '
            f'izz="{izz:.9f}"',
            f'{indent}           ixy="{ixy:.9f}" ixz="{ixz:.9f}" '
            f'iyz="{iyz:.9f}"/>',
            f'{indent}</inertial>',
        ])

    def to_mjcf_inertial(self, indent="  "):
        """Format as an MJCF <inertial/> element."""
        I = self._moi_com
        ixx, iyy, izz = I[0][0], I[1][1], I[2][2]
        ixy, ixz, iyz = I[0][1], I[0][2], I[1][2]
        cx, cy, cz = self._local_com

        diagonal = (abs(ixy) < 1e-12
                    and abs(ixz) < 1e-12
                    and abs(iyz) < 1e-12)
        if diagonal:
            return (f'{indent}<inertial '
                    f'pos="{cx:.6f} {cy:.6f} {cz:.6f}" '
                    f'mass="{self._mass:.6f}" '
                    f'diaginertia="{ixx:.9f} {iyy:.9f} {izz:.9f}"/>')
        else:
            return (f'{indent}<inertial '
                    f'pos="{cx:.6f} {cy:.6f} {cz:.6f}" '
                    f'mass="{self._mass:.6f}" '
                    f'fullinertia="{ixx:.9f} {iyy:.9f} {izz:.9f} '
                    f'{ixy:.9f} {ixz:.9f} {iyz:.9f}"/>')