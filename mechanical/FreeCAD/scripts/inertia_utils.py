# scripts/inertia_utils.py
"""
BodyInertial: Extract URDF/MJCF-ready inertial properties from FreeCAD bodies.

Usage
---
Typical usage in FreeCAD's Python console:

    import sys
    sys.path.insert(0, '/path/to/scripts')
    import inertia_utils

    # Option 1: Use the Shape's material density and FreeCAD's geometric CoM
    chassis = inertia_utils.BodyInertial('chassis')

    # Option 2: Provide measured mass and measured center of mass (CoM)
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

    Parameters
    ----------
    label : str
        The body's label in the active FreeCAD document.
    measured_mass : float, optional
        Empirically measured mass in kg. Overrides material-derived mass.
    measured_com : (x, y, z) tuple, optional
        Empirically measured CoM in meters, in the body's local frame.
        Overrides the geometric centroid.
    """
    def __init__(self, label, measured_mass=None, measured_com=None):
        """Constructor"""
        self._label        = label
        self._measured_com = tuple(measured_com) if measured_com else None

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

        # Volume, mm³ → m³
        self._volume = shape_local.Volume * 1e-9

        # Geometric CoM in local frame, mm → m
        com_mm = shape_local.CenterOfMass
        self._geometric_com = self._clean_com(
            (com_mm.x * 1e-3, com_mm.y * 1e-3, com_mm.z * 1e-3)
        )

        # FreeCAD's inertia tensor: g·mm² at unit density (1 g/mm³),
        # computed ABOUT THE GEOMETRIC CoM, with axes parallel to local frame.
        M = shape_local.MatrixOfInertia
        self._raw_moi_matrix = [
            [M.A11, M.A12, M.A13],
            [M.A21, M.A22, M.A23],
            [M.A31, M.A32, M.A33],
        ]

        # Placement (body origin in parent frame), mm → m
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
        """Read density (kg/m³) from the body's ShapeMaterial, or None."""
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
          1. Scale FreeCAD's raw tensor (g·mm² at unit density, about the
             geometric CoM) to SI units at the chosen mass.
          2. If measured_com was given, shift the tensor from the geometric
             CoM to the measured CoM using the parallel axis theorem.
          3. Also compute the tensor about the local origin (diagnostic).
        """
        # Step 1: pick the claimed CoM
        if self._measured_com is not None:
            self._local_com  = tuple(float(c) for c in self._measured_com)
            self._com_source = 'measured'
        else:
            self._local_com  = self._geometric_com
            self._com_source = 'geometric'

        # Step 2: scale raw tensor to SI at the chosen mass.
        # Unit derivation:
        #   FreeCAD raw: g·mm² at unit density 1 g/mm³
        #   To get kg·m² at real mass, we scale by (real_mass_g / V_mm³)
        #   and convert units (g·mm² → kg·m², factor 1e-9).
        #   Combined: scale = mass_kg * 1e-15 / V_m³
        scale = self._mass * 1e-15 / self._volume
        tensor_at_geom_com = [[v * scale for v in row]
                              for row in self._raw_moi_matrix]

        # Step 3: if the claimed CoM differs from the geometric CoM, shift
        # the tensor to be expressed about the claimed (measured) CoM.
        # The parallel axis theorem in "away from CoM" direction:
        #   I_P = I_com + m · (|d|² I₃ − d dᵀ)   where d = P − CoM
        delta = tuple(self._local_com[i] - self._geometric_com[i]
                      for i in range(3))
        if any(abs(d) > 1e-9 for d in delta):
            tensor_at_claimed_com = self._parallel_axis(
                tensor_at_geom_com, self._mass, delta, direction=+1
            )
        else:
            tensor_at_claimed_com = tensor_at_geom_com

        self._moi_com = self._clean_matrix(tensor_at_claimed_com)

        # Step 4: for diagnostic purposes, also compute I about the local
        # origin. Shift from the claimed CoM to the origin.
        self._moi_origin = self._parallel_axis(
            self._moi_com, self._mass, self._local_com, direction=+1
        )
        self._moi_origin = self._clean_matrix(self._moi_origin)

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

        # m · (|d|² I₃ − d dᵀ)
        shift = [
            [mass * (d2 - dx*dx), mass * (-dx*dy),      mass * (-dx*dz)     ],
            [mass * (-dy*dx),     mass * (d2 - dy*dy),  mass * (-dy*dz)     ],
            [mass * (-dz*dx),     mass * (-dz*dy),      mass * (d2 - dz*dz) ],
        ]
        return [[I_ref[i][j] + direction * shift[i][j] for j in range(3)]
                for i in range(3)]

    # --------------------------------------------------------------------------
    # Math helpers

    @staticmethod
    def _clean_com(com, tol=1e-9):
        """Zero out sub-nanometer noise."""
        return tuple(0.0 if abs(c) < tol else c for c in com)

    @staticmethod
    def _clean_matrix(M, tol=1e-12):
        """Zero out sub-picokgm² noise."""
        def clean(x):
            return 0.0 if abs(x) < tol else x
        return [[clean(v) for v in row] for row in M]

    @staticmethod
    def _print_matrix(M, indent=""):
        for row in M:
            print(f"{indent}[{row[0]:12.9f}  {row[1]:12.9f}  {row[2]:12.9f}]")

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
        """Volume in m³ (pure geometry)."""
        return self._volume

    def get_density(self):
        """Density in kg/m³, or None if on the measured-mass path."""
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
        """Inertia tensor about the CoM, kg·m². What URDF/MJCF wants."""
        return [row[:] for row in self._moi_com]

    def get_origin_moi(self):
        """Inertia tensor about the body's local origin, kg·m²."""
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
        print(f"  volume     = {self._volume*1e9:.2f} mm³  "
              f"({self._volume:.9f} m³)")

        if self._density is not None:
            print(f"  density    = {self._density:.1f} kg/m³  "
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
                      f"{geom_mass*1000:.2f} g at {d_kgm3:.0f} kg/m³; "
                      f"ratio measured/material = {ratio:.3f})")

        lcx, lcy, lcz = self._local_com
        print(f"  local CoM  = ({lcx:.6f}, {lcy:.6f}, {lcz:.6f}) m  "
              f"[{self._com_source}]")
        if self._com_source == 'measured':
            gx, gy, gz = self._geometric_com
            delta = [self._local_com[i] - self._geometric_com[i]
                     for i in range(3)]
            dist_mm = math.sqrt(sum(d*d for d in delta)) * 1000
            print(f"               (geometric would be "
                  f"({gx:.6f}, {gy:.6f}, {gz:.6f}) m; "
                  f"measured is {dist_mm:.2f}mm away)")

        gcx, gcy, gcz = self.get_global_com()
        print(f"  global CoM = ({gcx:.6f}, {gcy:.6f}, {gcz:.6f}) m  "
              f"(local + placement)")

        px, py, pz = self._placement
        print(f"  placement  = ({px:.6f}, {py:.6f}, {pz:.6f}) m  "
              f"(body origin in parent frame)")

        print(f"  MoI about CoM (kg·m²):")
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