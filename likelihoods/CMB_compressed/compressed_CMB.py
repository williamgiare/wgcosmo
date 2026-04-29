from pathlib import Path

import numpy as np
from cobaya.likelihood import Likelihood


class CompressedCMB(Likelihood):
    """Compressed Planck CMB Gaussian likelihood.

    Main interface:

        compression_basis: [omega_b, omega_m, theta_s]
        compression_basis: [omega_b, omega_m, 1/theta_s]
        compression_basis: [omega_b, omega_m, 1/theta_drag]

    2D sub-compressions are obtained by selecting a sub-block of the matching
    3x3 data vector/covariance, e.g.

        compression_basis: [omega_b, theta_s]
    """

    # Do not let Cobaya assign sampled parameters exclusively to this likelihood.
    # CLASS must still receive theta_s_100, omega_b, omega_cdm, etc.
    input_params = []

    compression_basis = None

    # Backward compatibility with the old interface. Ignored if compression_basis
    # is set explicitly.
    direction = None

    data_file = None
    data_dir = "compressed_data_vectors"
    debug = False

    z_drag_min = 1000.0
    z_drag_max = 1100.0
    z_drag_n = 4001

    _THETA_FILE = {
        "theta_s": "wb_wm_Thetas.dat",
        "1/theta_s": "wb_wm_invThetas.dat",
        "theta_drag": "wb_wm_Thetadrag.dat",
        "1/theta_drag": "wb_wm_invThetadrag.dat",
    }

    _PARAM_ALIASES = {
        "omega_b": "omega_b",
        "wb": "omega_b",
        "omegab": "omega_b",
        "omega_m": "omega_m",
        "wm": "omega_m",
        "omegam": "omega_m",
        "omegamh2": "omega_m",
        "theta_s": "theta_s",
        "thetas": "theta_s",
        "theta_star": "theta_s",
        "1/theta_s": "1/theta_s",
        "inv_theta_s": "1/theta_s",
        "inv_thetas": "1/theta_s",
        "theta_drag": "theta_drag",
        "thetadrag": "theta_drag",
        "theta_d": "theta_drag",
        "1/theta_drag": "1/theta_drag",
        "inv_theta_drag": "1/theta_drag",
        "inv_thetadrag": "1/theta_drag",
        "1/theta_d": "1/theta_drag",
        "inv_theta_d": "1/theta_drag",
    }

    _DIRECTION_TO_BASIS = {
        "theta_s": ["omega_b", "omega_m", "theta_s"],
        "1/theta_s": ["omega_b", "omega_m", "1/theta_s"],
        "theta_drag": ["omega_b", "omega_m", "theta_drag"],
        "1/theta_drag": ["omega_b", "omega_m", "1/theta_drag"],
        "inv_theta_s": ["omega_b", "omega_m", "1/theta_s"],
        "inv_theta_drag": ["omega_b", "omega_m", "1/theta_drag"],
    }

    def initialize(self):
        self.compression_basis = self._resolve_compression_basis()
        self.theta_param = self._get_theta_param(self.compression_basis)
        self.data_path = self._resolve_data_path()

        full_params, full_data, full_cov = self._read_data_file(self.data_path)
        self.full_data_params = full_params
        self.full_data = full_data
        self.full_cov = full_cov

        self._validate_full_data()
        self._select_basis_from_full_data()
        self._validate_selected_data()
        self.inv_cov = np.linalg.inv(self.cov)

        if self._uses_theta_drag():
            self.z_bg = np.linspace(
                float(self.z_drag_min),
                float(self.z_drag_max),
                int(self.z_drag_n),
            )

    def get_requirements(self):
        requirements = {
            "omega_b": None,
            "Omega_m": None,
            "H0": None,
        }

        if self._uses_theta_s():
            requirements["theta_s_100"] = None

        if self._uses_theta_drag():
            requirements.update({
                "z_d": None,
                "rs_drag": None,
                "angular_diameter_distance": {"z": self.z_bg},
            })

        return requirements

    @classmethod
    def _normalize_param(cls, value):
        key = str(value).strip().lower().replace(" ", "")
        key = key.replace("1/thetas", "1/theta_s")
        key = key.replace("1/thetadrag", "1/theta_drag")
        if key not in cls._PARAM_ALIASES:
            known = ", ".join(sorted(cls._PARAM_ALIASES))
            raise ValueError(f"Unknown compression-basis parameter '{value}'. Known aliases: {known}")
        return cls._PARAM_ALIASES[key]

    def _resolve_compression_basis(self):
        if self.compression_basis is not None:
            if isinstance(self.compression_basis, str):
                raw_basis = self.compression_basis.replace(",", " ").split()
            else:
                raw_basis = list(self.compression_basis)
            basis = [self._normalize_param(p) for p in raw_basis]
        else:
            direction = self.direction or "theta_s"
            direction_key = str(direction).strip().lower().replace(" ", "")
            direction_key = direction_key.replace("1/thetas", "1/theta_s")
            direction_key = direction_key.replace("1/thetadrag", "1/theta_drag")
            if direction_key not in self._DIRECTION_TO_BASIS:
                direction_key = self._normalize_param(direction_key)
            basis = list(self._DIRECTION_TO_BASIS[direction_key])

        if len(basis) not in (2, 3):
            raise ValueError(f"compression_basis must have length 2 or 3; got {basis}")
        if len(set(basis)) != len(basis):
            raise ValueError(f"compression_basis contains duplicate parameters: {basis}")
        return basis

    @staticmethod
    def _get_theta_param(basis):
        theta_params = [p for p in basis if p in {"theta_s", "1/theta_s", "theta_drag", "1/theta_drag"}]
        if len(theta_params) != 1:
            raise ValueError(
                "compression_basis must contain exactly one theta-like parameter "
                "among theta_s, 1/theta_s, theta_drag, 1/theta_drag; "
                f"got {basis}"
            )
        return theta_params[0]

    def _resolve_data_path(self):
        if self.data_file:
            candidates = [Path(self.data_file)]
        else:
            candidates = [Path(self.data_dir) / self._THETA_FILE[self.theta_param]]

        resolved = []
        here = Path(__file__).resolve().parent
        for path in candidates:
            if path.is_absolute():
                resolved.append(path)
            else:
                resolved.append(Path.cwd() / path)
                resolved.append(here / path)

        for path in resolved:
            if path.exists():
                return path.resolve()

        tried = "\n".join(f"  - {p}" for p in resolved)
        raise FileNotFoundError(f"Could not find compressed CMB data file. Tried:\n{tried}")

    @staticmethod
    def _read_data_file(path):
        params = None
        data = None
        cov_rows = []
        section = None

        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    text = line[1:].strip()
                    lower = text.lower()
                    if lower.startswith("params:"):
                        params = text.split(":", 1)[1].split()
                    elif lower == "data":
                        section = "data"
                    elif lower == "cov":
                        section = "cov"
                    continue

                values = [float(x) for x in line.split()]
                if section == "data":
                    data = np.asarray(values, dtype=float)
                    section = None
                elif section == "cov":
                    cov_rows.append(values)

        if params is None or data is None or len(cov_rows) != 3:
            raise ValueError(
                f"Malformed compressed CMB data file '{path}'. Expected params, one data row and 3 cov rows."
            )
        return list(params), data, np.asarray(cov_rows, dtype=float)

    def _validate_full_data(self):
        if self.full_data_params[:2] != ["omega_b", "omega_m"]:
            raise ValueError(f"Expected first two params ['omega_b', 'omega_m']; got {self.full_data_params[:2]}")
        if len(self.full_data_params) != 3:
            raise ValueError(f"Expected exactly 3 params in data file; got {self.full_data_params}")
        if self.full_data_params[2] != self.theta_param:
            raise ValueError(
                f"Basis/data mismatch: requested '{self.theta_param}' but file has '{self.full_data_params[2]}'"
            )
        if self.full_data.shape != (3,):
            raise ValueError(f"Data vector must have shape (3,), got {self.full_data.shape}")
        if self.full_cov.shape != (3, 3):
            raise ValueError(f"Covariance matrix must have shape (3, 3), got {self.full_cov.shape}")
        if not np.allclose(self.full_cov, self.full_cov.T, rtol=1e-10, atol=1e-18):
            raise ValueError("Covariance matrix is not symmetric")
        eigvals = np.linalg.eigvalsh(self.full_cov)
        if np.any(eigvals <= 0):
            raise ValueError(f"Covariance matrix is not positive definite; eigenvalues={eigvals}")

    def _select_basis_from_full_data(self):
        missing = [p for p in self.compression_basis if p not in self.full_data_params]
        if missing:
            raise ValueError(
                f"compression_basis {self.compression_basis} is not a subset of data-file params "
                f"{self.full_data_params}. Missing: {missing}"
            )
        idx = [self.full_data_params.index(p) for p in self.compression_basis]
        self.data_params = list(self.compression_basis)
        self.data = self.full_data[idx]
        self.cov = self.full_cov[np.ix_(idx, idx)]

    def _validate_selected_data(self):
        n = len(self.data_params)
        if self.data.shape != (n,):
            raise ValueError(f"Selected data vector must have shape ({n},), got {self.data.shape}")
        if self.cov.shape != (n, n):
            raise ValueError(f"Selected covariance matrix must have shape ({n}, {n}), got {self.cov.shape}")
        eigvals = np.linalg.eigvalsh(self.cov)
        if np.any(eigvals <= 0):
            raise ValueError(f"Selected covariance matrix is not positive definite; eigenvalues={eigvals}")

    def _uses_theta_s(self):
        return self.theta_param in {"theta_s", "1/theta_s"}

    def _uses_theta_drag(self):
        return self.theta_param in {"theta_drag", "1/theta_drag"}

    def _get_omega_m(self):
        Omega_m = self.provider.get_param("Omega_m")
        h = self.provider.get_param("H0") / 100.0
        return Omega_m * h**2

    def _get_theta_s(self):
        return self.provider.get_param("theta_s_100") / 100.0

    def _get_theta_drag(self):
        z_d = self.provider.get_param("z_d")
        if not (self.z_bg[0] <= z_d <= self.z_bg[-1]):
            raise ValueError(
                f"z_d={z_d} is outside interpolation grid "
                f"[{self.z_bg[0]}, {self.z_bg[-1]}]. Increase z_drag_min/max."
            )

        DA_grid = np.asarray(self.provider.get_angular_diameter_distance(self.z_bg), dtype=float)
        DA_zd = float(np.interp(z_d, self.z_bg, DA_grid))
        rs_drag = self.provider.get_param("rs_drag")
        return rs_drag / ((1.0 + z_d) * DA_zd)

    def _full_theory_dict(self):
        theory = {
            "omega_b": self.provider.get_param("omega_b"),
            "omega_m": self._get_omega_m(),
        }

        if self.theta_param == "theta_s":
            theory["theta_s"] = self._get_theta_s()
        elif self.theta_param == "1/theta_s":
            theory["1/theta_s"] = 1.0 / self._get_theta_s()
        elif self.theta_param == "theta_drag":
            theory["theta_drag"] = self._get_theta_drag()
        elif self.theta_param == "1/theta_drag":
            theory["1/theta_drag"] = 1.0 / self._get_theta_drag()
        else:
            raise RuntimeError(f"Unsupported theta parameter after validation: {self.theta_param}")

        return theory

    def _theory_vector(self):
        theory = self._full_theory_dict()
        return np.array([theory[p] for p in self.data_params], dtype=float)

    def logp(self, **params_values):
        theory = self._theory_vector()
        diff = theory - self.data
        chi2 = float(diff @ self.inv_cov @ diff)

        if self.debug:
            sigma = np.sqrt(np.diag(self.cov))
            print("\n[CompressedCMB debug]")
            print(f"data file         = {self.data_path}")
            print(f"full data params  = {self.full_data_params}")
            print(f"compression_basis = {self.data_params}")
            print(f"data              = {self.data}")
            print(f"theory            = {theory}")
            print(f"diff              = {diff}")
            print(f"pull              = {diff / sigma}")
            print(f"chi2              = {chi2:.12e}")

        return -0.5 * chi2


# Allows Cobaya syntax:
#
# likelihood:
#   compressed_CMB:
#     python_path: /path/to/dir
#     compression_basis: [omega_b, omega_m, theta_s]
compressed_CMB = CompressedCMB
