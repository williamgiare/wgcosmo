# Compressed CMB likelihood

Compressed Planck CMB likelihood for Cobaya.

- It reads a data vector and covariance matrix from `compressed_data_vectors/*.dat` and evaluates the corresponding Gaussian likelihood.
- The compressed files are obtained with `Covariance_Matrix.ipynb`.

## Compression basis

Use `compression_basis` to choose the compressed parameters.

The available parameters are:

```text
omega_b = Omega_b h^2
omega_m = Omega_m h^2
theta_s
1/theta_s
1/theta_drag
```

Examples:

```yaml
compression_basis: [omega_b, omega_m, theta_s]
compression_basis: [omega_b, omega_m, 1/theta_s]
compression_basis: [omega_b, omega_m, 1/theta_drag]
```

2D sub-compressions are also supported:

```yaml
compression_basis: [omega_b, theta_s]
compression_basis: [omega_b, 1/theta_s]
compression_basis: [omega_b, 1/theta_drag]
```

The corresponding default files are:

```text
compressed_data_vectors/wb_wm_Thetas.dat
compressed_data_vectors/wb_wm_invThetas.dat
compressed_data_vectors/wb_wm_invThetadrag.dat
```

## Cobaya usage

- see `Example.yaml`
