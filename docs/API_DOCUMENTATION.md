# DeFillet API Documentation

This document covers every public component exposed by the DeFillet codebase—classes, functions, command-line tools, and configuration knobs. It is intended for contributors building on top of the detection/removal pipeline as well as downstream users embedding the library inside their own tooling.

---

## 1. Architecture Overview

- **Detection stack (`DeFillet::FilletDetector`)** generates Voronoi-based descriptors for a normalized mesh, filters rolling-ball candidates, builds radius/radius-rate scalar fields, and classifies faces through graph-cut.
- **Removal stack (`DeFillet` / `DeFilletv2`)** consumes the detector’s segmentation to recover sharp geometry by solving constrained optimization problems over the fillet region.
- **Utility namespaces** wrap config loading, mesh normalization, persistence helpers, KNN queries, Voronoi generation, and graph-cut cost functors.
- **CLI (`detector_cli`)** orchestrates the full detection pipeline from a JSON config file and emits intermediate artifacts.

All APIs live under the `DeFillet` namespace unless explicitly noted.

---

## 2. Fillet Detection Pipeline (`include/fillet_detector.h`)

### 2.1 `FilletDetectorParameters`

| Field | Type | Meaning / Expected Range |
| --- | --- | --- |
| `input_path` | `std::string` | Absolute/relative path to an input `.ply` mesh. Used by the CLI. |
| `out_dir` | `std::string` | Directory where detector artifacts are written. |
| `epsilon` | `float` | Relative tolerance (scaled by Voronoi radius) for osculation checks and rolling-ball fitting. Typical: `0.05 – 0.15`. |
| `radius_thr` | `float` | Minimum allowable radius expressed as a fraction of the mesh bounding-box diagonal. Filters noisy Voronoi vertices. |
| `angle_thr` | `float` | Maximum dihedral angle (degrees) used when expanding regions and linking faces. |
| `sigma` | `float` | Bandwidth parameter for the Gaussian kernel in the 4D density estimator. |
| `lamdba` | `float` | Smoothness weight in the graph-cut energy (typo preserved from source). |
| `num_patches` | `int` | Sub-sampling budget for patch-wise Voronoi generation (`-1` processes the full mesh at once). |
| `num_neighbors` | `int` | Number of neighbors evaluated for Voronoi density estimation. |
| `num_smooth_iter` | `int` | Optional Laplacian smoothing passes for the radius-rate field (misspelled as `num_smmoth_iter` in JSON). |
| `num_sor_neighbors` | `int` | Number of nearest neighbors considered by the statistical outlier removal pass. |
| `num_sor_iter` | `int` | Iterations of SOR. |
| `num_sor_std_ratio` | `float` | Standard deviation multiplier for SOR pruning. |
| `num_threads` | `int` | Upper bound on OpenMP threads (`-1` auto-detects). |

### 2.2 Data Carriers

- `VoronoiVertices`: stores the 3D position, rolling-ball radius, density estimate, neighboring/corresponding mesh faces, fitted axis, and a boolean `flag`.
- `Sites`: one entry per mesh face. Keeps centroid position, owning face handle, fitted radius/rate, rolling-ball center, axis direction, status flag, and graph-cut label.

These types are plain aggregates that callers can inspect after running the detector (e.g., through accessors such as `radius_field()`).

### 2.3 `FilletDetector`

#### Construction

```cpp
DeFillet::FilletDetector detector(surface_mesh, params);
```

- Clones the incoming `SurfaceMesh`, caches bounding boxes, precomputes face normals, dihedral angles (`e:dihedral-angle`), and inter-centroid distances (`e:centroid_distance`).
- Seeds one `Sites` record per face using mesh centroids.

Ensure the mesh already carries consistent topology and has been normalized (see `normalize_model`). All work is performed on the internal copy; the original mesh is untouched.

#### Public Methods

| Method | Role |
| --- | --- |
| `void apply()` | Runs the full pipeline in order: Voronoi generation → filtering → density estimation → rolling-ball fitting → radius/rate computation → (optional) smoothing → graph-cut. Preferred entrypoint. |
| `void generate_voronoi_vertices()` | Builds local/global Voronoi diagrams of face centroids (optionally via farthest-point sampling) and collects unique Voronoi vertices that fall inside the bounding box. |
| `void filter_voronoi_vertices()` | Applies radius thresholding, two stages of statistical outlier removal in 4D (position + radius), and topological osculation checks (connectedness, tangency, genus). |
| `void compute_voronoi_vertices_density_field()` | Estimates densities for each Voronoi vertex using `KNN::KdSearch4D` and stores a dominant rolling axis via PCA (`axis_direction`). |
| `void rolling_ball_trajectory_transform()` | Maps Voronoi vertices back to mesh faces, projects centroids onto axis lines, validates rolling-ball fit, and populates `Sites::center`, `Sites::axis`, `Sites::radius`. |
| `void compute_fillet_radius_field()` | Flood-fills through dihedral-friendly neighbors to average valid radii per face. |
| `void compute_fillet_radius_rate_field()` | For every face, computes a normalized gradient of radii across adjacent faces to quantify how “fillet-like” the region is. |
| `void rate_field_smoothing()` | Optional Jacobi smoothing over the radius-rate field (`num_smooth_iter` passes). Not invoked by default inside `apply()`, but exposed for experimentation. |
| `void graph_cut()` | Builds a binary graph-cut problem with per-face data terms derived from radius-rate scores and smoothness terms weighted by edge length/dihedral gating. Labels are stored in `Sites::label`. |

#### Accessors

| Method | Description |
| --- | --- |
| `PointCloud* voronoi_vertices() const` | Returns a new `easy3d::PointCloud` with vertex positions equal to stored Voronoi vertices and normals encoding fitted axes. Caller owns the pointer. |
| `PointCloud* rolling_ball_centers() const` | Returns centers of accepted rolling balls (with normals as axes). |
| `std::vector<float> radius_field() const` | Per-face radii after aggregation. Same order as `mesh_->faces()`. |
| `std::vector<float> radius_rate_field() const` | Per-face normalized variation metric (0 → very smooth constant radius, 1 → high variation). |
| `std::vector<int> fillet_labels() const` | Graph-cut output (`0` non-fillet, `1` fillet). |

#### Usage Example

```cpp
#include <fillet_detector.h>
#include <utils.h>
#include <easy3d/fileio/surface_mesh_io.h>

int main() {
    auto mesh = easy3d::SurfaceMeshIO::load("wheel.ply");
    easy3d::vec3 centroid;
    double scale = 1.0;
    DeFillet::normalize_model(mesh, centroid, scale);

    auto params = DeFillet::load_detector_config("detector_config.json");
    DeFillet::FilletDetector detector(mesh, params);
    detector.apply();

    auto labels = detector.fillet_labels();
    DeFillet::save_fillet_segmentation(mesh, labels, "wheel_seg.ply");
    auto centers = detector.rolling_ball_centers();
    easy3d::PointCloudIO::save("rolling_centers.ply", centers);
}
```

#### Threading Notes

- Voronoi generation for sampled patches, density estimation, rolling-ball fitting, and graph-cut label extraction use OpenMP. Respect `parameters_.num_threads`.
- Callers embedding the detector inside multi-threaded applications should create independent instances or wrap calls in critical sections.

---

## 3. Detector Utilities (`include/utils.h`)

### Configuration & Normalization

| Function | Description / Usage |
| --- | --- |
| `FilletDetectorParameters load_detector_config(const std::string& filename)` | Parses a JSON config into `FilletDetectorParameters`. Throws `std::runtime_error` if the file is missing. Accepts both absolute and relative paths. |
| `void normalize_model(SurfaceMesh* mesh, vec3& centroids, double& scale)` | Recenters the mesh at the origin and scales it so the furthest vertex lies at unit radius. Returns the original centroid/scale so inverse normalization is possible. |
| `void inverse_normalize_model(SurfaceMesh* mesh, vec3& centroids, double& scale)` | Applies the inverse transformation recorded during normalization. |
| `std::string get_time_stamp(bool for_filename=false)` | Generates either `YYYY-MM-DD HH:MM:SS` (default) or `YYYYMMDD_HHMMSS` (filename-safe) timestamps. |

### Geometry Helpers

| Function | Purpose |
| --- | --- |
| `float angle_between(const vec3& n1, const vec3& n2)` | Numerically stable unsigned angle (degrees) using `atan2(||n1×n2||, n1·n2)`. |
| `double gaussian_kernel(double distance, double kernel_bandwidth)` | Scalar Gaussian weight used by density estimation. |
| `void sor(const std::vector<vec4>& points, std::vector<bool>& labels, int nb_neighbors, int num_sor_iter, float std_ratio)` | In-place statistical outlier removal. `labels` acts both as input mask (points already suppressed remain false) and output classification. |
| `vec3 axis_direction(std::vector<vec4> points)` | Performs PCA over positions of 4D samples (ignore `w`) and returns the dominant eigenvector (unit length). |
| `vec3 project_to_line(vec3 pos, vec3 point, vec3 dir)` | Orthogonally projects `pos` onto the infinite line defined by `point + t*dir`. |

### Persistence Helpers

| Function | Description |
| --- | --- |
| `void save_components(const SurfaceMesh* mesh, const std::vector<SurfaceMesh*> components, const std::string path)` | Colors each face in `mesh` based on membership of extracted components (assumes each component carries `f:original_index`) and writes it to disk. |
| `SurfaceMesh* split_component(const SurfaceMesh* mesh, SurfaceMesh::FaceProperty<int>& component_labels, int label)` | Extracts a new mesh consisting of faces whose label matches `label`. Copies positions, face/vertex indices, and stores back references via `v:original_index` / `f:original_index`. Caller takes ownership of the returned mesh. |
| `void save_field(const SurfaceMesh* mesh, const std::vector<float>& field, const std::string path)` | Maps scalar values to RGB via `igl::jet` and saves the colored mesh (`f:color`). |
| `void save_fillet_segmentation(const SurfaceMesh* mesh, const std::vector<int>& fillet_label, const std::string path)` | Emits a red/blue face-colored mesh representing the binary segmentation. |

---

## 4. Voronoi & KNN Support

### 4.1 4D KD-Tree (`include/knn4d.h`, `src/knn4d.cpp`)

- `struct KNN::Point4D` encapsulates a single sample `(x, y, z, w)` and is trivially constructible.
- `class KNN::KdSearch4D` wraps a `nanoflann` KD-tree:
  - `KdSearch4D(std::vector<Point4D>& points)` builds an index over the provided vector (no copy; keep the vector alive).
  - `~KdSearch4D()` destroys the KD-tree.
  - `void kth_search(const Point4D& p, int k, std::vector<size_t>& neighbors, std::vector<double>& squared_distances) const` performs a kNN query around `p`, returning neighbor indices into the original vector plus squared distances. Distances remain unsorted if `k` exceeds dataset size.

Usage tip: reuse a single `KdSearch4D` instance when scanning multiple query points; construction dominates runtime.

### 4.2 3D Voronoi Generator (`include/voronoi3d.h`)

```cpp
void voronoi3d(const std::vector<vec3>& s,
               const easy3d::Box3& box,
               std::vector<vec3>& vv,
               std::vector<float>& vvr,
               std::vector<std::vector<int>>& snv,
               std::vector<std::vector<int>>& vns);
```

- Builds a CGAL Delaunay triangulation from input `s` (site centroids), inserts oversized bounding-box corners to bound cells, and emits:
  - `vv`: Voronoi vertex locations within `box`.
  - `vvr`: Averaged radii per Voronoi vertex.
  - `snv`: for each site, the indices of Voronoi vertices adjacent to it.
  - `vns`: for each Voronoi vertex, the indices of the four generating sites.
- The function clears and resizes outputs internally. Pass empty containers by reference.

---

## 5. Graph-Cut Helpers (`include/gcp.h`)

The detector uses Kolmogorov & Zabih’s `GCoptimization` library via lightweight functors:

- `class GCP::DataCost : public GCoptimization::DataCostFunctor`
  - Construct with `std::vector<double>& data`, number of nodes, and number of labels.
  - Implements `compute(SiteID s, LabelID l)` by indexing the flattened `data` array at `l * nb_nodes + s`. Populate `data` accordingly (e.g., first `n` entries for label 0, next `n` for label 1).

- `class GCP::SmoothCost : public GCoptimization::SmoothCostFunctor`
  - Takes a list of undirected edges `(site_a, site_b)` and matching weights.
  - Stores weights in a nested map keyed by sorted vertex pairs.
  - `compute(SiteID s1, SiteID s2, LabelID l1, LabelID l2)` returns zero when `l1 == l2`; otherwise it looks up the stored weight for the undirected edge. Missing edges default to zero.

Use these functors when building additional graph-cut formulations to stay consistent with the detector’s weighting scheme.

---

## 6. Fillet Removal Modules

### 6.1 Legacy `DeFillet` (`src/defillet.h/.cpp`)

#### Configuration

- Constructor initializes angle (`45°`), optimization weights (`beta = gamma = 1.0`), and `num_opt_iter = 5`.
- Setters:
  - `void set_mesh(SurfaceMesh* mesh)` – also caches the bounding box as `box`.
  - `void set_fillet_mesh(SurfaceMesh* fillet_mesh)` – optional manual override before `run_geodesic`.
  - `void set_angle(double angle)` – controls angular thresholds used when refining target normals.
  - `void set_beta(double beta)`, `void set_gamma(double gamma)`, `void set_num_opt_iter(double num_opt_iter)` – weights and iteration budget for the optimization system.
- Getters: `SurfaceMesh* get_fillet_mesh()`, plus timing accessors `get_geodesic_time()`, `get_defillet_init_time()`, `get_defillet_iter_time()`, `get_target_normals_refine_time()`.

#### Workflow

1. **`extract_fillet_region()`**  
   - Reads a face property `f:gcp_labels` to isolate faces labeled as fillet.
   - Leverages `SurfaceMeshSegmenter` to produce `fillet_mesh_` (shared topology subset).
   - Copies vertex flags `v:sources` from the original mesh.
2. **`run_geodesic()`**  
   - Extracts seed vertices from `fillet_mesh_`, runs Xin & Wang’s exact geodesic algorithm (`geodesic/`), populates geodesic distances (`v:geo_dis`) and ancestor faces (`f:sources`), and sets up target normals derived from source vertex normals.
3. **`refine_target_normal()`**  
   - Iteratively smooths/propagates target normals based on angular compatibility among neighboring faces (controlled by `angle_`).
4. **`init_opt()`**  
   - Builds sparse constraint matrices capturing edge-directional consistency, face-normal alignment, and positional constraints at source vertices.
   - Factorizes the KKT system once using `Eigen::SparseLU`.
5. **`bool opt()`**  
   - Solves the system for updated vertex positions and writes them back to `fillet_mesh_`.
6. **`bool run_defillet()`**  
   - Calls `init_opt()` if needed and repeats `opt()` `num_opt_iter_` times, measuring iteration timing.

Use `run_geodesic()` → `run_defillet()` sequentially after the detector has produced the `f:gcp_labels` face property. The original mesh will be updated in-place using `mesh_->position(v) = fillet_mesh_->position(mapped_v)` upon completion.

#### Example

```cpp
DeFillet remover;
remover.set_mesh(original_mesh);
remover.set_angle(55.0);
remover.set_beta(0.8);
remover.set_gamma(1.2);
remover.set_num_opt_iter(8);

remover.run_geodesic();
if (remover.run_defillet()) {
    easy3d::SurfaceMeshIO::save("defillet_result.ply", original_mesh);
}
```

### 6.2 Experimental `DeFilletv2` (`src/defillet_v2.h/.cpp`)

Construct with the mesh pointer and (optionally) custom thresholds/weights/iteration counts. Key members:

- `SurfaceMesh* mesh_`, `fillet_mesh_`, `non_fillet_mesh_`, `focus_area_`: working meshes (focus area is auto-extracted).
- `std::vector<vec3> tar_nor` and `s_`: per-face target normals and seed positions.
- `Eigen::VectorXd d_` and `Eigen::SparseLU solver_`: reused linear system structures.
- Accumulated energy curves: `std::vector<double> e_e, e_f, e_c`.

#### Methods

| Method | Description |
| --- | --- |
| `void run()` | Convenience wrapper calling `initialize()` followed by `optimize()`. |
| `void initialize()` | Identifies focus areas based on `f:fillet_labels`, segments out the region of interest, selects boundary anchors (respecting dihedral and border criteria), runs Xin-Wang geodesics to derive per-face roots/target normals, and stores distance fields. Outputs debug meshes (`../out/focus_area.ply`, etc.). |
| `void optimize()` | Builds and factorizes the augmented system combining edge-directional energy `E`, face-normal energy `F`, and positional constraints `D`. Executes `num_opt_iter_` iterations, recording energy metrics via `computeEe_and_E_f()`, updates vertex positions, and writes the focus area back into `mesh_`. |
| `void optimize2()` | Alternative solver path using LDLᵗ factorization—currently experimental (kept for reference). |
| `void computeEe_and_E_f()` | Evaluates edge-alignment and face-normal deviation energies for diagnostics. Called each iteration inside `optimize()`. |

**Preconditions:**  
`mesh_` must expose `f:fillet_labels`, `f:labels`, and `v:sources` properties consistent with the detector. The class writes temporary files to `../out/` for inspection; ensure the path exists or adjust before deploying on Linux.

---

## 7. Command-Line Interface (`cli/detector_cli.cpp`)

### Usage

```bash
detector_cli -c path/to/detector_config.json
```

- Relies on [CLI11](https://github.com/CLIUtils/CLI11); `-c / --config` is mandatory.
- Workflow:
  1. Load parameters via `DeFillet::load_detector_config`.
  2. Auto-configure OpenMP thread count (caps at `omp_get_max_threads()`).
  3. Load the `.ply` mesh (Easy3D), normalize geometry, and instantiate `FilletDetector`.
  4. Run `detector.apply()`.
  5. Create an output directory named `<out_dir>/<basename>_<timestamp>/`.
  6. Persist:
     - Voronoi vertices (`*_voronoi_vertices.ply`)
     - Rolling-ball centers (`*_rolling_ball_centers.ply`)
     - Radius field and rate field (`*_radius_field.ply`, `*_rate_field.ply`) via `save_field`
     - Binary segmentation (`*_seg.ply`) via `save_fillet_segmentation`

### Exit Codes

- Returns `0` on success.
- Propagates exceptions (e.g., JSON parsing, IO failures) to stderr and exits with non-zero (courtesy of the thrown exception).

---

## 8. Configuration Reference (`detector_config.json`)

| Key | Example | Description |
| --- | --- | --- |
| `path` | `"C:\\\\Users\\\\...\\\\1898.ply"` | Input mesh file. |
| `out_dir` | `"../out/"` | Parent directory for detector outputs. |
| `epsilon` | `0.08` | Rolling-ball tolerance. |
| `radius_thr` | `0.03` | Min radius (relative). |
| `lamdba` | `0.4` | Graph-cut smoothness weight. |
| `angle_thr` | `60` | Dihedral cutoff (degrees). |
| `sigma` | `1.0` | Gaussian kernel bandwidth. |
| `num_patches` | `-1` | Number of sampled patches (`-1` = global). |
| `num_neighbors` | `100` | kNN size for density. |
| `num_smmoth_iter` | `50` | (!) Typo: maps to `num_smooth_iter`. |
| `num_sor_iter` | `3` | SOR iterations. |
| `num_sor_neighbors` | `50` | SOR kNN size. |
| `num_sor_std_ratio` | `0.8` | Threshold multiplier. |
| `num_threads` | `-1` | Auto-detect threads. |

**Tip:** When programmatically constructing configs, either stick to the existing key spellings (even the typo) or post-process JSON to ensure compatibility.

---

## 9. Putting It All Together

1. **Mesh Preparation**
   - Load surface meshes with Easy3D.
   - Normalize them before detection; store centroid & scale for later.
2. **Detection**
   - Either call `detector_cli` (batch workflows) or embed `FilletDetector` directly.
   - Consume APIs: `fillet_labels()` for segmentation, `radius_field()` for scalar visualization, `rolling_ball_centers()` for geometric debugging.
3. **Post-Processing**
   - Save visual artifacts using `save_field`/`save_fillet_segmentation`.
   - Optionally run `DeFillet` or `DeFilletv2` to reconstruct sharp CAD geometry from segmented regions.
4. **Denormalization**
   - Before exporting final meshes, call `inverse_normalize_model` with the stored centroid/scale.

Following these steps ensures a reproducible pipeline compatible with the SIGGRAPH 2025 DeFillet implementation.

---

For questions or contributions, refer to `README.md` for maintainer contact details.
