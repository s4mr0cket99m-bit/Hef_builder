"""Utilities to represent antibiotic molecular orbitals and properties as tensors.

The goal of this module is to provide a reproducible way to encode a small-molecule
antibiotic into a set of numpy arrays that an LLM or other model can ingest. It
intentionally avoids heavyweight cheminformatics dependencies so it can run in
restricted environments; the numerical approximations are *illustrative* and not
intended to replace quantum-chemistry packages.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from pathlib import Path

import numpy as np

# -------------------------
# Core domain definitions
# -------------------------


@dataclass
class Atom:
    """Simple atom description.

    Attributes:
        element: Chemical symbol (e.g., "C", "N", "O").
        position: 3D Cartesian coordinates in angstroms.
        partial_charge: Optional Mulliken/ESP-style partial charge.
        properties: Additional scalar properties (e.g., hybridization, H-bond role).
    """

    element: str
    position: Tuple[float, float, float]
    partial_charge: float = 0.0
    properties: Mapping[str, float] = field(default_factory=dict)

    def as_feature_vector(self, element_vocab: Sequence[str]) -> np.ndarray:
        """Encode the atom as a numeric feature vector.

        The vector concatenates a one-hot element embedding, the partial charge,
        and any extra scalar properties provided in ``properties`` (sorted by key
        for deterministic ordering).
        """

        one_hot = np.zeros(len(element_vocab), dtype=np.float32)
        try:
            one_hot[element_vocab.index(self.element)] = 1.0
        except ValueError:
            pass  # Unknown elements remain all-zero but can still carry properties.

        property_values = [self.partial_charge]
        for key in sorted(self.properties):
            property_values.append(float(self.properties[key]))
        return np.concatenate([one_hot, np.asarray(property_values, dtype=np.float32)])


@dataclass
class FunctionalGroup:
    """Collection of atoms representing a functional group (e.g., beta-lactam ring)."""

    name: str
    atom_indices: Sequence[int]
    properties: Mapping[str, float] = field(default_factory=dict)


@dataclass
class MolecularOrbital:
    """Approximate molecular orbital description.

    Attributes:
        energy_ev: Orbital energy in electron volts.
        occupancy: Occupancy (0, 1, or 2 for closed-shell).
        center: Spatial center used for heuristic orbital shape generation.
        spread: Controls Gaussian width (in angstroms) for the orbital field.
        symmetry: Descriptive label (e.g., "pi", "sigma", "n").
    """

    energy_ev: float
    occupancy: float
    center: Tuple[float, float, float]
    spread: float
    symmetry: str = ""


@dataclass
class MoleculeSpec:
    """High-level specification of an antibiotic molecule."""

    name: str
    atoms: List[Atom]
    orbitals: List[MolecularOrbital]
    functional_groups: List[FunctionalGroup]

    metadata: Mapping[str, str] = field(default_factory=dict)


# -------------------------
# Tensorization utilities
# -------------------------


class MoleculeTensorBuilder:
    """Builds tensor representations for molecules.

    This class creates several arrays:
      * atom_positions: (n_atoms, 3) Cartesian coordinates
      * atom_features: (n_atoms, feature_dim) numeric descriptors
      * functional_group_mask: (n_groups, n_atoms) binary association matrix
      * orbital_grid: (n_orbitals, gx, gy, gz) orbital amplitude values
      * electron_density: (gx, gy, gz) aggregated electron density
    """

    def __init__(
        self,
        element_vocab: Iterable[str] | None = None,
        grid_resolution: float = 0.5,
        margin: float = 2.0,
    ) -> None:
        self.element_vocab = list(element_vocab or ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"])
        self.grid_resolution = grid_resolution
        self.margin = margin

    # Public API
    # ---------

    def build_tensors(self, spec: MoleculeSpec) -> Dict[str, np.ndarray]:
        """Generate tensor dictionaries from a molecule specification."""

        positions = np.asarray([atom.position for atom in spec.atoms], dtype=np.float32)
        feature_rows = [atom.as_feature_vector(self.element_vocab) for atom in spec.atoms]
        atom_features = self._pad_features(feature_rows)

        functional_mask = self._build_functional_group_mask(len(spec.atoms), spec.functional_groups)
        grid, grid_origin = self._allocate_grid(positions)
        orbital_grid = self._render_orbitals(spec.orbitals, grid, grid_origin)
        electron_density = self._estimate_electron_density(positions, spec.atoms, grid, grid_origin)

        return {
            "atom_positions": positions,
            "atom_features": atom_features,
            "functional_group_mask": functional_mask,
            "orbital_grid": orbital_grid,
            "electron_density": electron_density,
            "grid_origin": grid_origin.astype(np.float32),
            "grid_resolution": np.float32(self.grid_resolution),
        }

    # Internal helpers
    # ----------------

    def _pad_features(self, feature_rows: List[np.ndarray]) -> np.ndarray:
        max_len = max(row.shape[0] for row in feature_rows)
        padded = np.zeros((len(feature_rows), max_len), dtype=np.float32)
        for idx, row in enumerate(feature_rows):
            padded[idx, : row.shape[0]] = row
        return padded

    def _build_functional_group_mask(self, n_atoms: int, groups: Sequence[FunctionalGroup]) -> np.ndarray:
        mask = np.zeros((len(groups), n_atoms), dtype=np.float32)
        for gi, group in enumerate(groups):
            for atom_idx in group.atom_indices:
                if 0 <= atom_idx < n_atoms:
                    mask[gi, atom_idx] = 1.0
        return mask

    def _allocate_grid(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mins = positions.min(axis=0) - self.margin
        maxs = positions.max(axis=0) + self.margin
        grid_axes = [
            np.arange(start, stop + self.grid_resolution, self.grid_resolution)
            for start, stop in zip(mins, maxs)
        ]
        grid_shape = (len(grid_axes[0]), len(grid_axes[1]), len(grid_axes[2]))
        grid = np.zeros(grid_shape, dtype=np.float32)
        return grid, mins.astype(np.float32)

    def _render_orbitals(
        self,
        orbitals: Sequence[MolecularOrbital],
        grid_template: np.ndarray,
        origin: np.ndarray,
    ) -> np.ndarray:
        orbital_stack = np.zeros((len(orbitals),) + grid_template.shape, dtype=np.float32)
        xs, ys, zs = self._grid_coordinates(grid_template.shape, origin)

        for i, orbital in enumerate(orbitals):
            center = np.asarray(orbital.center, dtype=np.float32)
            sigma2 = (orbital.spread ** 2)
            # Gaussian orbital amplitude approximation
            dist2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2 + (zs - center[2]) ** 2
            amplitude = np.exp(-0.5 * dist2 / sigma2) * np.sign(orbital.occupancy or 1.0)
            orbital_stack[i] = amplitude.astype(np.float32)
        return orbital_stack

    def _estimate_electron_density(
        self,
        positions: np.ndarray,
        atoms: Sequence[Atom],
        grid_template: np.ndarray,
        origin: np.ndarray,
    ) -> np.ndarray:
        xs, ys, zs = self._grid_coordinates(grid_template.shape, origin)
        density = np.zeros_like(grid_template, dtype=np.float32)
        for atom, pos in zip(atoms, positions):
            sigma2 = 0.25  # narrow spread for nuclear-centered density
            dist2 = (xs - pos[0]) ** 2 + (ys - pos[1]) ** 2 + (zs - pos[2]) ** 2
            # weight by typical valence electron count to highlight heteroatoms
            valence = self._valence_electrons(atom.element)
            density += valence * np.exp(-0.5 * dist2 / sigma2)
        return density.astype(np.float32)

    def _grid_coordinates(self, shape: Tuple[int, int, int], origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx, gy, gz = shape
        xs = origin[0] + np.arange(gx)[:, None, None] * self.grid_resolution
        ys = origin[1] + np.arange(gy)[None, :, None] * self.grid_resolution
        zs = origin[2] + np.arange(gz)[None, None, :] * self.grid_resolution
        return np.broadcast_to(xs, shape), np.broadcast_to(ys, shape), np.broadcast_to(zs, shape)

    @staticmethod
    def _valence_electrons(element: str) -> int:
        table = {
            "H": 1,
            "C": 4,
            "N": 5,
            "O": 6,
            "S": 6,
            "P": 5,
            "F": 7,
            "Cl": 7,
            "Br": 7,
            "I": 7,
        }
        return table.get(element, 4)


# -------------------------
# Antibiotic library
# -------------------------


def _scaffold_beta_lactam() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0), partial_charge=-0.1),
        Atom("C", (1.4, 0.0, 0.0), partial_charge=-0.1),
        Atom("N", (2.0, 1.2, 0.1), partial_charge=-0.2),
        Atom("C", (0.5, 1.3, -0.2), partial_charge=-0.1),
        Atom("O", (-0.8, 1.5, 0.0), partial_charge=-0.3),
        Atom("S", (3.2, -0.5, 0.2), partial_charge=-0.1),
        Atom("O", (3.9, 0.7, 0.4), partial_charge=-0.3),
        Atom("O", (3.8, -1.6, 0.1), partial_charge=-0.3),
        Atom("N", (-1.5, -0.2, -0.1), partial_charge=-0.2),
        Atom("C", (-2.3, 1.0, -0.2), partial_charge=-0.1),
    ]

    groups = [
        FunctionalGroup("beta_lactam", [0, 1, 2, 3, 4]),
        FunctionalGroup("thiazolidine", [1, 2, 5, 6, 7]),
        FunctionalGroup("amine", [8]),
        FunctionalGroup("carboxyl", [9, 4]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-10.5, occupancy=2, center=(0.5, 0.6, 0.0), spread=0.8, symmetry="sigma"),
        MolecularOrbital(energy_ev=-7.8, occupancy=2, center=(1.6, 0.4, 0.1), spread=0.9, symmetry="pi"),
        MolecularOrbital(energy_ev=-3.2, occupancy=2, center=(2.7, -0.2, 0.2), spread=1.1, symmetry="pi"),
        MolecularOrbital(energy_ev=-1.1, occupancy=1, center=(-1.0, 0.6, -0.1), spread=1.2, symmetry="n"),
    ]
    return atoms, groups, orbitals


def _scaffold_macrolide() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0), partial_charge=-0.1),
        Atom("C", (1.2, 0.6, 0.2), partial_charge=-0.1),
        Atom("C", (2.4, 0.1, -0.1), partial_charge=-0.1),
        Atom("O", (3.4, 0.8, 0.0), partial_charge=-0.25),
        Atom("C", (4.4, 0.2, 0.3), partial_charge=-0.1),
        Atom("O", (5.4, 0.9, 0.4), partial_charge=-0.25),
        Atom("C", (2.3, -1.2, -0.6), partial_charge=-0.1),
        Atom("N", (1.0, -1.4, -0.4), partial_charge=-0.15),
        Atom("C", (3.5, -1.8, -0.4), partial_charge=-0.1),
        Atom("O", (4.6, -1.5, -0.2), partial_charge=-0.25),
        Atom("C", (0.8, 1.7, 0.4), partial_charge=-0.1),
        Atom("O", (1.7, 2.6, 0.6), partial_charge=-0.25),
    ]

    groups = [
        FunctionalGroup("macrolactone_ring", list(range(0, 6))),
        FunctionalGroup("desosamine_amine", [7]),
        FunctionalGroup("cladinose", [10, 11]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-9.8, occupancy=2, center=(2.0, 0.5, 0.0), spread=1.4, symmetry="sigma"),
        MolecularOrbital(energy_ev=-5.7, occupancy=2, center=(1.5, -0.9, -0.3), spread=1.2, symmetry="sigma"),
        MolecularOrbital(energy_ev=-2.6, occupancy=1, center=(3.8, -1.2, -0.1), spread=1.1, symmetry="pi"),
    ]
    return atoms, groups, orbitals


def _scaffold_aminoglycoside() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("C", (1.1, 0.8, 0.0)),
        Atom("C", (2.2, 0.1, 0.0)),
        Atom("O", (-0.9, 0.7, 0.1), partial_charge=-0.2),
        Atom("O", (3.1, 0.8, -0.1), partial_charge=-0.2),
        Atom("N", (1.1, -1.1, 0.1), partial_charge=-0.25),
        Atom("N", (2.2, -1.8, 0.2), partial_charge=-0.25),
        Atom("N", (-1.0, -0.9, -0.1), partial_charge=-0.25),
        Atom("C", (-2.0, 0.0, 0.0)),
    ]

    groups = [
        FunctionalGroup("aminocyclitol_core", [0, 1, 2, 3, 4]),
        FunctionalGroup("amine_1", [5]),
        FunctionalGroup("amine_2", [6]),
        FunctionalGroup("amine_3", [7]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-10.2, occupancy=2, center=(1.0, 0.2, 0.0), spread=1.0, symmetry="sigma"),
        MolecularOrbital(energy_ev=-6.3, occupancy=2, center=(1.5, -1.4, 0.1), spread=0.9, symmetry="n"),
        MolecularOrbital(energy_ev=-3.1, occupancy=1, center=(-0.5, -0.4, 0.0), spread=0.8, symmetry="n"),
    ]
    return atoms, groups, orbitals


def _scaffold_tetracycline() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("C", (1.3, 0.2, 0.0)),
        Atom("C", (2.6, -0.1, 0.0)),
        Atom("C", (3.8, 0.3, 0.0)),
        Atom("C", (5.0, -0.1, 0.0)),
        Atom("O", (0.5, 1.1, 0.0), partial_charge=-0.3),
        Atom("O", (2.1, 1.1, 0.0), partial_charge=-0.3),
        Atom("O", (4.0, 1.2, 0.0), partial_charge=-0.3),
        Atom("N", (5.8, 1.0, 0.0), partial_charge=-0.2),
    ]

    groups = [
        FunctionalGroup("polyketide_ring", [0, 1, 2, 3, 4]),
        FunctionalGroup("keto_enol", [5, 6, 7]),
        FunctionalGroup("dimethylamine", [8]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-8.7, occupancy=2, center=(2.5, 0.5, 0.0), spread=1.3, symmetry="pi"),
        MolecularOrbital(energy_ev=-5.9, occupancy=2, center=(4.5, 0.3, 0.0), spread=1.1, symmetry="pi"),
        MolecularOrbital(energy_ev=-2.8, occupancy=1, center=(5.6, 0.9, 0.0), spread=1.0, symmetry="n"),
    ]
    return atoms, groups, orbitals


def _scaffold_fluoroquinolone() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("C", (1.3, 0.0, 0.0)),
        Atom("C", (2.5, 0.0, 0.0)),
        Atom("C", (3.7, 0.0, 0.0)),
        Atom("N", (5.0, 0.0, 0.0), partial_charge=-0.2),
        Atom("O", (1.3, 1.1, 0.0), partial_charge=-0.3),
        Atom("O", (2.5, 1.1, 0.0), partial_charge=-0.3),
        Atom("F", (3.7, 1.1, 0.0), partial_charge=-0.1),
        Atom("N", (0.0, 1.1, 0.0), partial_charge=-0.2),
    ]

    groups = [
        FunctionalGroup("quinolone_core", [0, 1, 2, 3, 4]),
        FunctionalGroup("carboxyl", [5, 6]),
        FunctionalGroup("fluorine", [7]),
        FunctionalGroup("piperazine", [8]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-9.1, occupancy=2, center=(2.2, 0.2, 0.0), spread=1.0, symmetry="pi"),
        MolecularOrbital(energy_ev=-6.0, occupancy=2, center=(1.9, 1.0, 0.0), spread=0.9, symmetry="pi"),
        MolecularOrbital(energy_ev=-3.0, occupancy=1, center=(4.5, 0.1, 0.0), spread=1.1, symmetry="pi"),
    ]
    return atoms, groups, orbitals


def _scaffold_glycopeptide() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("C", (1.2, 0.5, 0.2)),
        Atom("C", (2.3, -0.3, 0.0)),
        Atom("O", (-0.8, 0.9, 0.1), partial_charge=-0.3),
        Atom("O", (3.2, 0.6, 0.0), partial_charge=-0.3),
        Atom("N", (1.1, -1.2, -0.1), partial_charge=-0.2),
        Atom("N", (2.8, -1.0, 0.1), partial_charge=-0.2),
        Atom("C", (0.5, -2.1, -0.2)),
        Atom("C", (3.5, -1.8, 0.2)),
        Atom("O", (4.5, -1.2, 0.1), partial_charge=-0.3),
    ]

    groups = [
        FunctionalGroup("heptapeptide_core", [0, 1, 2, 3, 4, 5, 6]),
        FunctionalGroup("chlorophenyl", [7, 8, 9]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-9.5, occupancy=2, center=(1.2, 0.1, 0.0), spread=1.2, symmetry="sigma"),
        MolecularOrbital(energy_ev=-6.4, occupancy=2, center=(2.0, -0.7, 0.0), spread=1.1, symmetry="pi"),
        MolecularOrbital(energy_ev=-2.7, occupancy=1, center=(3.9, -1.4, 0.2), spread=1.0, symmetry="pi"),
    ]
    return atoms, groups, orbitals


def _scaffold_membrane_lipopeptide() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("C", (1.1, 0.6, 0.0)),
        Atom("C", (2.2, 0.0, 0.0)),
        Atom("C", (3.3, 0.6, 0.0)),
        Atom("O", (-0.9, 0.8, 0.0), partial_charge=-0.3),
        Atom("N", (1.1, -1.0, 0.0), partial_charge=-0.2),
        Atom("N", (2.2, -1.6, 0.0), partial_charge=-0.2),
        Atom("O", (3.3, -1.0, 0.0), partial_charge=-0.3),
        Atom("C", (4.4, 0.0, 0.0)),
        Atom("C", (5.5, -0.6, 0.0)),
    ]

    groups = [
        FunctionalGroup("cyclic_lipopeptide", [0, 1, 2, 3, 4, 5, 6, 7]),
        FunctionalGroup("lipid_tail", [8, 9]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-8.9, occupancy=2, center=(1.5, 0.2, 0.0), spread=1.3, symmetry="sigma"),
        MolecularOrbital(energy_ev=-6.2, occupancy=2, center=(2.5, -0.8, 0.0), spread=1.0, symmetry="pi"),
        MolecularOrbital(energy_ev=-3.2, occupancy=1, center=(4.8, -0.2, 0.0), spread=1.1, symmetry="sigma"),
    ]
    return atoms, groups, orbitals


def _scaffold_rifamycin() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("C", (1.2, 0.4, 0.0)),
        Atom("C", (2.3, -0.3, 0.0)),
        Atom("C", (3.4, 0.4, 0.0)),
        Atom("C", (4.5, -0.3, 0.0)),
        Atom("O", (1.2, 1.4, 0.0), partial_charge=-0.3),
        Atom("O", (2.3, 1.4, 0.0), partial_charge=-0.3),
        Atom("O", (3.4, 1.4, 0.0), partial_charge=-0.3),
        Atom("O", (4.5, 1.4, 0.0), partial_charge=-0.3),
        Atom("N", (-0.9, 0.7, 0.0), partial_charge=-0.2),
    ]

    groups = [
        FunctionalGroup("ansamycin_macrocycle", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        FunctionalGroup("piperazine", [9]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-8.4, occupancy=2, center=(2.5, 0.5, 0.0), spread=1.3, symmetry="pi"),
        MolecularOrbital(energy_ev=-5.5, occupancy=2, center=(3.5, 0.4, 0.0), spread=1.1, symmetry="pi"),
        MolecularOrbital(energy_ev=-2.5, occupancy=1, center=(1.5, 0.8, 0.0), spread=1.0, symmetry="pi"),
    ]
    return atoms, groups, orbitals


def _scaffold_folate_inhibitor() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("C", (1.2, 0.0, 0.0)),
        Atom("C", (2.4, 0.0, 0.0)),
        Atom("N", (3.6, 0.0, 0.0), partial_charge=-0.2),
        Atom("O", (1.2, 1.0, 0.0), partial_charge=-0.3),
        Atom("O", (2.4, 1.0, 0.0), partial_charge=-0.3),
        Atom("N", (0.0, 1.0, 0.0), partial_charge=-0.2),
    ]

    groups = [
        FunctionalGroup("benzyl_or_pyrimidine", [0, 1, 2, 3]),
        FunctionalGroup("sulfonamide_or_diaminopyrimidine", [4, 5, 6]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-9.0, occupancy=2, center=(1.5, 0.1, 0.0), spread=1.0, symmetry="pi"),
        MolecularOrbital(energy_ev=-6.1, occupancy=2, center=(1.8, 0.9, 0.0), spread=0.9, symmetry="n"),
        MolecularOrbital(energy_ev=-3.4, occupancy=1, center=(0.4, 0.8, 0.0), spread=0.9, symmetry="n"),
    ]
    return atoms, groups, orbitals


def _scaffold_nitroimidazole() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("N", (1.2, 0.0, 0.0), partial_charge=-0.2),
        Atom("C", (2.4, 0.0, 0.0)),
        Atom("N", (3.6, 0.0, 0.0), partial_charge=-0.2),
        Atom("O", (1.2, 1.0, 0.0), partial_charge=-0.3),
        Atom("O", (2.4, 1.0, 0.0), partial_charge=-0.3),
        Atom("N", (0.0, 1.0, 0.0), partial_charge=-0.2),
    ]

    groups = [
        FunctionalGroup("nitroimidazole_ring", [0, 1, 2, 3]),
        FunctionalGroup("nitro_group", [4, 5, 6]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-8.2, occupancy=2, center=(1.8, 0.1, 0.0), spread=1.0, symmetry="pi"),
        MolecularOrbital(energy_ev=-5.4, occupancy=2, center=(1.8, 0.9, 0.0), spread=0.9, symmetry="n"),
        MolecularOrbital(energy_ev=-2.9, occupancy=1, center=(0.4, 0.8, 0.0), spread=0.9, symmetry="n"),
    ]
    return atoms, groups, orbitals


def _scaffold_polymyxin() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("C", (1.1, 0.7, 0.0)),
        Atom("C", (2.2, 0.0, 0.0)),
        Atom("C", (3.3, 0.7, 0.0)),
        Atom("N", (1.1, -1.0, 0.0), partial_charge=-0.2),
        Atom("N", (2.2, -1.7, 0.0), partial_charge=-0.2),
        Atom("O", (3.3, -1.0, 0.0), partial_charge=-0.3),
        Atom("C", (4.4, 0.0, 0.0)),
        Atom("C", (5.5, -0.6, 0.0)),
        Atom("C", (6.6, -1.2, 0.0)),
    ]

    groups = [
        FunctionalGroup("cyclic_polypeptide", [0, 1, 2, 3, 4, 5, 6]),
        FunctionalGroup("fatty_acid_tail", [7, 8, 9]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-8.8, occupancy=2, center=(1.6, 0.1, 0.0), spread=1.2, symmetry="sigma"),
        MolecularOrbital(energy_ev=-6.0, occupancy=2, center=(2.4, -0.9, 0.0), spread=1.0, symmetry="pi"),
        MolecularOrbital(energy_ev=-3.1, occupancy=1, center=(5.0, -0.6, 0.0), spread=1.2, symmetry="sigma"),
    ]
    return atoms, groups, orbitals


def _scaffold_misc_protein_synthesis() -> Tuple[List[Atom], List[FunctionalGroup], List[MolecularOrbital]]:
    atoms = [
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("C", (1.1, 0.4, 0.0)),
        Atom("C", (2.2, -0.2, 0.0)),
        Atom("O", (3.2, 0.5, 0.0), partial_charge=-0.3),
        Atom("N", (1.1, -1.0, 0.0), partial_charge=-0.2),
        Atom("Cl", (2.8, -1.2, 0.0), partial_charge=-0.1),
    ]

    groups = [
        FunctionalGroup("aryl_core", [0, 1, 2, 3]),
        FunctionalGroup("amide_or_morpholine", [4]),
        FunctionalGroup("halogen", [5]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-8.0, occupancy=2, center=(1.5, 0.1, 0.0), spread=1.0, symmetry="pi"),
        MolecularOrbital(energy_ev=-5.7, occupancy=2, center=(2.0, -0.7, 0.0), spread=0.9, symmetry="pi"),
        MolecularOrbital(energy_ev=-3.5, occupancy=1, center=(2.8, -1.0, 0.0), spread=0.9, symmetry="n"),
    ]
    return atoms, groups, orbitals


def _clone_scaffold(scaffold_fn, name: str, mechanism: str, structural_class: str) -> MoleculeSpec:
    atoms, groups, orbitals = scaffold_fn()
    return MoleculeSpec(
        name=name,
        atoms=copy.deepcopy(atoms),
        orbitals=copy.deepcopy(orbitals),
        functional_groups=copy.deepcopy(groups),
        metadata={
            "mechanism": mechanism,
            "class": structural_class,
            "example": name,
        },
    )


def build_antibiotic_library() -> List[MoleculeSpec]:
    """Return MoleculeSpecs for a broad set of approved antibiotics grouped by mechanism."""

    library: List[MoleculeSpec] = []

    beta_lactams = [
        "penicillin_g",
        "penicillin_v",
        "amoxicillin",
        "ampicillin",
        "oxacillin",
        "cloxacillin",
        "dicloxacillin",
        "nafcillin",
        "methicillin",
        "piperacillin",
        "ticarcillin",
        "cefazolin",
        "cephalexin",
        "cefuroxime",
        "cefotaxime",
        "ceftriaxone",
        "cefepime",
        "cefdinir",
        "cefixime",
        "ceftazidime",
        "ceftaroline",
        "imipenem",
        "meropenem",
        "ertapenem",
        "doripenem",
        "aztreonam",
    ]
    library.extend(
        _clone_scaffold(_scaffold_beta_lactam, name, "cell_wall_synthesis_inhibitor", "beta-lactam")
        for name in beta_lactams
    )

    macrolides = [
        "erythromycin",
        "azithromycin",
        "clarithromycin",
        "fidaxomicin",
        "telithromycin",
    ]
    library.extend(
        _clone_scaffold(_scaffold_macrolide, name, "protein_synthesis_inhibitor_50s", "macrolide")
        for name in macrolides
    )

    aminoglycosides = [
        "gentamicin",
        "tobramycin",
        "amikacin",
        "streptomycin",
        "plazomicin",
    ]
    library.extend(
        _clone_scaffold(_scaffold_aminoglycoside, name, "protein_synthesis_inhibitor_30s", "aminoglycoside")
        for name in aminoglycosides
    )

    tetracyclines = [
        "tetracycline",
        "doxycycline",
        "minocycline",
        "tigecycline",
        "eravacycline",
        "omadacycline",
    ]
    library.extend(
        _clone_scaffold(_scaffold_tetracycline, name, "protein_synthesis_inhibitor_30s", "tetracycline")
        for name in tetracyclines
    )

    fluoroquinolones = [
        "ciprofloxacin",
        "levofloxacin",
        "moxifloxacin",
        "ofloxacin",
        "delafloxacin",
    ]
    library.extend(
        _clone_scaffold(_scaffold_fluoroquinolone, name, "dna_gyrase_topoisomerase_inhibitor", "fluoroquinolone")
        for name in fluoroquinolones
    )

    glycopeptides = [
        "vancomycin",
        "teicoplanin",
        "dalbavancin",
        "oritavancin",
    ]
    library.extend(
        _clone_scaffold(_scaffold_glycopeptide, name, "cell_wall_synthesis_inhibitor", "glycopeptide")
        for name in glycopeptides
    )

    membrane_actives = [
        "daptomycin",
    ]
    library.extend(
        _clone_scaffold(_scaffold_membrane_lipopeptide, name, "membrane_depolarization", "lipopeptide")
        for name in membrane_actives
    )

    rifamycins = [
        "rifampin",
        "rifabutin",
    ]
    library.extend(
        _clone_scaffold(_scaffold_rifamycin, name, "rna_polymerase_inhibitor", "rifamycin")
        for name in rifamycins
    )

    folate_inhibitors = [
        "trimethoprim",
        "sulfamethoxazole",
        "sulfadiazine",
        "sulfisoxazole",
    ]
    library.extend(
        _clone_scaffold(_scaffold_folate_inhibitor, name, "folate_pathway_inhibitor", "antifolate")
        for name in folate_inhibitors
    )

    nitroimidazoles = [
        "metronidazole",
        "tinidazole",
        "secnidazole",
    ]
    library.extend(
        _clone_scaffold(_scaffold_nitroimidazole, name, "dna_strand_breakage_after_reduction", "nitroimidazole")
        for name in nitroimidazoles
    )

    polymyxins = [
        "colistin",
        "polymyxin_b",
    ]
    library.extend(
        _clone_scaffold(_scaffold_polymyxin, name, "outer_membrane_disruption", "polymyxin")
        for name in polymyxins
    )

    misc_50s = [
        "linezolid",
        "tedizolid",
        "clindamycin",
        "chloramphenicol",
        "lefamulin",
    ]
    library.extend(
        _clone_scaffold(_scaffold_misc_protein_synthesis, name, "protein_synthesis_inhibitor_50s", "oxazolidinone_or_related")
        for name in misc_50s
    )

    return library


def _pad_to_shape(array: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    pad_width = [(0, max(t - s, 0)) for s, t in zip(array.shape, target_shape)]
    return np.pad(array, pad_width, mode="constant")


def _update_mechanism_accumulator(accumulator: Dict, tensors: Mapping[str, np.ndarray], mechanism: str) -> None:
    if mechanism not in accumulator:
        accumulator[mechanism] = {"sums": {}, "max_shape": {}, "count": 0}

    mech_entry = accumulator[mechanism]

    for key, value in tensors.items():
        if isinstance(value, np.ndarray) and value.dtype != object:
            current_max = mech_entry["max_shape"].get(key, value.shape)
            new_max = tuple(max(a, b) for a, b in zip(current_max, value.shape))

            if key in mech_entry["sums"] and new_max != current_max:
                mech_entry["sums"][key] = _pad_to_shape(mech_entry["sums"][key], new_max)

            padded_value = _pad_to_shape(value, new_max)
            mech_entry["sums"][key] = mech_entry["sums"].get(key, np.zeros(new_max, dtype=np.float32)) + padded_value
            mech_entry["max_shape"][key] = new_max

    mech_entry["count"] += 1


def _write_average_mechanism_matrices(output_dir: Path, accumulator: Mapping[str, Dict]) -> None:
    for mechanism, entry in accumulator.items():
        averages = {key: value / float(entry["count"]) for key, value in entry["sums"].items()}
        averages["mechanism"] = np.array([mechanism])
        np.savez_compressed(output_dir / f"{mechanism}_average.npz", **averages)


def generate_full_antibiotic_dataset(output_dir: str = "antibiotic_tensor_matrices") -> List[Path]:
    """Generate .npz tensors for all antibiotics in the library plus mechanism averages."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    builder = MoleculeTensorBuilder(grid_resolution=0.5, margin=2.5)
    library = build_antibiotic_library()

    mechanism_accumulator: Dict[str, Dict] = {}
    saved_files: List[Path] = []

    for spec in library:
        tensors = builder.build_tensors(spec)
        tensors["metadata"] = np.array([spec.metadata], dtype=object)
        filename = output_path / f"{spec.name}.npz"
        np.savez_compressed(filename, **tensors)
        saved_files.append(filename)
        _update_mechanism_accumulator(mechanism_accumulator, tensors, spec.metadata.get("mechanism", "unknown"))

    _write_average_mechanism_matrices(output_path, mechanism_accumulator)
    return saved_files


# -------------------------
# Example usage
# -------------------------


def _example_amoxicillin_spec() -> MoleculeSpec:
    """Hand-crafted, coarse amoxicillin-like molecule description.

    Coordinates are placeholders roughly capturing the spatial arrangement of a
    beta-lactam antibiotic. For real applications, feed coordinates from a
    geometry-optimized structure file (e.g., SDF or PDB) and orbitals from a
    quantum-chemistry calculation.
    """

    atoms = [
        Atom("C", (0.0, 0.0, 0.0), partial_charge=-0.1),
        Atom("C", (1.4, 0.0, 0.0), partial_charge=-0.1),
        Atom("N", (2.0, 1.2, 0.1), partial_charge=-0.2),
        Atom("C", (0.5, 1.3, -0.2), partial_charge=-0.1),
        Atom("O", (-0.8, 1.5, 0.0), partial_charge=-0.3),
        Atom("S", (3.2, -0.5, 0.2), partial_charge=-0.1),
        Atom("O", (3.9, 0.7, 0.4), partial_charge=-0.3),
        Atom("O", (3.8, -1.6, 0.1), partial_charge=-0.3),
        Atom("N", (-1.5, -0.2, -0.1), partial_charge=-0.2),
        Atom("C", (-2.3, 1.0, -0.2), partial_charge=-0.1),
    ]

    groups = [
        FunctionalGroup("beta_lactam", [0, 1, 2, 3, 4]),
        FunctionalGroup("thiazolidine", [1, 2, 5, 6, 7]),
        FunctionalGroup("amine", [8]),
        FunctionalGroup("carboxyl", [9, 4]),
    ]

    orbitals = [
        MolecularOrbital(energy_ev=-10.5, occupancy=2, center=(0.5, 0.6, 0.0), spread=0.8, symmetry="sigma"),
        MolecularOrbital(energy_ev=-7.8, occupancy=2, center=(1.6, 0.4, 0.1), spread=0.9, symmetry="pi"),
        MolecularOrbital(energy_ev=-3.2, occupancy=2, center=(2.7, -0.2, 0.2), spread=1.1, symmetry="pi"),
        MolecularOrbital(energy_ev=-1.1, occupancy=1, center=(-1.0, 0.6, -0.1), spread=1.2, symmetry="n"),
    ]

    metadata = {
        "mechanism": "inhibits transpeptidase by acylating active-site serine",
        "class": "beta-lactam",
        "example": "amoxicillin",
    }

    return MoleculeSpec(
        name="amoxicillin_mock",
        atoms=atoms,
        orbitals=orbitals,
        functional_groups=groups,
        metadata=metadata,
    )


def build_example_dataset() -> Dict[str, np.ndarray]:
    """Build tensors for a sample antibiotic molecule.

    Returns a dictionary of numpy arrays suitable for persisting with ``np.savez``
    or feeding into a model for training. Additional molecules can be appended by
    repeating this process with different ``MoleculeSpec`` definitions.
    """

    builder = MoleculeTensorBuilder(grid_resolution=0.4, margin=2.5)
    spec = _example_amoxicillin_spec()
    tensors = builder.build_tensors(spec)
    return tensors


if __name__ == "__main__":
    tensors = build_example_dataset()
    np.savez_compressed("amoxicillin_tensor_example.npz", **tensors)
    print("Saved amoxicillin_tensor_example.npz with keys:", list(tensors))

    outputs = generate_full_antibiotic_dataset()
    print(f"Generated {len(outputs)} antibiotic tensors and mechanism averages in 'antibiotic_tensor_matrices'.")
