"""Context and scaffold building for Mosaic binder design.

Three layers:
  1. DesignSpace — encapsulates scaffold/free-design position split
  2. Loss building — registry-based, produces weighted LinearCombination trees
  3. Context building — combines targets + scaffold into ContextSpec list
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gemmi
import jax.numpy as jnp
from jax import Array

from mosaic.common import TOKENS, LossTerm, LinearCombination
from mosaic.losses.transformations import SetPositions, NoCys
from mosaic.structure_prediction import TargetChain

Loss = LossTerm | LinearCombination
import mosaic.losses.structure_prediction as sp
import mosaic.losses.indexed_scaffolding as is_
import mosaic.losses.unindexed_scaffolding as us
import mosaic.losses.protein_mpnn as mpnn_losses

from mosaic_utils import log, load_mpnn


# ============================================================================
# Loss registry
# ============================================================================

STRUCTURE_LOSSES: Dict[str, type] = {
    "PLDDTLoss": sp.PLDDTLoss,
    "WithinBinderContact": sp.WithinBinderContact,
    "HelixLoss": sp.HelixLoss,
    "BinderPTMLoss": sp.BinderPTMLoss,
    "WithinBinderPAE": sp.WithinBinderPAE,
    "ActualRadiusOfGyration": sp.ActualRadiusOfGyration,
    "ColabDesignContactLoss": sp.ColabDesignContactLoss,
    "DistogramRadiusOfGyration": sp.DistogramRadiusOfGyration,
    "MAERadiusOfGyration": sp.MAERadiusOfGyration,
}

BINDING_LOSSES: Dict[str, type] = {
    "BinderTargetContact": sp.BinderTargetContact,
    "BinderTargetPAE": sp.BinderTargetPAE,
    "TargetBinderPAE": sp.TargetBinderPAE,
    "IPTMLoss": sp.IPTMLoss,
    "BinderTargetIPTM": sp.BinderTargetIPTM,
    "BinderTargetIPSAE": sp.BinderTargetIPSAE,
    "TargetBinderIPSAE": sp.TargetBinderIPSAE,
    "IPSAE_min": sp.IPSAE_min,
    "pTMEnergy": sp.pTMEnergy,
}

SCAFFOLDING_LOSSES: Dict[str, type] = {
    "FAPE": is_.FAPE,
    "RMSD": is_.RMSD,
    "DistogramCCE": is_.DistogramCCE,
    "ScaffoldPLDDT": is_.ScaffoldPLDDT,
    "UnindexedRMSD": us.UnindexedRMSD,
    "UnindexedDistogramCCE": us.UnindexedDistogramCCE,
    "CollisionPenalty": us.CollisionPenalty,
    "GeometricSearch": us.GeometricSearch,
}

MPNN_LOSSES: Dict[str, type] = {
    "ProteinMPNNLoss": mpnn_losses.ProteinMPNNLoss,
    "InverseFoldingSequenceRecovery": mpnn_losses.InverseFoldingSequenceRecovery,
    "AllResiduePLLLoss": mpnn_losses.AllResiduePLLLoss,
}

LOSS_REGISTRY: Dict[str, type] = {**STRUCTURE_LOSSES, **BINDING_LOSSES, **SCAFFOLDING_LOSSES, **MPNN_LOSSES}


# ============================================================================
# DesignSpace
# ============================================================================


class DesignSpace:
    """Encapsulates the fixed/variable position split for binder design.

    Always exists — for free design (no scaffold) every position is variable
    and wrap_loss / expand_pssm are identity. The runner calls through
    DesignSpace uniformly without branching on scaffold presence.

    Three modes:
      - No scaffold: all positions variable, no motif constraints
      - Indexed scaffold (loops specified): motif positions known, PSSM
        optimized over variable positions only via SetPositions wrapping
      - Unindexed scaffold (no loops): motif positions unknown, PSSM
        covers full design length, an assignment matrix (N, M) maps
        positions to motif residues during optimization
    """

    def __init__(
        self,
        design_length: int,
        scaffold_cfg: Optional[Dict] = None,
        allow_cys: bool = True,
    ):
        self.design_length = design_length
        self.allow_cys = allow_cys
        self.n_aa = 20 if allow_cys else 19
        self.has_scaffold = False
        self.is_indexed = False
        self._scaffold = None
        self._scaffold_cfg: Optional[Dict] = None
        self._motif_positions: Optional[List[int]] = None
        self._theozyme_residues: Optional[List[str]] = None
        self.sequence = "X" * design_length

        if scaffold_cfg is not None:
            scaffold_pdb = Path(scaffold_cfg["pdb_path"])
            if not scaffold_pdb.exists():
                raise FileNotFoundError(f"scaffold pdb not found: {scaffold_pdb}")

            self.has_scaffold = True
            self._scaffold_cfg = scaffold_cfg
            keep_intervals = scaffold_cfg.get("keep_intervals")
            if keep_intervals is None:
                raise ValueError("keep_intervals is required for scaffolding")

            if "loops" in scaffold_cfg:
                scaffold = is_.IndexedScaffold(
                    path_to_structure=scaffold_pdb,
                    length=design_length,
                    keep_intervals=keep_intervals,
                    order=scaffold_cfg.get("order"),
                    loops=scaffold_cfg["loops"],
                )
                self.is_indexed = True
                self._scaffold = scaffold
                self.sequence = scaffold.sequence
                mask = scaffold.mask
                self._motif_positions = [int(i) + 1 for i, m in enumerate(mask) if m]
            else:
                scaffold = us.UnindexedScaffold(
                    path_to_structure=scaffold_pdb,
                    keep_intervals=keep_intervals,
                    length=design_length,
                )
                self._scaffold = scaffold

            self._theozyme_residues = scaffold._residue_labels

            name = scaffold_cfg.get("name", scaffold_pdb.stem)
            log(f"Scaffold '{name}': length={len(scaffold)}, pdb_path={scaffold_pdb}, "
                f"indexed={self.is_indexed}, n_motif={scaffold.n_motif if not self.is_indexed else int(mask.sum())}")

        self.n_variable = self.sequence.count("X")
        self._wildtype = jnp.array([TOKENS.index(aa) if aa != "X" else -1 for aa in self.sequence])
        self._variable_positions = jnp.array([i for i, aa in enumerate(self.sequence) if aa == "X"])

    @property
    def n_motif(self) -> int:
        if self._scaffold is None:
            return 0
        if self.is_indexed:
            return int(self._scaffold.mask.sum())
        return self._scaffold.n_motif

    def wrap_loss(self, loss: Loss) -> Loss:
        if self.is_indexed:
            loss = SetPositions(self._wildtype, self._variable_positions, loss)
        if not self.allow_cys:
            loss = NoCys(loss)
        return loss

    def expand_pssm(self, pssm: Array, loss: Loss) -> Array:
        if not self.allow_cys:
            pssm = NoCys.sequence(pssm)
            loss = loss.loss
        if self.is_indexed:
            pssm = loss.sequence(pssm)
        return pssm

    def motif_metadata(self, assignment=None) -> Optional[Dict[str, int]]:
        if not self.has_scaffold:
            return None
        if self.is_indexed:
            return {res: pos for res, pos in zip(self._theozyme_residues, self._motif_positions)}
        positions = jnp.argmax(assignment, axis=0)
        return {res: int(positions[m]) for m, res in enumerate(self._theozyme_residues)}


# ============================================================================
# Loss building
# ============================================================================


def build_loss(
    loss_terms: List[Dict],
    prefix: Optional[str] = None,
    scaffold: Optional[is_.IndexedScaffold | us.UnindexedScaffold] = None,
    allowed_terms: Optional[set] = None,
) -> tuple[List[LossTerm], Dict[str, Any]]:
    terms: List[LossTerm] = []
    weight_schedules: Dict[str, Any] = {}

    for term_cfg in loss_terms:
        term_name = term_cfg["term"]

        if term_name not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss term: {term_name}. Available: {sorted(LOSS_REGISTRY)}")

        if allowed_terms is not None and term_name not in allowed_terms:
            raise ValueError(f"Loss term '{term_name}' not allowed in this context type. Available: {sorted(allowed_terms)}")

        kwargs = {k: v for k, v in term_cfg.items() if k not in ("term", "weight")}
        term_suffix = str(kwargs.pop("name", "")) or LOSS_REGISTRY[term_name].name
        full_name = f"{prefix}_{term_suffix}" if prefix else term_suffix
        kwargs["name"] = full_name

        if term_name in MPNN_LOSSES:
            mpnn = load_mpnn(kwargs.pop("mpnn_variant", "default"), kwargs.pop("backbone_noise", 0.0))
            if scaffold is not None:
                kwargs.setdefault("scaffold_mask", scaffold.mask)
            term = LOSS_REGISTRY[term_name](mpnn=mpnn, **kwargs)
        elif term_name in SCAFFOLDING_LOSSES:
            term = LOSS_REGISTRY[term_name].from_scaffold(scaffold=scaffold, **kwargs)
        else:
            term = LOSS_REGISTRY[term_name](**kwargs)

        weight_schedules[full_name] = term_cfg.get("weight", 1.0)
        terms.append(term)

    return terms, weight_schedules


# ============================================================================
# Target & context building
# ============================================================================


@dataclass
class ContextSpec:
    name: str
    pdb_path: Optional[str]
    target_chains: List[TargetChain]
    loss_terms: List[LossTerm] = field(default_factory=list)
    weight_schedules: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_target_chains(self) -> int:
        return len(self.target_chains)


def _build_target_chains(cfg: Dict) -> List[TargetChain]:
    target_pdb = Path(cfg["pdb_path"])
    structure = gemmi.read_structure(str(target_pdb))
    structure.remove_ligands_and_waters()
    use_msa = bool(cfg.get("use_msa", False))

    return [
        TargetChain(
            sequence=gemmi.one_letter_code([r.name for r in chain]),
            use_msa=use_msa,
            template_chain=chain,
        )
        for chain in structure[0]
    ]


def _build_context(
    design_space: DesignSpace,
    shared_loss_terms: List[Dict],
    target_cfg: Optional[Dict] = None,
) -> ContextSpec:
    if target_cfg is not None:
        pdb_path = Path(target_cfg["pdb_path"])
        if not pdb_path.exists():
            raise FileNotFoundError(f"target pdb not found: {pdb_path}")
        context_name = target_cfg.get("name", pdb_path.stem)
        target_chains = _build_target_chains(target_cfg)
        target_terms, target_ws = build_loss(
            loss_terms=target_cfg.get("loss_terms", []),
            prefix=context_name,
            allowed_terms=BINDING_LOSSES,
        )
    else:
        context_name = "scaffold"
        pdb_path = None
        target_chains = []
        target_terms, target_ws = [], {}

    scaffold_terms, scaffold_ws = [], {}
    if design_space.has_scaffold:
        scaffold_name = design_space._scaffold_cfg.get("name", Path(design_space._scaffold_cfg["pdb_path"]).stem)
        scaffold_terms, scaffold_ws = build_loss(
            loss_terms=design_space._scaffold_cfg.get("loss_terms", []),
            prefix=f"{context_name}_{scaffold_name}",
            scaffold=design_space._scaffold,
            allowed_terms=SCAFFOLDING_LOSSES,
        )

    shared_terms, shared_ws = build_loss(
        loss_terms=shared_loss_terms,
        prefix=context_name,
        scaffold=design_space._scaffold,
    )

    all_terms = target_terms + scaffold_terms + shared_terms
    if not all_terms:
        raise ValueError(f"Context '{context_name}' has no loss terms")

    ctx_weight_schedules: Dict[str, Any] = {}
    ctx_weight_schedules.update(target_ws)
    ctx_weight_schedules.update(scaffold_ws)
    ctx_weight_schedules.update(shared_ws)

    log(
        f"Context '{context_name}': target_chains={len(target_chains)}, pdb_path={pdb_path}"
    )
    return ContextSpec(
        name=context_name,
        pdb_path=pdb_path,
        target_chains=target_chains,
        loss_terms=all_terms,
        weight_schedules=ctx_weight_schedules,
    )


def build_contexts(
    target_cfgs: List[Dict],
    design_space: DesignSpace,
    shared_loss_terms: Optional[List[Dict]] = None,
) -> List[ContextSpec]:
    shared = shared_loss_terms or []

    if target_cfgs:
        log(f"Building {len(target_cfgs)} target context(s)")
        specs = []
        for target_cfg in target_cfgs:
            specs.append(_build_context(design_space, shared, target_cfg))
        return specs

    elif design_space.has_scaffold:
        log("Scaffold-only mode (no targets)")
        return [_build_context(design_space, shared)]

    else:
        raise ValueError("Config must specify at least 'targets' or 'scaffolds'")
