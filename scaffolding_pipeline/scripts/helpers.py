from mosaic.optimizers import PSSMOptimizer, SimplexAPGM, LogitAPGM

from mosaic.models.af2 import AlphaFold2
from mosaic.models.protenix import Protenix2025, ProtenixBase, ProtenixMini, ProtenixTiny
from mosaic.models.boltz1 import Boltz1
from mosaic.models.boltz2 import Boltz2

from mosaic.common import LinearCombination, LossTerm
import mosaic.losses.structure_prediction as sp
import mosaic.losses.motif_scaffolding as ms
import numpy as np


OPTIMIZERS = {
    "SimplexAPGM": SimplexAPGM,
    "LogitAPGM": LogitAPGM,
}

MODELS = {
    # "AlphaFold2": AlphaFold2, # need call to build_loss instead of multisample loss
    "Protenix2025": Protenix2025,
    "ProtenixBase": ProtenixBase,
    "ProtenixMini": ProtenixMini,
    "ProtenixTiny": ProtenixTiny,
    # "Boltz1": Boltz1, # need call to build_loss instead of multisample loss
    "Boltz2": Boltz2,
}

STRUCTURE_LOSSES = {
    "PLDDTLoss": sp.PLDDTLoss,
    "WithinBinderPAE": sp.WithinBinderPAE,
    "WithinBinderContact": sp.WithinBinderContact,
    "BinderPTMLoss": sp.BinderPTMLoss,
    "DistogramRadiusOfGyration": sp.DistogramRadiusOfGyration,
    "HelixLoss": sp.HelixLoss,
}

SCAFFOLDING_LOSSES = {
    "DistogramCCE": ms.DistogramCCE,
    "RMSD": ms.RMSD,
    "FAPE": ms.FAPE,
}


def build_loss(loss_terms: dict[str, float], scaffold: ms.Scaffold) -> LossTerm:
    losses = []
    for name, weight in loss_terms.items():
        if weight != 0:
            if name in STRUCTURE_LOSSES:
                losses.append(weight * STRUCTURE_LOSSES[name]())
            elif name in SCAFFOLDING_LOSSES:
                losses.append(weight * SCAFFOLDING_LOSSES[name].from_scaffold(scaffold))
            else:
                raise ValueError(f"Unsupported loss term '{name}', must be one of "
                                f"{list(STRUCTURE_LOSSES) + list(SCAFFOLDING_LOSSES)}")
        else:
            print(f"Ignoring loss term '{name}', weight is 0")

    if not losses:
        raise ValueError("All loss terms have weight 0, no loss to optimize")
    loss = losses[0]
    for l in losses[1:]:
        loss += l
    return loss


def build_optimizer(
    opt_params: dict[str, float | str],
    loss_fn: LinearCombination | LossTerm,
    scaffold_len: int,
) -> PSSMOptimizer:
    opt_str = opt_params["type"]
    if opt_str not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{opt_str}', must be one of {list(OPTIMIZERS.keys())}")
 
    kwargs = {k: v for k, v in opt_params.items() if k not in ("type", "step_size")}
    kwargs["stepsize"] = opt_params["step_size"] * np.sqrt(scaffold_len)
 
    return OPTIMIZERS[opt_str](loss_fn=loss_fn, **kwargs)