import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from mosaic.optimizers import simplex_APGM
    import mosaic.losses.structure_prediction as sp
    import matplotlib.pyplot as plt
    import jax
    import numpy as np
    import gemmi
    from mosaic.notebook_utils import pdb_viewer
    from mosaic.losses.protein_mpnn import (
        InverseFoldingSequenceRecovery,
    )
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    import importlib
    import mosaic
    import equinox as eqx

    import jax.numpy as jnp
    from protenix.protenij import TrunkEmbedding
    from mosaic.structure_prediction import TargetChain
    from mosaic.models.protenix import Protenix2025
    from mosaic.proteinmpnn.mpnn import load_abmpnn
    from mosaic.losses.ablang import AbLangPseudoLikelihood, load_ablang
    from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc
    from mosaic.losses.transformations import SetPositions


@app.cell
def _():
    from mosaic.common import TOKENS

    return (TOKENS,)


@app.cell
def _():
    mo.callout("Demo VHH CDR design using Protenix v1.0", kind="success")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    We want to use Protenix2025 to design a vhh against PDL1; the world will never have enough de novo binders to PDL1.
    """)
    return


@app.cell
def _():
    protenix = Protenix2025()
    return (protenix,)


@app.cell
def _():
    target_structure = gemmi.read_structure("PDL1.pdb")
    target_sequence = gemmi.one_letter_code(
        [r.name for r in target_structure[0][0]]
    )
    return target_sequence, target_structure


@app.cell
def _(target_structure):
    pdb_viewer(target_structure)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Given that we are working with a VHH, the optimization is only done for the CDRs (which determines the binding)
    """)
    return


@app.cell
def _():
    masked_framework_sequence = "QVQLVESGGGLVQPGGSLRLSCAASXXXXXXXXXXXLGWFRQAPGQGLEAVAAXXXXXXXXYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCXXXXXXXXXXXXXXXXXXWGQGTLVTVS"
    return (masked_framework_sequence,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Mosaic supports ESMC (a large protein model focused on biologically meaningful representation) and AbLang (a large protein model specifically trained on Ab sequences).

    Even though not strictly necessary, including the PLLs of both models in our loss function, we guide the optimization towards more plausible and more antibody-like sequences.
    """)
    return


@app.cell
def _():
    ablang, ablang_tokenizer = load_ablang("heavy")
    ablang_pll = AbLangPseudoLikelihood(
        model=ablang,
        tokenizer=ablang_tokenizer,
        stop_grad=True,
    )

    ESMC_pll = ESMCPseudoLikelihood(load_esmc("esmc_300m"), stop_grad=True)
    return ESMC_pll, ablang_pll


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As of now, if binder_features is used to initialize features,  the sidechains of the binder will not be predicted.

    Instead, we'll use target_only_features so we properly handle sidechains on the framework. this shouldn't make a huge difference but feels right.
    """)
    return


@app.cell
def _(masked_framework_sequence, protenix, target_sequence, target_structure):
    design_features, design_structure = protenix.target_only_features(
        chains=[
            TargetChain(
                masked_framework_sequence,
                use_msa=True,
            ),
            TargetChain(
                target_sequence,
                use_msa=True,
                template_chain=target_structure[0][0],
            ),
        ],
    )
    return design_features, design_structure


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now let's build up the loss function which will contain as main loss, the BinderTargetContact.

    If we really don't like "sidebinders" and want contact ONLY with the CDRs we could add a **negative** BinderTargetContact term here that only applies to framework residues.

    I set this to zero because it seems silly to me: plenty of natural VHHs have framework-target contacts to really discourage these kinds of poses you'd also need to downweight some of the terms below that prefer secondary structure within the binder (WithinBinderPAE, pLDDT, etc)
    """)
    return


@app.cell
def _(design_features, masked_framework_sequence, protenix):
    structure_loss = (
        # encourage binding with the CDRs
        sp.BinderTargetContact(
            paratope_idx=np.array(
                [
                    i for (i, c) in enumerate(masked_framework_sequence) if c == "X"
                ]  
            )
            # if you have a particular hotspot you're going for you could use `epitope_idx` here.
        )
        # discourage binding with the framework
        - 0.0 * sp.BinderTargetContact(
            paratope_idx=np.array(
                [
                    i for (i, c) in enumerate(masked_framework_sequence) if c != "X"
                ]  
            )
        )
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
    )

    model_loss = protenix.build_multisample_loss(
        loss=structure_loss,
        features=design_features,
        recycling_steps=2,
        sampling_steps=20,
    )
    return (model_loss,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As for now the loss would be computed on the entire sequence (including the framework regions)... Mosaic provides a `SetPositions` wrapper for losses, that can be built from a sequence and only computes gradients with respect to positions initialized with the unknown amino acid `X`.
    """)
    return


@app.cell
def _(ESMC_pll, ablang_pll, masked_framework_sequence, model_loss):
    loss = SetPositions.from_sequence(
        wildtype=masked_framework_sequence,
        loss=0.1 * ESMC_pll + 2 * model_loss + 0.1 * ablang_pll,
    )
    return (loss,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now let's define and run a multiphase optimizer on the CDRs.

    JIT will take a very long time the first time we run the following cell. Rerun for more samples!
    """)
    return


@app.cell
def _(masked_framework_sequence):
    num_designed_residues = len([c for c in masked_framework_sequence if c == "X"])
    pssm_init = 0.5 * jax.random.gumbel(
        key=jax.random.key(np.random.randint(1000000)),
        shape=(num_designed_residues, 20),
    )
    return (pssm_init,)


@app.cell
def _(TOKENS, loss, pssm_init):
    _, pssm_soft = simplex_APGM(
        loss_function=loss,
        x=pssm_init,
        n_steps=50,
        stepsize=1.5 * np.sqrt(pssm_init.shape[0]),
        momentum=0.2,
        scale=1.0,
        max_gradient_norm=1.0,
        logspace=True,
    )

    _, pssm_sharp = simplex_APGM(
        loss_function=loss,
        x=pssm_soft,
        n_steps=30,
        stepsize=0.5 * np.sqrt(pssm_init.shape[0]),
        momentum=0.0,
        scale=1.1,
        max_gradient_norm=1.0,
    )

    _, pssm_partial = simplex_APGM(
        loss_function=loss,
        x=pssm_sharp,
        n_steps=30,
        stepsize=0.25 * np.sqrt(pssm_init.shape[0]),
        momentum=0.0,
        scale=1.1,
        max_gradient_norm=1.0,
        logspace=True,
    )

    print("".join(TOKENS[i] for i in pssm_partial.argmax(-1)))
    return (pssm_partial,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's try to improve our design using MCMC
    """)
    return


@app.cell
def _():
    from mosaic.optimizers import gradient_MCMC

    return (gradient_MCMC,)


@app.cell
def _(gradient_MCMC, loss, pssm_partial):
    s_mcmc = gradient_MCMC(
        loss=loss,
        sequence=jax.device_put(pssm_partial.argmax(-1)),
        steps=30,
        fix_loss_key=False,
        proposal_temp=1e-5,
        max_path_length=1,
    )
    return (s_mcmc,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We have been optimizing only the pssm of the CDRs. It's very important we add the framework residues back into our sequence before e.g. repredicting!

    This can be done using the `loss.sequence()` function of our `SetPositions` wrapped loss.
    """)
    return


@app.cell
def _(loss, s_mcmc):
    final_pssm = loss.sequence(jax.nn.one_hot(s_mcmc, 20))
    return (final_pssm,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Lastly, repredict the full vhh with optimized CDRs in complex with the target.
    """)
    return


@app.cell
def _(TOKENS, design_features, design_structure, final_pssm, protenix):
    prediction_inpaint = protenix.predict(
        PSSM=final_pssm,
        writer=design_structure,
        features=design_features,
        recycling_steps=10,
        key=jax.random.key(np.random.randint(10000)),
    )

    design_str = "".join(TOKENS[i] for i in final_pssm.argmax(-1))
    print(f"Final sequence: {design_str}")
    return (prediction_inpaint,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Visualize some prediction infos and save the predicted structure
    """)
    return


@app.cell
def _(prediction_inpaint):
    plt.imshow(prediction_inpaint.pae)
    plt.title(f"Predicted Aligned Error, IPTM {prediction_inpaint.iptm: 0.3f}")
    return


@app.cell
def _(masked_framework_sequence, prediction_inpaint):
    _f = plt.figure()
    plt.plot(prediction_inpaint.plddt)
    plt.title("pLDDT")
    plt.vlines(
        [len(masked_framework_sequence)],
        ymin=prediction_inpaint.plddt.min(),
        ymax=prediction_inpaint.plddt.max(),
        linestyles="dashed",
        color="red",
    )
    _f
    return


@app.cell
def _(prediction_inpaint):
    pdb_viewer(prediction_inpaint.st)
    return


@app.cell
def _(prediction_inpaint):
    mo.download(data=prediction_inpaint.st.make_pdb_string(), filename="vhh.pdb")
    return


if __name__ == "__main__":
    app.run()
