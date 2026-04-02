import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")

with app.setup:
    import jax

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mosaic.optimizers import simplex_APGM, gradient_MCMC
    import mosaic.losses.structure_prediction as sp
    from mosaic.models.boltz1 import Boltz1

    from mosaic.common import TOKENS
    from mosaic.losses.transformations import SoftClip
    from mosaic.notebook_utils import pdb_viewer
    from jaxtyping import Float, Array
    from mosaic.common import LossTerm
    from mosaic.structure_prediction import TargetChain
    from mosaic.models.af2 import AlphaFold2
    from mosaic.proteinmpnn.mpnn import ProteinMPNN


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Warning**
    1. You'll almost certainly need a GPU or TPU
    2. Because JAX uses JIT compilation the first execution of a cell may take quite a while
    3. You might have to run these optimization methods multiple times before you get a reasonable binder
    4. If you wanted to, you could certainly find better hyperparameters for these examples (for faster or better optimization)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Basic example with Boltz1
    """)
    return


@app.cell
def _():
    boltz1 = Boltz1()
    return (boltz1,)


@app.cell
def _():
    target_sequence = "SFPASVQLHTAVEMHHWCIPFSVDGQPAPSLRWLFNGSVLNETSFIFTEFLEPAANETVRHGCLRLNQPTHVNNGNYTLLAANPFGQASASIMAAF"
    return (target_sequence,)


@app.cell
def _(boltz1):
    def predict(sequence, features, writer):
        pred = boltz1.predict(PSSM = sequence, features=features, writer=writer, key = jax.random.key(11))
        return pred, pdb_viewer(pred.st)

    return (predict,)


@app.cell
def _():
    binder_length = 55
    return (binder_length,)


@app.cell
def _(binder_length, boltz1, target_sequence):
    boltz_features, boltz_writer = boltz1.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(sequence=target_sequence)],
    )
    return boltz_features, boltz_writer


@app.cell(hide_code=True)
def _():
    mo.md("""
    First let's define a simple loss function to optimize.
    """)
    return


@app.cell
def _(boltz1, boltz_features):
    loss = boltz1.build_loss(
        loss=2 * sp.BinderTargetContact() + sp.WithinBinderContact(),
        features=boltz_features,
        recycling_steps=1,
    )
    return (loss,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    Now we run an optimizer -- in this case an accelerated proximal gradient method -- to get an initial solution

    We perform a three phase optimization with increasing sharpening optimizers (pushing from a dense pssm to a sparse/onehot pssm)
    """)
    return


@app.cell
def _(binder_length, loss):
    # we can sharpen these logits using weight decay, i.e. increasing scale (which is equivalent to adding entropic regularization)
    pssm_init = 0.5 * jax.random.gumbel(
        key=jax.random.key(np.random.randint(100000)),
        shape=(binder_length, 20),
    )

    _, pssm_soft = simplex_APGM(
        loss_function=loss,
        x=pssm_init,
        n_steps=100,
        stepsize=0.1,
        scale=1.0,
        momentum=0.9,
    )

    _, pssm_intermediate = simplex_APGM(
        loss_function=loss,
        x=pssm_soft,
        n_steps=25,
        stepsize=0.2,
        scale=1.1,
        momentum=0.9,
    )

    pssm_sharp, _ = simplex_APGM(
        loss_function=loss,
        x=pssm_intermediate, 
        n_steps=25,
        stepsize=0.2,
        scale=1.5,
        momentum=0.0,
    )
    return pssm_sharp, pssm_soft


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_soft):
    soft_output, _viewer = predict(
        pssm_soft, boltz_features, boltz_writer
    )
    _viewer
    return (soft_output,)


@app.cell
def _(pssm_soft, soft_output, visualize_output):
    visualize_output(soft_output, pssm_soft)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    After the first phase, the pssm already looks pretty good (usually), but it isn't a single sequence (check out the PSSM above)!
    """)
    return


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_sharp):
    sharp_outputs, _viewer = predict(
        pssm_sharp, boltz_features, boltz_writer
    )
    _viewer
    return (sharp_outputs,)


@app.cell
def _(pssm_sharp, sharp_outputs, visualize_output):
    visualize_output(sharp_outputs, pssm_sharp)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    After sharpening, hopefully the pssm still looks pretty good and is now a single sequence!

    Lastly let's repredict using AlphaFoldMultimer
    """)
    return


@app.cell
def _(af2, af_features, pssm_sharp):
    _o_af_repredict = af2.predict(features=af_features, PSSM = pssm_sharp, key = jax.random.key(12))
    print(_o_af_repredict.iptm)
    pdb_viewer(_o_af_repredict.st)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. AF2 optimization of a 7S5B-scaffolded binder
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    Okay, that was fun but let's do a something a little more complicated: we'll use AlphaFold2 (instead of Boltz) to design a binder that adheres to a specified fold. [7S5B](https://www.rcsb.org/structure/7S5B) is a denovo triple-helix bundle originally designed to bind IL-7r; let's see if we can find a sequence _with the same fold_ that AF thinks will bind to our target instead.
    """)
    return


@app.cell
def _():
    af2 = AlphaFold2()
    return (af2,)


@app.cell
def _():
    scaffold_sequence = "SVIEKLRKLEKQARKQGDEVLVMLARMVLEYLEKGWVSEEDADESADRIEEVLKK"
    return (scaffold_sequence,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    We also need the structure to use as scaffold, we'll predict it alone using AF2 (we could use the crystal structure instead but this works fine).
    """)
    return


@app.cell
def _(af2, scaffold_sequence):
    _scaffold_features, _= af2.target_only_features(chains = [TargetChain(sequence=scaffold_sequence, use_msa = False)])

    o_af_scaffold = af2.predict(
        features = _scaffold_features,
        recycling_steps = 3,
        key=jax.random.key(0),
        writer = None
    )

    af_scaffold_logits = af2.model_output(
         features = _scaffold_features,
        recycling_steps = 1,
        key=jax.random.key(0),
    ).distogram_logits

    pdb_viewer(o_af_scaffold.st)
    return af_scaffold_logits, o_af_scaffold


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To enforce the scaffolding during optimization, we will use two loss terms:
    1. The log-likelihood of our sequence according to ProteinMPNN applied to the scaffold structure
    """)
    return


@app.cell
def _(o_af_scaffold):
    from mosaic.losses.protein_mpnn import (
        FixedStructureInverseFoldingLL,
    )
    # Create inverse folding LL term
    scaffold_inverse_folding_LL = FixedStructureInverseFoldingLL.from_structure(
        name="scaffold_mpnn_ll",
        st=o_af_scaffold.st,
        mpnn=ProteinMPNN.from_pretrained(),
    )
    return (scaffold_inverse_folding_LL,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    2. Cross-entropy between the predicted distogram of our sequence and the original 7S5B sequence
    """)
    return


@app.cell
def _(af_scaffold_logits):
    distogramCE = sp.DistogramCE(
                    jax.nn.softmax(af_scaffold_logits),
                    name="scaffoldCE",
                )
    return (distogramCE,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    Let's also add a loss term that penalizes cysteines to show that implementing new loss terms is trivial.
    """)
    return


@app.class_definition
class NoCysteine(LossTerm):
    def __call__(self, seq: Float[Array, "N 20"], *, key):
        p_cys = seq[:, TOKENS.index("C")].sum()
        return p_cys, {"p_cys": p_cys}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We predict first the target alone with Boltz1, and use the prediction as template during optimization with AF2
    """)
    return


@app.cell
def _(af2, binder_length, boltz1, target_sequence):
    # Target only prediction
    target_features, target_writer = boltz1.target_only_features(chains = [TargetChain(sequence = target_sequence)])
    o_target = boltz1.predict(features = target_features, writer = target_writer, key = jax.random.key(0))
    target_st = o_target.st
    viewer_target = pdb_viewer(target_st)
    viewer_target

    # Define features using the target prediction as template
    af_features, _ = af2.binder_features(
        binder_length=binder_length,
        chains=[
            TargetChain(
                target_sequence, use_msa=False, template_chain=target_st[0][0]
            )
        ],
    )
    return (af_features,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define the loss to optimize with out custom losses (note how easy it is to modify loss terms, such as clipping them)
    """)
    return


@app.cell
def _(af2, af_features, distogramCE, scaffold_inverse_folding_LL):
    af_loss = (
        af2.build_loss(
            loss=1.0 * sp.PLDDTLoss()
            + 1 * sp.BinderTargetContact()
            + 0.05 * sp.TargetBinderPAE()
            + 0.05 * sp.BinderTargetPAE()
            + 0.025 * sp.IPTMLoss()
            + 0.4 * sp.WithinBinderPAE()
            + 1.0 * sp.WithinBinderContact()
            + 2.5*SoftClip(distogramCE, 2.5, 3),
            features=af_features,
        )
        + 1.0*SoftClip(scaffold_inverse_folding_LL, 2.5, 3.0)
        + NoCysteine()
    )
    return (af_loss,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Lastly define the optimizer and run the optimization
    """)
    return


@app.cell
def _(af_loss, binder_length):
    pssm_init_af = 0.5 * jax.random.gumbel(
        key=jax.random.key(np.random.randint(100000)),
        shape=(binder_length, 20),
    )

    _, pssm_soft_af = simplex_APGM(
        loss_function=af_loss,
        x=pssm_init_af,
        n_steps=100,
        stepsize=0.1,
        scale=1.0,
        momentum=0.0,
        serial_evaluation=True,
    )

    pssm_sharp_af, _ = simplex_APGM(
        loss_function=af_loss,
        x=pssm_soft_af,
        n_steps=25,
        stepsize=0.2,
        scale=1.5,
        momentum=0.0,
        serial_evaluation=True,
    )
    return (pssm_sharp_af,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Predict and visualize the optimized structure with both Boltz1 and AF2
    """)
    return


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_sharp_af):
    boltz_output, _viewer = predict(pssm_sharp_af, boltz_features, boltz_writer)
    _viewer
    return (boltz_output,)


@app.cell
def _(boltz_output, pssm_sharp_af, visualize_output):
    visualize_output(boltz_output, pssm_sharp_af)
    return


@app.cell
def _(af2, af_features, pssm_sharp_af):
    af_o = af2.predict(PSSM = pssm_sharp_af, features=af_features,key = jax.random.key(1))
    pdb_viewer(af_o.st)
    return (af_o,)


@app.cell
def _(af_o, pssm_sharp_af, visualize_output):
    visualize_output(af_o, pssm_sharp_af)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For fun (and to show how easy it is to use different optimization algorithms) let's try polishing this design using gradient-assisted MCMC
    """)
    return


@app.cell
def _(af_loss, pssm_sharp_af):
    seq_mcmc = gradient_MCMC(
        af_loss,
        jax.device_put(pssm_sharp_af.argmax(-1)),
        temp=0.001,
        proposal_temp=0.00001,
        steps=100,
        fix_loss_key=False,
        serial_evaluation=True
    )
    return (seq_mcmc,)


@app.cell
def _(seq_mcmc):
    plt.imshow(jax.nn.one_hot(seq_mcmc, 20))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare the MCMC sequence with the starting sequence
    """)
    return


@app.cell
def _(pssm_sharp_af):
    "".join([TOKENS[i] for i in pssm_sharp_af.argmax(-1)])
    return


@app.cell
def _(seq_mcmc):
    "".join([TOKENS[i] for i in seq_mcmc])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Predict the MCMC output with AF2 and save the structure
    """)
    return


@app.cell
def _(af2, af_features, seq_mcmc):
    af_o_mcmc = af2.predict(PSSM = jax.nn.one_hot(seq_mcmc, 20), features=af_features,key = jax.random.key(4))
    print(af_o_mcmc.iptm)
    plt.imshow(af_o_mcmc.pae)
    return (af_o_mcmc,)


@app.cell
def _(af_o_mcmc):
    pdb_viewer(af_o_mcmc.st)
    return


@app.cell
def _(af_o_mcmc):
    mo.download(
        af_o_mcmc.st.make_pdb_string(),
        filename="mcmc.pdb",
        label="AF2 predicted complex",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. AF2 + Boltz1 joint optimization
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    As a final example let's try optimizing the *sum* of these loss terms; so we're calling both AF2 and Boltz1 at every iteration. In `mosaic` this is trivial.
    """)
    return


@app.cell
def _(af_loss, binder_length, loss):
    pssm_init_both = 0.5 * jax.random.gumbel(
        key=jax.random.key(np.random.randint(100000)),
        shape=(binder_length, 20),
    )
    _, pssm_both = simplex_APGM(
        loss_function=af_loss + loss,
        x=pssm_init_both,
        n_steps=150,
        stepsize=0.15,
        momentum=0.0,
        serial_evaluation=True
    )
    return (pssm_both,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Predict with Boltz1 and visualize the outputs
    """)
    return


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_both):
    both_output, _viewer_both = predict(
        pssm_both, boltz_features, boltz_writer
    )
    _viewer_both
    return (both_output,)


@app.cell
def _(both_output, pssm_both):
    def visualize_output(outputs, pssm):
        _f = plt.figure(dpi=125)
        plt.imshow(outputs.pae)
        plt.title("PAE")
        plt.colorbar()

        _g = plt.figure(dpi=125)
        plt.plot(outputs.plddt)
        plt.title("pLDDT")
        plt.vlines([pssm.shape[0]], 0, 1, color="red", linestyles="--")

        _h = plt.figure(dpi=125)
        plt.imshow(pssm)
        plt.xlabel("Amino acid")
        plt.ylabel("Sequence position")

        return mo.ui.tabs({"PAE": _f, "pLDDT": _g, "PSSM": _h})

    visualize_output(both_output, pssm_both)
    return (visualize_output,)


if __name__ == "__main__":
    app.run()
