import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import marimo as mo

    import matplotlib.pyplot as plt
    from mosaic.optimizers import simplex_APGM
    from mosaic.common import TOKENS, LossTerm
    import numpy as np

    from mosaic.notebook_utils import pdb_viewer
    import mosaic.losses.structure_prediction as sp
    from mosaic.losses.protein_mpnn import FixedStructureInverseFoldingLL, InverseFoldingSequenceRecovery

    from mosaic.models.boltz2 import Boltz2
    from mosaic.models.af2 import AlphaFold2
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    from mosaic.structure_prediction import TargetChain


@app.cell
def _():
    mo.md("""
    **Warning**

    1. You'll almost certainly need a GPU or TPU to run this
    2. Because JAX uses JIT compilation the first execution of a cell may take quite a while
    3. You might have to run these optimization methods multiple times before you get a reasonable binder
    4. If you change targets you'll likely have to fiddle with hyperparameters!
    5. This is pretty experimental, I highly recommend you stick with BindCraft if you're designing a minibinder against a protein target
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Boltz2 binder optimization demo
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We will use Boltz2 to optimize the binder sequence, while AF2 will serve for refolding as sanity check that our prediction are plausible.
    """)
    return


@app.cell
def _():
    model_af = AlphaFold2()
    model = Boltz2()
    return model, model_af


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In addition to the models above, we will be using ProteinMPNN as loss term to move the generated sequences towards sequences that AF2-multimer also likes. This is slower because we have to run the Boltz-2 structure module. Try removing it for faster generation!
    """)
    return


@app.cell
def _():
    mpnn = ProteinMPNN.from_pretrained()
    return (mpnn,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define the target and dimension of the binder
    """)
    return


@app.cell
def _():
    binder_length = 75
    target_sequence = "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA" 
    return binder_length, target_sequence


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Create Boltz2 features and structure writer and define the loss (note the inclusion of the inverse folding with ProteinMPNN)
    """)
    return


@app.cell
def _(binder_length, model, target_sequence):
    features, structure_writer = model.binder_features(binder_length=binder_length, chains = [TargetChain(target_sequence)])
    return features, structure_writer


@app.cell
def _(features, model, mpnn):
    loss = model.build_loss(
        loss=2 * sp.BinderTargetContact()
        + sp.WithinBinderContact()
        + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.01)),
        features=features,
    )
    return (loss,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define the optimizer and run optimization on the binder
    """)
    return


@app.cell
def _(binder_length, loss):
    pssm_init = 0.5 * jax.random.gumbel(
        key=jax.random.key(np.random.randint(100000)),
        shape=(binder_length, 20),
    )

    _, pssm_soft = simplex_APGM(
        loss_function=loss,
        x=pssm_init,
        n_steps=75,
        stepsize=0.1,
        scale=1.0,
        momentum=0.0,
    )

    pssm_sharp, _ = simplex_APGM(
        loss_function=loss,
        x=pssm_soft,
        n_steps=25,
        stepsize=0.5,
        scale=1.5,
        momentum=0.0,
    )
    return pssm_sharp, pssm_soft


@app.cell
def _(pssm_sharp):
    binder_seq = "".join(TOKENS[i] for i in pssm_sharp.argmax(-1))
    binder_seq
    return (binder_seq,)


@app.cell
def _(pssm_sharp):
    plt.imshow(pssm_sharp)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Repredict using Boltz2 both the soft and sharp version of the pssm
    """)
    return


@app.cell
def _(model):
    def predict(sequence, features, writer):
        return model.predict(PSSM = sequence, features = features, writer = writer, key = jax.random.key(0))

    return (predict,)


@app.cell
def _(features, predict, pssm_soft, structure_writer):
    soft_pred = predict(
        pssm_soft, features, structure_writer
    )
    pdb_viewer(soft_pred.st)
    return (soft_pred,)


@app.cell
def _(features, predict, pssm_sharp, structure_writer):
    sharp_pred = predict(
        pssm_sharp, features, structure_writer
    )
    pdb_viewer(sharp_pred.st)
    return (sharp_pred,)


@app.cell
def _(sharp_pred, soft_pred):
    print(f"{soft_pred.iptm=}")
    print(f"{sharp_pred.iptm=}")
    return


@app.cell
def _(sharp_pred, soft_pred):
    plt.plot(sharp_pred.plddt)
    plt.plot(soft_pred.plddt)
    return


@app.cell
def _(soft_pred):
    _f = plt.figure()
    plt.imshow(soft_pred.pae)
    plt.colorbar()
    _f
    return


@app.cell
def _(sharp_pred):
    _f = plt.figure()
    plt.imshow(sharp_pred.pae)
    plt.colorbar()
    _f
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Save to disk the sharp structure
    """)
    return


@app.cell
def _(sharp_pred):
    mo.download(data=sharp_pred.st.make_pdb_string(), filename="a.pdb", label = "Boltz-2 predicted complex")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Refold with AF2
    """)
    return


@app.cell
def _():
    mo.md("""
    Make a template structure of the target alone we can use with AF2 multimer
    """)
    return


@app.cell
def _(model, target_sequence):
    template_features, template_writer = model.target_only_features(chains=[TargetChain(sequence=target_sequence)])
    return template_features, template_writer


@app.cell
def _(predict, target_sequence, template_features, template_writer):
    template_st = predict(
        jax.nn.one_hot([TOKENS.index(c) for c in target_sequence], 20),
        template_features,
        template_writer,
    )
    pdb_viewer(template_st.st)
    return (template_st,)


@app.cell
def _(binder_length, model_af, target_sequence, template_st):
    af_features, _ = model_af.binder_features(binder_length=binder_length, chains = [TargetChain(target_sequence, use_msa=False, template_chain=template_st.st[0][0])])
    return (af_features,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Then refold the optimized pssm in complex with the target
    """)
    return


@app.cell
def _(af_features, model_af, pssm_sharp):
    af_pred = model_af.predict(features=af_features, PSSM = pssm_sharp, writer = None, key = jax.random.key(12))
    pdb_viewer(af_pred.st)
    return (af_pred,)


@app.cell
def _(af_pred):
    print(f"{af_pred.iptm=}")
    return


@app.cell
def _(af_pred):
    plt.imshow(af_pred.pae)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Inverse fold of the predicted complex
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's do it live! We can inverse fold the predicted complex using MPNN and the jacobi iteration in a few lines of code.

    At every iteration, to every position is assigned the amino acid that minimizes the loss gradients.
    Note that optimization happens in discrete space (not in continuous space as for the other optimizers).
    """)
    return


@app.cell
def _():
    from mosaic.optimizers import _eval_loss_and_grad as eval_loss_and_grad

    def jacobi(loss, iters, sequence, key):
        for _ in range(iters):
            (v, aux), g = eval_loss_and_grad(loss, jax.nn.one_hot(sequence, 20), key = key)
            sequence = g.argmin(-1)
            print(v)

        return sequence

    return (jacobi,)


@app.cell
def _(mpnn, soft_pred):
    if_ll = FixedStructureInverseFoldingLL.from_structure(
        st=soft_pred.st,
        mpnn=mpnn,
        name="if_ll",
        stop_grad=True,
    )
    return (if_ll,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Add random noise to gradients to avoid getting stuck in the same minima every time (increase diversity)
    """)
    return


@app.class_definition
class GumbelPerturbation(LossTerm):
    key: any

    def __call__(self, sequence, key):
        v = (jax.random.gumbel(self.key, sequence.shape)*sequence).sum()
        return v, {"gumbel": v}


@app.cell
def _(binder_length, if_ll, jacobi):
    seq_mpnn = jacobi(
        loss=if_ll + 0.0005 * GumbelPerturbation(jax.random.key(np.random.randint(1000000))),
        iters=10,
        sequence=np.random.randint(low=0, high=20, size=(binder_length)),
        key=jax.random.key(np.random.randint(1000000)),
    )
    return (seq_mpnn,)


@app.cell
def _(binder_seq, seq_mpnn):
    if_seq = "".join(TOKENS[i] for i in seq_mpnn)
    print(f"Original optimized sequence: {binder_seq}")
    print(f"Inverse folded sequence: {if_seq}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Optimize batch of 10 binders
    """)
    return


@app.cell
def _():
    mo.md("""
    For fun let's design 10 complexes
    """)
    return


@app.cell
def _(binder_length, features, loss, predict, structure_writer):
    def design():
        pssm_init = 0.5 * jax.random.gumbel(
            key=jax.random.key(np.random.randint(100000)),
            shape=(binder_length, 20),
        )
        pssm, _ = simplex_APGM(
            loss_function=loss,
            x=pssm_init,
            n_steps=75,
            stepsize=0.1,
            momentum=0.9,
        )
        prediction = predict(
            pssm, features, structure_writer
        )
        return prediction.st

    return (design,)


@app.cell
def _(design):
    designs = [design() for _ in mo.status.progress_bar(range(10))]
    return (designs,)


@app.cell
def _():
    from mosaic.notebook_utils import gemmi_structure_from_models

    return (gemmi_structure_from_models,)


@app.cell
def _(designs, gemmi_structure_from_models):
    complexes = gemmi_structure_from_models("designs", [st[0] for st in designs])
    return (complexes,)


@app.cell
def _(complexes):
    pdb_viewer(complexes)
    return


if __name__ == "__main__":
    app.run()
