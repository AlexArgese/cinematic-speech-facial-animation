# Cinematic Speech as a Stress Test for Facial Animation Systems

**Author:** Alex Argese — MSc Data Science Engineering (EURECOM) · MSc Digital Media Engineering (Politecnico di Torino)
**Type:** Exploratory portfolio research project
**Original:** April 2026 · **Methodology revision:** November 2026

This project asks whether cinematic dialogue could provide a useful stress test
for expressive speech-driven facial animation. It compares text emotion labels and
explores facial displacement using one VOCASET/FaceFormer mesh pair and five
animated clips. It does not yet evaluate FaceFormer on cinematic audio.

## Methodology corrections

- The 73-dialogue DCC contains Frozen 12, Tangled 10, Coco 12, Encanto 12, Up 11,
  Inside Out 8 and Moana 8. The old paper's Tangled/Moana counts were incorrect.
- DCC motion was extracted with MediaPipe, rather than estimated from emotional
  labels as the old methods text suggested.
- FLAME and MediaPipe use different regions, coordinates and preprocessing. The
  earlier **“8× cinematic motion demand” claim is withdrawn as a calibrated effect
  size**. Dimensionless ratios do not make these pipelines directly comparable.
- The original mesh arrays have 278 and 138 frames. The notebook now requires
  verified FPS and synchronized starts, checks duration and aligns timestamps.
- FLAME and MediaPipe indices have separate names. Invalid indices raise errors.
- The upper-face threshold follows the code (`y > .015`), whose stored output
  reported 1766 vertices; the old paper instead stated `.005` and 2300.
- LVE is an evaluation metric. The [official FaceFormer training implementation](https://github.com/EvelynFan/FaceFormer/blob/main/main.py)
  uses vertex MSE; the old lip-only training-loss explanation was incorrect.

## Historical observations, not new results

| Original observation | Scope |
|---|---|
| VOCASET classifier-neutral: 68.5% | Original 73-sentence sample; full deduplicated corpus: 65.2% |
| DCC classifier-neutral: 32.9% | Curated 73-dialogue text corpus |
| Author–classifier agreement: 23/73 (31.5%) | One author, context-aware labels mapped to seven categories |
| DCC upper/lip ratios: about 0.77–1.55 | Five clips tracked with MediaPipe; not metric mesh measurements |

Stored mesh outputs disagree on the upper-face mean and timing was not validated.
The old **−35%/−59% suppression** results are withheld pending a rerun. No new
empirical results are claimed. Notebook outputs were cleared so old results do not
appear next to revised code; they remain available in Git history. Original figures
are retained for traceability. Only the historical text-distribution figure is used
in the revised paper; cross-pipeline charts are not validated evidence.

The author used narrative context while the classifier sees isolated text. Low
agreement can reflect context, annotation and mapping choices, or classifier
limitations. Multiple annotators and controlled experiments are needed to separate
these explanations. Text labels do not measure recorded audiovisual emotion.

## Run the notebooks

Use Google Colab or Python with NumPy, pandas, Matplotlib, Plotly, Transformers,
PyTorch and tqdm. Module 2 also needs MediaPipe and OpenCV. Setup cells install
extra dependencies. Run cells in order.

**Module 1:** provide the official `sentencestext.zip` in the notebook's working
directory; obtain it through [VOCASET](https://voca.is.tue.mpg.de/). The notebook
reads that archive directly, not a `voca` Python package. DCC text is inline.
Files now load in sorted order; the old filesystem order was unrecorded, so the
historical seed-42 sample is not assumed to reproduce. The last cell exports all
predictions, sampled sentences, annotations, mapping and model/package metadata to
`analysis_exports/`. Use full-corpus proportions as the primary comparison.

**Module 2:** provide the actual Git LFS `.npy` contents from the
[upstream evaluation data](https://github.com/kaist-ami/Perceptual-3D-Talking-Head/tree/main/evaluation/data_MTM):

- `FaceTalk_170731_00024_TA_sentence21.npy`
- `FaceTalk_170731_00024_TA_sentence21_condition_FaceTalk_170725_00137_TA.npy`

Fill `GT_FPS`, `PRED_FPS` and `SAME_START_CONFIRMED` using verified source metadata.
The notebook stops until these are supplied; do not infer FPS from frame counts.
The first frame is a displacement reference, not an assumed neutral expression.
Mesh coordinates must use the original orientation and meters.

The independent DCC section needs `videos/hans.mp4`, `gothel.mp4`, `miguel.mp4`,
`dolores.mp4`, `moana.mp4` and the MediaPipe model downloaded by its setup cell.
Videos and original mesh data are not included. Record clip editions, scene
timecodes, hashes, FPS, shot boundaries and tracking coverage. Nose centering and
cheek scaling do not remove head rotation, perspective or tracking error.

## Checks and paper build

Regression checks require NumPy and use synthetic data only:

```sh
python -m unittest discover -s tests -v
```

They cover timestamp alignment, required metadata, duration mismatch, malformed
meshes, degenerate landmarks and FLAME/MediaPipe index confusion. They do not
validate original scientific results or substitute for an end-to-end data run.

Build from the repository root with a TeX installation:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paper paper/cinematic_speech_paper.tex
```

The PDF and LaTeX source describe the same revision.

## Next empirical work

1. Verify timing and rerun across sequences, speakers and style conditions.
2. Validate anatomical regions and use a common representation for both corpora.
3. Assess head pose, shot changes and MediaPipe tracking on stylized faces.
4. Evaluate FaceFormer on cinematic audio and report sequence-level variability.
5. Obtain independent text-only and context-aware annotations, with agreement and
   mapping-sensitivity analysis. Test causal hypotheses through controlled ablations.

## Citation

```bibtex
@misc{argese2026cinematic,
  author = {Argese, Alex},
  title = {Cinematic Speech as a Stress Test for Facial Animation Systems},
  year = {2026},
  url = {https://github.com/AlexArgese/cinematic-speech-facial-animation}
}
```
