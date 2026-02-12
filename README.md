# AION Voice 🔊🤖

AION Voice is an experimental, modular voice-synthesis and interactive auditory agent project. It combines spiking neural networks (LSM), high-dimensional computing (HDC), attention/global workspace mechanisms, and simple generative readouts to learn and mimic short voice patterns in real-time.

## Quick Overview 🚀
- **Purpose**: Real-time audio perception, memory formation, and short-sequence generative replay ("dreaming").
- **Main idea**: Use a Liquid State Machine (LSM) as a temporal encoder and a simple ridge-regression readout to synthesize next-frame mel features.
- **Target audience**: Researchers, hobbyists, and students exploring neuromorphic-inspired audio models and creative AI sound agents.

## Features ✨
- Lightweight LSM-based online perception and generative playback
- Simple training pipeline (collect neural states -> ridge regression solution)
- Live microphone interaction with "dreaming"/replay when idle
- Visdom-based dashboard for visualization (optional)
- Modular code: `src/` modules + runnable scripts in `scripts/`

## Repository structure 🗂️
- `src/` — core modules (lsm, adapter, gwt, mhm/hopfield, dashboard, utils)
- `scripts/` — example scripts: `interaction_loop.py` (live agent), `train_generative.py` (training readout)
- `datasets/` — expected location for .wav files used for preloading and training
- `README.md` — this file
- `requirements.txt` — Python dependencies

## Requirements ⚙️
- Python 3.8+
- PyTorch
- librosa, sounddevice, numpy, visdom (optional for dashboard)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart — Run the live interaction agent 🎤
1. Plug in a microphone and ensure your system audio is configured.
2. (Optional) Put some .wav files under `datasets/` to preload memories.
3. Run the agent (GPU recommended but not strictly required):

```bash
python scripts/interaction_loop.py
```

**What to expect**:
- The agent listens and converts short mel frames into spike patterns.
- If it hears speech it updates internal memories. If silent for several seconds it enters a "dream" mode and replays generated snippets.

## Training the readout (offline) 🏗️
To train the LSM readout weights from a collection of audio files, use the training script which collects states S and targets Y and solves a ridge regression:

```bash
python scripts/train_generative.py --data /path/to/datasets --limit 200
```

Trained weights are saved to the path specified in `src/config.py` (default: `lsm_readout_weights.pt`).

## Configuration 🔧
See `src/config.py` for tunable parameters such as:
- AUDIO_SR, HOP_LENGTH, OBS_SHAPE (mel bins)
- LSM_N_NEURONS and other network hyperparameters
- WEIGHTS_PATH and DEVICE detection

## Development tips 🛠️
- The code adds the project root to `sys.path` in scripts; run scripts from the repository root.
- If visdom dashboard is not running, the agent will continue but skip visualization.
- For reproducible training, set seeds where necessary (SEED in config).

## Contributing 🤝
Contributions welcome! A suggested workflow:
1. Fork the repo and create a descriptive branch: `feature/your-feature` or `fix/bug-...`
2. Add tests/examples if possible.
3. Open a pull request with a clear description and screenshots or recordings when relevant.

## Notes & Limitations ⚠️
- This is experimental and research-oriented — audio quality is basic and design favors interpretability and simplicity.
- The generative path is a simple frame-by-frame readout (not a high-fidelity neural vocoder).
- Use responsibly: do not use generated audio to impersonate people without consent.

## Contact & Credits ✉️
- **Author**: lkcfqy
- **Repo**: https://github.com/lkcfqy/AION_voice

## License 📄
This project is released under the MIT License. See the LICENSE file for details.
