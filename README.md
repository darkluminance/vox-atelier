# Vox Atelier

A private local voice studio for the checkpoints in `D:\models`. Double-click `start.cmd`, or run `./start.ps1` in PowerShell. The app opens at `http://127.0.0.1:8765`. Keep the terminal open while using it.

## What works

- Select a checkpoint to preload it. Only one model stays in memory at a time; selecting another unloads the previous worker.
- Qwen3 1.7B CustomVoice: named speaker plus emotion/style instruction. Qwen3 1.7B VoiceDesign: a free-form voice design prompt. Qwen3 Base: a saved reference WAV and its exact spoken transcript. Qwen3 0.6B CustomVoice: named speaker only because it ignores style instructions.
- Chatterbox Original: built-in or saved voice, expressiveness and CFG controls. Turbo/Nano: built-in or saved voice, `[laugh]`, `[chuckle]`, `[cough]` cues in script. Sampling sliders apply to Chatterbox.
- **Full render** saves a complete WAV before playback. **Sentence stream** synthesizes sentences in order, plays each finished WAV, then saves a joined final WAV. It is sentence-level progressive generation, not model-frame streaming. Sentence boundaries can affect prosody.
- **Clone from reference** is available on both Qwen Base checkpoints, all three Chatterbox checkpoints, and Sesame CSM. Save a WAV once in `data/profiles`, then select it in the app for later generations. Qwen Base and CSM need the exact spoken transcript; Chatterbox does not. Turbo and Nano need a reference longer than five seconds. Qwen CustomVoice and VoiceDesign do not accept an audio reference in their installed APIs.
- Qwen Base prepares its clone prompt once per selected profile in a loaded session; Chatterbox keeps prepared conditionals in the loaded model. CSM reuses the stored reference for conversation context. Saved profiles do not fine-tune checkpoint weights.
- Sesame CSM 1B supports built-in speaker 0 or 1, or a saved reference voice.

## Setup from a fresh clone

Install Python 3.11 and Node.js. The source repository deliberately excludes Python environments, `node_modules`, compiled frontend files, checkpoints, generated audio, and saved voice references. In PowerShell, from the project folder:

```powershell
python -m pip install -r backend/requirements.txt
npm ci
npm run build
```

Install model runtimes in separate Python 3.11 environments because Qwen and the currently tested Chatterbox/CSM stack use different Transformers versions. Install a compatible PyTorch and torchaudio build for your CPU or GPU using the [PyTorch selector](https://pytorch.org/get-started/locally/) first, then:

```powershell
python -m venv .venv-qwen
.\.venv-qwen\Scripts\python.exe -m pip install 'qwen-tts==0.1.1' 'transformers==4.57.3'
python -m venv .venv-chatterbox
.\.venv-chatterbox\Scripts\python.exe -m pip install 'chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2' 'transformers==5.2.0' 'mistral-common==1.12.0'
```

The two commands above describe the tested top-level package versions, not a complete lockfile. Install the PyTorch build inside each environment before installing the model packages. On the original machine, the environments use system site packages and share `torch 2.5.1+cu121`; a fresh machine needs its own compatible setup. `start.ps1` builds the frontend automatically if `dist` is absent. Run `./start.ps1` or double-click `start.cmd` after setup.

The server uses `TTS_PYTHON_QWEN`, `TTS_PYTHON_CHATTERBOX`, and `TTS_PYTHON_CSM` if set; otherwise it uses the Python running the server. Set `TTS_MODEL_ROOT` or pass `-ModelRoot` to the launcher to use a checkpoint root other than `D:\models`.

All model checkpoints live under `D:\models`: Qwen in `qwen3-tts/checkpoints`, Chatterbox in the Hugging Face cache, and Sesame CSM's Transformers weights and processor in `csm-1b/transformers` and `csm-1b/processor`. The app keeps generated audio, saved voice profiles, and logs in its own `data` folder. CSM loaded on this machine's RTX 3060 and produced 24 kHz WAVs with both its built-in speaker and a saved reference voice.

The UI uses the free [Arc UI](https://uiarc.dev/) components. Arc's MIT source attribution is in `ARC-UI-LICENSE.md`. Generated WAVs remain in `data/generated`. That folder also contains saved voice references and logs; it is excluded from Git. This repository is public source with no overall license grant yet; the Arc UI notice applies to its included components.

If generation fails, the app reloads the selected model automatically so you can change settings or switch between full render and sentence stream without refreshing the page.
