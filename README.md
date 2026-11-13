# small-csm

Fast, lightweight packaging of Sesame **CSM-1B** (fp16) with CUDA graph optimization for real-time speech synthesis.

## Requirements

- NVIDIA GPU with CUDA (~3.7 GB VRAM)
- Hugging Face login with the `sesame/csm-1b` licence accepted

## Install

```bash
pip install .
# or with uv:
uv pip install .
```

## Usage

CLI:

```bash
small-csm "Hey, how's it going?" -o out.wav
```

Python:

```python
from small_csm import CSMSpeaker

spk = CSMSpeaker()
for chunk in spk.stream("Hey, how's it going?"):
    ...  # float32 mono, 24 kHz

# or synthesize full clip
audio = spk.synthesize("Wait, seriously? Congratulations!")
```
