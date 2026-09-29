# FlyBrain × SmolLM2 × Stable Diffusion actual run

- FlyBrain upstream commit: `9191824d17871b7851645782d53d23f213ddb938`
- Connectome: **139,255 neurons / 2,698,236 edges**
- Connectome SHA-256: `fbf8d440ca1207c7573e1acdd2366f9d0beb9b533c1710f21681264f81b1cc49`
- Brain adapter: `63 → 64 → 32 → 4`, final train accuracy `1.0000`
- LLM: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Stable Diffusion: `segmind/tiny-sd`
- Total elapsed: `72.465 s`

## LLM outputs
- **sweet_food** → **sweet_food**: The fly's brain is currently engaged in processing sweet food and engaging in feeding behaviors.
- **visual_light** → **visual_light**: The fly's visual system is actively focused on detecting and processing visual information from its environment, with VIS_ME and VIS_R1R6 being the most prominent activity areas.
- **touch** → **touch**: The fly's motor control system is currently engaged in a rapid, high-frequency response to a sudden mechanical stimulus, with the most active functional groups being MECH_BRISTLE.
- **warm** → **warm**: The fly's brain is currently responding to warmth-induced changes in its thermosensory activity, with the most active functional groups being THERMO_WARM.

## Stable Diffusion outputs
- **sweet_food** → **sweet_food**: `sd_sweet_food.png` · sha256 `42749d06490edb120fec5ae2d0852a8c01f800cf675fec917f1deaa3110c9404`
- **touch** → **touch**: `sd_touch.png` · sha256 `b2e7b2b9722825893c9de4a55cc953c19562a3fe5503c4444c035a9a2db772bb`

## Training scope
The biological connectome is executed as an LIF network. A small neural adapter is actually trained on its 63-dimensional activity states. The open-source LLM and Stable Diffusion backbones are kept frozen and are conditioned by the trained adapter. This is real adapter training and real inference; it is not a claim that the fly connectome itself has been converted into an LLM or diffusion UNet.
