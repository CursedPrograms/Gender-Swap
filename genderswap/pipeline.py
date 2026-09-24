"""FaceApp-style gender swap: align -> e4e invert -> (optional) pivotal tuning -> latent edit -> paste back."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .align import FaceAligner, paste_back
from .classify import GenderClassifier
from .encoder import Encoder4Editing
from .stylegan2 import Generator
from . import weights

# The gender direction only touches the coarse/middle W+ layers; the fine layers
# (skin texture, lighting, colour) stay those of the input photo.
EDIT_LAYERS = slice(0, 8)
# Distance moved along the unit gender direction at strength 1.0.
BASE_STEP = 5.0


@dataclass
class SwapResult:
    image: np.ndarray          # full photo with the swapped face pasted back
    aligned: np.ndarray        # aligned 1024x1024 input crop
    reconstruction: np.ndarray  # generator's reconstruction of the crop (before editing)
    edited: np.ndarray         # aligned 1024x1024 swapped crop
    detected: str              # 'male' or 'female', from the classifier
    target: str


def to_tensor(image, size, device):
    """RGB uint8 HxWx3 -> 1x3xSxS float in [-1, 1]."""
    t = torch.from_numpy(image).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode='area')
    return t


def to_image(tensor):
    """1x3xHxW in [-1, 1] -> RGB uint8 HxWx3."""
    t = tensor[0].detach().clamp(-1, 1).add(1).mul(127.5).round()
    return t.permute(1, 2, 0).byte().cpu().numpy()


class GenderSwapper:
    def __init__(self, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        paths = weights.ensure_all()

        self.aligner = FaceAligner(str(paths['yunet']))
        self.classifier = GenderClassifier(paths['gender_cls'])

        ckpt = torch.load(paths['e4e'], map_location='cpu', weights_only=True)
        state = ckpt['state_dict']
        self.encoder = Encoder4Editing(1024)
        self.encoder.load_state_dict({k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')})
        self.generator = Generator(1024)
        self.generator.load_state_dict({k[len('decoder.'):]: v for k, v in state.items() if k.startswith('decoder.')})
        self.latent_avg = ckpt['latent_avg'].to(self.device)
        del ckpt, state

        # The encoder is ~1 GB of weights and runs once per photo, so it stays on the CPU;
        # the GPU is left for the generator and pivotal tuning (fits in 4 GB).
        self.encoder.eval()
        self.generator.eval().to(self.device).requires_grad_(False)

        # gender.npy repeats one 512-d vector for all 18 layers; keep it as a unit vector.
        # Positive along this direction = more male.
        d = torch.from_numpy(np.load(paths['gender'])[0]).float().to(self.device)
        self.direction = d / d.norm()
        self._lpips = None

    @torch.no_grad()
    def invert(self, aligned):
        w = self.encoder(to_tensor(aligned, 256, 'cpu'))
        return w.to(self.device) + self.latent_avg

    def edit(self, w, target, strength):
        sign = 1.0 if target == 'male' else -1.0
        w = w.clone()
        w[:, EDIT_LAYERS] += sign * strength * BASE_STEP * self.direction
        return w

    def tune(self, w, aligned, steps, lr=3e-4, progress=None):
        """Pivotal Tuning (Roich et al. 2021): fine-tune the generator so that the e4e latent
        reproduces this exact face, so edits keep the person's identity. Modifies
        self.generator in place; swap() restores the pretrained weights afterwards."""
        if self._lpips is None:
            import warnings
            import lpips
            with warnings.catch_warnings():  # lpips uses deprecated torchvision/torch.load arguments
                warnings.simplefilter('ignore')
                self._lpips = lpips.LPIPS(net='alex', verbose=False).to(self.device).eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)

        gen = self.generator.requires_grad_(True)
        opt = torch.optim.Adam(gen.parameters(), lr=lr)
        target = to_tensor(aligned, 1024, self.device)
        target_256 = F.interpolate(target, size=(256, 256), mode='area')

        for i in range(steps):
            out = gen(w)
            out_256 = F.interpolate(out, size=(256, 256), mode='area')
            loss = self._lpips(out_256, target_256).mean() + F.mse_loss(out_256, target_256)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if progress:
                progress(i + 1, steps, loss.item())
            if loss.item() < 0.02:
                break

        gen.requires_grad_(False)

    def swap(self, image, target='auto', strength=1.0, tune_steps=0, progress=None):
        """Gender-swap the largest face in `image` (RGB uint8). `target` is 'auto', 'male' or 'female'."""
        landmarks = self.aligner.detect(image)
        if landmarks is None:
            raise ValueError('No face found in the image.')
        aligned, matrix = self.aligner.align(image, landmarks, 1024)

        w = self.invert(aligned)
        detected, _ = self.classifier(aligned)
        if target == 'auto':
            target = 'female' if detected == 'male' else 'male'

        pretrained = None
        if tune_steps > 0:
            pretrained = {k: v.cpu().clone() for k, v in self.generator.state_dict().items()}
        try:
            if pretrained is not None:
                self.tune(w, aligned, tune_steps, progress=progress)
            with torch.no_grad():
                recon = self.generator(w)
                edited = self.generator(self.edit(w, target, strength))
        finally:
            if pretrained is not None:
                self.generator.load_state_dict(pretrained)
                torch.cuda.empty_cache()

        edited_img = to_image(edited)
        return SwapResult(
            image=paste_back(image, edited_img, matrix),
            aligned=aligned,
            reconstruction=to_image(recon),
            edited=edited_img,
            detected=detected,
            target=target,
        )
