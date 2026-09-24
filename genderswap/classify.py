"""Face gender classifier (ViT fine-tuned on FairFace, int8 ONNX) used to pick the swap direction."""
import cv2
import numpy as np
import onnxruntime as ort


class GenderClassifier:
    # FairFace images are tight face crops; trim the looser FFHQ framing to match.
    MARGIN = 0.16

    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, aligned):
        """Return ('male' | 'female', confidence) for an FFHQ-aligned RGB crop."""
        m = int(aligned.shape[0] * self.MARGIN)
        face = aligned[m:aligned.shape[0] - m, m:aligned.shape[1] - m]
        x = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1
        logits = self.session.run(None, {self.input_name: x.transpose(2, 0, 1)[None]})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        # Labels: 0 = Female, 1 = Male
        return ('male', float(p[1])) if p[1] > p[0] else ('female', float(p[0]))
