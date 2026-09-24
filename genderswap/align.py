"""Face detection and FFHQ-style alignment (the crop StyleGAN2/e4e expect), plus paste-back."""
import cv2
import numpy as np


class FaceAligner:
    def __init__(self, yunet_path, score_threshold=0.7):
        # OpenCV 5's new DNN engine warns that it can't set the (default, CPU) target
        # YuNet asks for; it runs on the CPU either way, so hide just that warning.
        log = cv2.utils.logging
        level = log.getLogLevel()
        log.setLogLevel(log.LOG_LEVEL_ERROR)
        try:
            self.detector = cv2.FaceDetectorYN.create(yunet_path, '', (320, 320), score_threshold, 0.3, 5000)
        finally:
            log.setLogLevel(level)

    def detect(self, image):
        """Return five landmarks (eye_l, eye_r, nose, mouth_l, mouth_r) of the largest face, or None.

        `image` is an RGB uint8 array. Left/right are in image coordinates.
        """
        h, w = image.shape[:2]
        # YuNet is fast but loses small faces in huge photos; run it on a bounded copy.
        scale = min(1.0, 1280 / max(h, w))
        small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else image
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        pts = face[4:14].reshape(5, 2).astype(np.float64) / scale
        eyes = sorted(pts[0:2].tolist())
        mouth = sorted(pts[3:5].tolist())
        return np.array([eyes[0], eyes[1], pts[2], mouth[0], mouth[1]])

    @staticmethod
    def ffhq_quad(landmarks):
        """The oriented crop square used to build FFHQ (see the FFHQ dataset's alignment script)."""
        eye_left, eye_right, _, mouth_left, mouth_right = landmarks
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        # top-left, bottom-left, bottom-right, top-right
        return np.stack([c - x - y, c - x + y, c + x + y, c + x - y])

    @staticmethod
    def crop_matrix(quad, size):
        """Affine matrix mapping original-image coordinates into the aligned `size` x `size` crop."""
        dst = np.float32([[0, 0], [0, size], [size, size]])
        return cv2.getAffineTransform(np.float32(quad[:3]), dst)

    def align(self, image, landmarks, size=1024):
        """Warp `image` to an aligned crop. Returns (crop, matrix) where matrix maps image -> crop."""
        quad = self.ffhq_quad(landmarks)
        matrix = self.crop_matrix(quad, size)

        # Area-downsample first when shrinking a lot, so the crop isn't aliased.
        qsize = np.hypot(*(quad[3] - quad[0]))
        shrink = int(np.floor(qsize / size * 0.5))
        src = image
        warp = matrix
        if shrink > 1:
            h, w = image.shape[:2]
            src = cv2.resize(image, (w // shrink, h // shrink), interpolation=cv2.INTER_AREA)
            warp = self.crop_matrix(quad / shrink, size)

        crop = cv2.warpAffine(src, warp, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        return crop, matrix


def paste_back(original, crop, matrix, feather=0.08):
    """Blend an edited aligned crop back into the original photo with a soft-edged mask."""
    h, w = original.shape[:2]
    size = crop.shape[0]

    # Soft square in crop space: 1 in the middle, fading to 0 at the borders.
    ramp = np.clip(np.minimum(np.arange(size), size - 1 - np.arange(size)) / (feather * size), 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep
    mask = np.outer(ramp, ramp).astype(np.float32)

    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    warped = cv2.warpAffine(crop, matrix, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT)
    warped_mask = cv2.warpAffine(mask, matrix, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT)[..., None]

    out = original.astype(np.float32) * (1 - warped_mask) + warped.astype(np.float32) * warped_mask
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
