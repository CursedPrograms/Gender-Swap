"""FaceApp-style gender swap.

    python main.py                      # launch the web UI
    python main.py photo.jpg            # swap, writes photo_swapped.jpg
    python main.py photo.jpg -o out.jpg --to female --strength 1.2 --quality 0
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

QUALITY_STEPS = {'fast': 0, 'balanced': 100, 'best': 250}


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f'Could not read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    ok, buf = cv2.imencode(Path(path).suffix or '.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise SystemExit(f'Could not write image: {path}')
    buf.tofile(str(path))


def run_cli(args):
    from genderswap import GenderSwapper

    swapper = GenderSwapper(args.device)
    steps = QUALITY_STEPS[args.quality] if args.steps is None else args.steps

    def progress(i, total, loss):
        print(f'\r  tuning {i}/{total}  loss {loss:.4f}', end='', flush=True)

    for path in args.images:
        src = Path(path)
        print(f'{src.name}:')
        result = swapper.swap(read_rgb(src), args.to, args.strength, steps, progress)
        if steps:
            print()
        out = Path(args.output) if args.output and len(args.images) == 1 else \
            src.with_name(f'{src.stem}_swapped{src.suffix}')
        write_rgb(out, result.image)
        print(f'  detected {result.detected} -> {result.target}, saved {out}')
        if args.save_crops:
            strip = np.concatenate([result.aligned, result.reconstruction, result.edited], axis=1)
            write_rgb(out.with_name(f'{out.stem}_crops.jpg'), strip)


def build_ui(swapper):
    import gradio as gr

    def swap(image, direction, strength, quality, progress=gr.Progress()):
        if image is None:
            raise gr.Error('Upload a photo first.')
        target = {'Auto': 'auto', 'To female': 'female', 'To male': 'male'}[direction]
        steps = QUALITY_STEPS[quality.lower()]

        def report(i, total, _loss):
            progress(i / total, desc='Learning your face')

        try:
            result = swapper.swap(image, target, strength, steps, report)
        except ValueError as e:
            raise gr.Error(str(e))
        return result.image, result.edited, f'Detected {result.detected}, swapped to {result.target}.'

    with gr.Blocks(title='Gender Swap') as demo:
        gr.Markdown('# Gender Swap\nUpload a photo with a clearly visible face.')
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label='Photo', type='numpy')
                direction = gr.Radio(['Auto', 'To female', 'To male'], value='Auto', label='Swap')
                strength = gr.Slider(0.25, 2.0, value=1.0, step=0.05, label='Strength')
                quality = gr.Radio(['Fast', 'Balanced', 'Best'], value='Balanced', label='Quality',
                                   info='Balanced and Best fine-tune the model on your face so it still looks like you.')
                btn = gr.Button('Swap', variant='primary')
            with gr.Column():
                out = gr.Image(label='Result', format='png')
                face = gr.Image(label='Face close-up', format='png')
                status = gr.Markdown()
        btn.click(swap, [inp, direction, strength, quality], [out, face, status], api_name='swap')

    return demo


def run_ui(args):
    from genderswap import GenderSwapper

    build_ui(GenderSwapper(args.device)).queue().launch(inbrowser=True)


def main():
    parser = argparse.ArgumentParser(description='FaceApp-style gender swap (StyleGAN2 + e4e + PTI).')
    parser.add_argument('images', nargs='*', help='Input photo(s). Omit to launch the web UI.')
    parser.add_argument('-o', '--output', help='Output path (single input only).')
    parser.add_argument('--to', choices=['auto', 'female', 'male'], default='auto')
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--quality', choices=list(QUALITY_STEPS), default='balanced',
                        help='fast = encoder only; balanced/best = also fine-tune on the face (keeps identity).')
    parser.add_argument('--steps', type=int, help='Override the number of fine-tuning steps.')
    parser.add_argument('--save-crops', action='store_true', help='Also save aligned input / reconstruction / edit.')
    parser.add_argument('--device', help='cuda or cpu (default: cuda if available).')
    parser.add_argument('--download-models', action='store_true', help='Download any missing models and exit.')
    args = parser.parse_args()

    if args.download_models:
        from genderswap import weights
        weights.ensure_all()
    elif args.images:
        run_cli(args)
    else:
        run_ui(args)


if __name__ == '__main__':
    main()
